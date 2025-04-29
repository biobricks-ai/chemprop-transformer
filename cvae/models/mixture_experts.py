import pathlib, torch, torch.nn as nn, torch.nn.functional as F
from cvae.models.multitask_transformer import PositionalEncoding, generate_custom_subsequent_mask, MultitaskTransformer, SelfiesPropertyValTokenizer
import cvae.models.multitask_transformer as mt
import cvae.utils
import json
import logging
import torch
import random
from collections import defaultdict

class PropertyLossTracker:
    """
    Tracks property frequencies using EMA and calculates stratified loss
    based on inverse frequency weighting.

    Simplified version: removes loss-based boosting and EMA loss tracking.
    Weighting is based solely on frequency, controlled by max_weight.

    Corrected EMA update: Properly decays frequencies for properties not
    present in the current batch.
    """
    def __init__(self, power=0.5, ema_decay=0.99, min_frequency=1e-5, warmup_batches=1000, slow_warmup=3000, frozen=False, max_weight=30, ramp_batches=200):
        """
        Args:
            ema_decay (float): Exponential moving average decay rate for frequencies.
            min_frequency (float): Minimum frequency to use in weighting to avoid division by zero or huge weights.
            warmup_batches (int): Number of batches during which weights are 1.0.
            max_weight (float): Maximum raw weight assigned to any property before normalization.
                                Controls the maximum boost for rare properties. Set close to 1.0
                                for subtle weighting (e.g., 1.001 or 1.01).
            ramp_batches (int): Number of batches over which the frequency weighting ramps up after warmup.
        """
        self.power = power
        self.ema_decay = ema_decay
        self.min_frequency = min_frequency
        self.warmup_batches = warmup_batches
        self.max_weight = max_weight # Controls how much boost rare properties get
        self.ramp_batches = ramp_batches

        # State variables - NOT registered as buffers, will be lost on save/load
        self.property_frequencies = defaultdict(lambda: self.min_frequency)
        self.batch_counter = 0         # Tracks number of batches processed
        self.is_started = False
        self.slow_warmup = slow_warmup
        self.frozen = frozen

    def update(self, property_ids, token_losses, device):
        """
        Updates frequency EMAs and calculates the stratified loss for the current batch.

        Args:
            property_ids (torch.Tensor): Tensor of property IDs corresponding to each loss in token_losses.
                                         Shape: (num_value_tokens_in_batch,)
            token_losses (torch.Tensor): Tensor of individual CrossEntropyLoss values for value tokens.
                                         Shape: (num_value_tokens_in_batch,)
            device (torch.device): The device the tensors are on.

        Returns:
            tuple: (stratified_loss, unweighted_loss)
                   stratified_loss (torch.Tensor): The frequency-weighted mean loss for the batch.
                   unweighted_loss (torch.Tensor): The simple mean loss for the batch (for logging/comparison).
        """
        self.batch_counter += 1
        
        # If no value tokens in this batch, return zero loss
        if property_ids.numel() == 0:
            unweighted_loss = token_losses.mean() if token_losses.numel() > 0 else torch.tensor(0.0, device=device)
            # Decay frequencies for all known properties even if no value tokens are present
            for prop_id in self.property_frequencies:
                 self.property_frequencies[prop_id] *= self.ema_decay
            return torch.tensor(0.0, device=device), unweighted_loss

        unique_props_in_batch = property_ids.unique().tolist()
        property_losses = []
        property_weights = [] # Raw weights before normalization

        # Calculate mean loss for each unique property in the batch
        property_loss_dict = {}
        for prop in unique_props_in_batch:
             prop_mask = (property_ids == prop)
             if prop_mask.any():
                  property_loss_dict[prop] = token_losses[prop_mask].mean()


        if not self.frozen:
            
            # 1. Decay EMA for ALL properties currently tracked
            for prop_id in self.property_frequencies:
                self.property_frequencies[prop_id] *= self.ema_decay

            # 2. Update EMA for properties present in the current batch
            current_batch_total_value_tokens = property_ids.numel()
            for prop_id in unique_props_in_batch:
                prop_mask = (property_ids == prop_id)
                current_freq_in_batch = prop_mask.float().sum().item() / (current_batch_total_value_tokens + 1e-6) # Proportion in batch

                if prop_id not in self.property_frequencies:
                    # Initialize new property frequency
                    self.property_frequencies[prop_id] = current_freq_in_batch
                else:
                    # Blend current batch frequency with decayed EMA
                    self.property_frequencies[prop_id] += (1 - self.ema_decay) * current_freq_in_batch

        # Calculate weights for properties *present in this batch* based on their *global EMA frequency*
        for prop_id in unique_props_in_batch:
            # Ensure the property was actually processed and has a loss entry
            if prop_id in property_loss_dict:
                property_losses.append(property_loss_dict[prop_id])

                # Use the EMA frequency (clamped at min_frequency)
                freq = max(self.property_frequencies.get(prop_id, self.min_frequency), self.min_frequency) # Use .get with default for safety

                # Calculate raw weight based on EMA frequency
                if self.batch_counter > self.warmup_batches and not self.is_started:
                    logging.info(f"Ending propertylosstracker warmup phase")
                    self.is_started = True

                if self.batch_counter <= self.warmup_batches:
                    raw_weight = 1.0
                elif self.batch_counter <= self.slow_warmup :
                    # Simple inverse frequency weighting
                    raw_weight = 1.0 / (freq + 1e-6) ** self.power # Add epsilon for numerical stability

                    # Apply ramp-up after warmup
                    ramp = min(1.0, (self.batch_counter - self.warmup_batches) / self.ramp_batches)
                    # Ramp the weight from 1.0 (at end of warmup) towards the calculated raw_weight
                    raw_weight = 1.0 + ramp * (raw_weight - 1.0)

                    # Apply max_weight cap
                    raw_weight = min(self.max_weight, raw_weight) * (self.batch_counter / self.slow_warmup) + 1.0 * (1 - self.batch_counter / self.slow_warmup)
                else:
                    # Simple inverse frequency weighting
                    raw_weight = 1.0 / (freq + 1e-6) ** self.power # Add epsilon for numerical stability

                    # Apply ramp-up after warmup
                    ramp = min(1.0, (self.batch_counter - self.warmup_batches) / self.ramp_batches)
                    # Ramp the weight from 1.0 (at end of warmup) towards the calculated raw_weight
                    raw_weight = 1.0 + ramp * (raw_weight - 1.0)

                    # Apply max_weight cap
                    raw_weight = min(self.max_weight, raw_weight)

                property_weights.append(raw_weight)


        if random.random() < 0.01:
            logging.info(f"batch {self.batch_counter} property_weights min: {min(property_weights)}, max: {max(property_weights)}")

        # If no properties were processed (e.g., due to filtering, though none implemented here)
        if not property_losses:
            unweighted_loss = token_losses.mean() if token_losses.numel() > 0 else torch.tensor(0.0, device=device)
            return torch.tensor(0.0, device=device), unweighted_loss

        # Convert lists to tensors
        property_losses = torch.stack(property_losses) # Shape: (num_unique_props_in_batch,)
        property_weights = torch.tensor(property_weights, device=device) # Shape: (num_unique_props_in_batch,)

        # Normalize weights for the properties in THIS batch so they sum to 1
        # This determines the relative contribution of each property's mean loss
        property_weights = property_weights / (property_weights.sum() + 1e-6)

        # Calculate the stratified loss (weighted sum of property mean losses)
        stratified_loss = (property_losses * property_weights).sum()

        # Calculate the unweighted mean loss across all value tokens (for comparison/logging)
        unweighted_loss = token_losses.mean()

        # Return both the stratified and unweighted loss
        return stratified_loss, unweighted_loss



class MoE(nn.Module):

    def __init__(self, tokenizer, num_experts=2, k=2, hdim=32, nhead=8, dim_feedforward=32, noise_factor=.1, 
                 dropout_rate=.1, balance_loss_weight=1.0, diversity_loss_weight=10, expert_layers=4, output_size=None):
        super().__init__()
        self.tokenizer: SelfiesPropertyValTokenizer = tokenizer
        self.expert_layers = expert_layers
        self.output_size = output_size if output_size is not None else tokenizer.vocab_size
        mkexpert = lambda: MultitaskTransformer(tokenizer, hdim, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=expert_layers, output_size=self.output_size)
        self.experts = nn.ModuleList([mkexpert() for _ in range(num_experts)])
        self.gating_network = MultitaskTransformer(tokenizer, hdim, output_size=num_experts, nhead=nhead)
        self.hdim = hdim
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.noise_factor = noise_factor
        self.balance_loss_weight = balance_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.balance_loss = 0.0
        self.diversity_loss = 0.0
        self.dropout_rate = dropout_rate
        self.current_train_step = 0
        self.num_experts = num_experts
        self.k = k
        
        
        self.property_loss_tracker = PropertyLossTracker()


    def compute_gating_distribution(self, input, teach_forcing, expert_mask=None):
        gating_scores = self.gating_network(input, teach_forcing)
        if self.training and self.noise_factor > 0:
            gating_scores = gating_scores + torch.randn_like(gating_scores) * self.noise_factor

        if expert_mask is not None:
            expert_mask = expert_mask.view(1, 1, -1)
            gating_scores = gating_scores * expert_mask

        gating_distribution = F.softmax(gating_scores, dim=-1)
        return gating_distribution

    def compute_balance_loss(self, gating_distribution):
        eps = 1e-8
        expert_usage = gating_distribution.sum(dim=(0, 1))  # [num_experts]
        total_usage = expert_usage.sum()
        if total_usage < eps:
            return torch.tensor(0.0, device=gating_distribution.device)

        expert_usage = expert_usage / total_usage  # normalized

        target_usage = torch.ones_like(expert_usage) / self.num_experts
        balance_loss = F.kl_div(
            expert_usage.log(),
            target_usage,
            reduction='batchmean',
            log_target=False
        )
        
        # print min and max of expert_usage
        # print(f"min expert_usage: {expert_usage.min().item()}, max expert_usage: {expert_usage.max().item()}")

        return self.balance_loss_weight * balance_loss

    def compute_diversity_loss(self, expert_outputs):
        # expert_outputs: list of [batch, seq_len, vocab] tensors, one per expert
        valid_outputs = [eo for eo in expert_outputs if eo is not None]
        if len(valid_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)

        # For each expert's output tensor [batch, seq_len, vocab], compute mean across batch and sequence dimensions
        # This gives us a single vector per expert of size [vocab], then stack all expert vectors into [num_experts, vocab]
        flattened = torch.stack([eo.mean(dim=(0, 1)) for eo in valid_outputs], dim=0)  # [E, V]

        # Normalize to unit length — if any vector is all zeros, skip
        norm = torch.norm(flattened, p=2, dim=1, keepdim=True)  # [E, 1]
        valid_mask = norm.squeeze() > 1e-6
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=flattened.device)

        flattened = flattened[valid_mask]
        flattened = flattened / (norm[valid_mask] + 1e-8)  # [E_filtered, D]

        # Cosine similarity matrix: [E_filtered, E_filtered]
        similarity = torch.matmul(flattened, flattened.T)
        similarity = torch.clamp(similarity, -1.0, 1.0)

        # Keep only upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(similarity), diagonal=1)  # [E, E]
        n_pairs = mask.sum()

        # Penalize similarity → encourage diversity
        diversity_loss = (similarity ** 2 * mask).sum() / (n_pairs + 1e-8)
        diversity_loss = torch.tanh(diversity_loss)

        return self.diversity_loss_weight * diversity_loss

    def forward(self, input, teach_forcing=None):
        
        B, T = input.shape
        E = self.num_experts
        V = self.tokenizer.vocab_size

        # Compute expert outputs: List of [B, T, V]
        expert_outputs = [expert(input, teach_forcing) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # [B, T, V, E]

        # Compute gating scores: [B, T, E]
        gating_scores = self.gating_network(input, teach_forcing)
        if self.training and self.noise_factor > 0:
            gating_scores = gating_scores + torch.randn_like(gating_scores) * self.noise_factor

        # For balance loss
        soft_distribution = F.softmax(gating_scores, dim=-1)

        # Gumbel noise (optionally add more control here)
        if self.training and self.noise_factor > 0:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(gating_scores) + 1e-9) + 1e-9)
            gating_scores = gating_scores + gumbel_noise * self.noise_factor

        # Get top-2 experts: indices and values
        topk_values, topk_indices = torch.topk(gating_scores, k=self.k, dim=-1)  # [B, T, 2]

        # Softmax over top-k only
        topk_weights = F.softmax(topk_values, dim=-1)  # [B, T, 2]

        # Create one-hot mask to select top-k experts
        topk_mask = F.one_hot(topk_indices, num_classes=E).float()  # [B, T, 2, E]

        # Distribute weights into full [B, T, E] tensor
        routing_weights = (topk_weights.unsqueeze(-1) * topk_mask).sum(dim=2)  # [B, T, E]

        # Apply routing weights to expert outputs
        output = (stacked_outputs * routing_weights.unsqueeze(2)).sum(dim=-1)  # [B, T, V]

        # Auxiliary losses
        if self.training:
            self.balance_loss = self.compute_balance_loss(soft_distribution)
            self.diversity_loss = self.compute_diversity_loss(expert_outputs)

        return output
    
    def build_stratified_lossfn(self):
        ignore_index = self.tokenizer.pad_idx
        ce_lossfn = mt.LabelSmoothingCrossEntropySequence(epsilon_ls=.05)
        
        def lossfn(parameters, logits, output):
            batch_size, vocab_size, seq_len = logits.size()

            token_indices = torch.arange(seq_len, device=logits.device)
            is_value_position = (token_indices >= 2) & (token_indices % 2 == 0)
            is_value_position = is_value_position.unsqueeze(0).expand(batch_size, seq_len)

            is_not_pad = (output != ignore_index)
            is_value_position_runtime = is_value_position.to(output.device)
            
            final_mask = is_value_position_runtime & is_not_pad
            
            logits = logits.transpose(1, 2).contiguous()
            logits_selected = logits[final_mask]
            output_selected = output[final_mask]

            token_losses = ce_lossfn(logits_selected, output_selected)

            return token_losses

    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        torch.save(self.state_dict(), path / "mtransformer.pt")

        config = {
            "num_experts": self.num_experts,
            "k": self.k,
            "hdim": self.hdim,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "noise_factor": self.noise_factor,
            "dropout_rate": self.dropout_rate,
            "balance_loss_weight": self.balance_loss_weight,
            "diversity_loss_weight": self.diversity_loss_weight,
            "expert_layers": self.expert_layers,
            "output_size": self.output_size
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return path

    @staticmethod
    def load(dirpath=pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        config = json.load(open(dirpath / "config.json"))
        model = MoE(tokenizer, **config)
        state_dict = torch.load(dirpath / 'mtransformer.pt', map_location='cpu')

        model.load_state_dict(state_dict)
        model.eval()
        return model
