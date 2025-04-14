import pathlib, torch, torch.nn as nn, torch.nn.functional as F
from cvae.models.multitask_transformer import PositionalEncoding, generate_custom_subsequent_mask, MultitaskTransformer, SelfiesPropertyValTokenizer
import cvae.models.multitask_transformer as mt
import cvae.utils

class MoE(nn.Module):
    
    def __init__(self, tokenizer, num_experts=2, hdim=32, nhead=8, dim_feedforward=32, noise_factor=.1, top_k_percent=.5, 
                 dropout_rate=.1, balance_loss_weight=1.0, diversity_loss_weight=0.1, ema_decay=0.99):
        super().__init__()
        self.tokenizer : SelfiesPropertyValTokenizer = tokenizer
        mkexpert = lambda: MultitaskTransformer(tokenizer, hdim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.experts = nn.ModuleList([mkexpert() for _ in range(num_experts)])
        self.gating_network = MultitaskTransformer(tokenizer, hdim, output_size=num_experts)
        self.hdim = hdim
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.noise_factor = noise_factor
        self.top_k = int(num_experts * top_k_percent)
        self.balance_loss_weight = balance_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.balance_loss = 0.0
        self.diversity_loss = 0.0
        self.dropout_rate = dropout_rate
        self.current_train_step = 0
        self.num_experts = num_experts
        self.ema_decay = ema_decay
        self.register_buffer('expert_usage_ema', torch.ones(num_experts) / num_experts)

    def compute_expert_outputs(self, input, teach_forcing):
        """Compute outputs from all experts with dropout during training."""
        expert_outputs = []
        expert_mask = torch.ones(len(self.experts), device=input.device)

        for i, expert in enumerate(self.experts):
            if self.training and torch.rand(1).item() < self.dropout_rate:
                expert_outputs.append(None)  # Drop this expert entirely
                expert_mask[i] = 0
            else:
                expert_outputs.append(expert(input, teach_forcing))
        return expert_outputs, expert_mask
    
    def compute_gating_distribution(self, input, teach_forcing, expert_mask):
        """Compute gating distribution with noise, masking, and top-k selection."""
        # Get gating scores
        gating_scores = self.gating_network(input, teach_forcing)
        
        # Add noise during training for better load balancing
        if self.training:
            noise = torch.randn_like(gating_scores) * self.noise_factor * 2.0
            gating_scores = gating_scores + noise
        
        # Apply expert mask to gating scores (set dropped experts to -inf)
        if self.training:
            # Expand expert_mask to match gating_scores dimensions
            expanded_mask = expert_mask.expand(gating_scores.size(0), gating_scores.size(1), -1)
            # Set scores for dropped experts to -inf so they won't be selected in top-k
            gating_scores = torch.where(expanded_mask > 0, gating_scores, 
                                      torch.tensor(-float('inf'), device=gating_scores.device))
        
        # Get top-k experts per token
        top_k_scores, top_k_indices = torch.topk(gating_scores, 
                                               min(self.top_k, expert_mask.sum().int().item()), 
                                               dim=-1)
        
        # Apply softmax only on selected experts
        gating_distribution = F.softmax(top_k_scores, dim=-1)
        
        # Distribute the gating weights back to full expert dimension
        full_gating_distribution = torch.zeros_like(gating_scores).scatter_(-1, top_k_indices, gating_distribution)
        
        return full_gating_distribution, top_k_indices
    
    def compute_balance_loss(self, gating_distribution):
        """Compute smoothed balance loss using EMA of expert usage."""
        eps = 1e-8
        
        # Current batch expert usage
        expert_usage = gating_distribution.sum(dim=(0, 1))
        total_usage = expert_usage.sum()
        if total_usage < eps:
            return torch.tensor(0.0, device=gating_distribution.device)
            
        expert_usage = expert_usage / total_usage
        
        # Update exponential moving average
        with torch.no_grad():
            self.expert_usage_ema = (
                self.ema_decay * self.expert_usage_ema + 
                (1 - self.ema_decay) * expert_usage
            )
        
        # Use smoothed expert usage for loss calculation
        smoothed_usage = self.expert_usage_ema + eps
        smoothed_usage = smoothed_usage / smoothed_usage.sum()
        
        # Target is uniform distribution
        target_usage = torch.ones_like(smoothed_usage) / self.num_experts
        
        # Compute smooth L1 loss (Huber loss) instead of KL
        balance_loss = F.smooth_l1_loss(
            smoothed_usage,
            target_usage,
            beta=0.1,  # Controls the smoothness transition point
            reduction='sum'
        )
        
        # Debug print
        print(f"Expert usage fractions: {smoothed_usage.detach().cpu().numpy()}")
        print(f"Max expert usage: {smoothed_usage.max().item():.4f}, "
              f"Min usage: {smoothed_usage.min().item():.4f}")
        print(f"Balance loss: {balance_loss.item():.4f}")
        
        return self.balance_loss_weight * balance_loss

    def compute_diversity_loss(self, expert_outputs):
        """Compute diversity loss to encourage expert specialization."""
        # Filter out dropped experts
        valid_outputs = [eo for eo in expert_outputs if eo is not None]
        if len(valid_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)

        # Add small noise to prevent exact zeros
        eps = 1e-6
        
        stacked_outputs = torch.stack(valid_outputs, dim=0)  # [num_valid_experts, batch, seq, vocab]
        reshaped = stacked_outputs.view(len(valid_outputs), -1, stacked_outputs.size(-1))
        
        # Normalize with epsilon and clipping for stability
        norms = torch.norm(reshaped, p=2, dim=-1, keepdim=True).clamp(min=eps)
        normalized = reshaped / norms
        
        # Compute cosine similarity matrix
        similarity = torch.matmul(normalized, normalized.transpose(-2, -1))
        
        # Clip similarities to prevent extreme values
        similarity = torch.clamp(similarity, -1.0, 1.0)
        
        # Only consider upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(similarity), diagonal=1)
        similarity = similarity * mask
        
        # Average the similarities and scale to [0,1] range
        n_pairs = mask.sum()
        if n_pairs == 0:
            return torch.tensor(0.0, device=similarity.device)
        
        # Scale down the loss significantly
        diversity_loss = (similarity ** 2).sum() / n_pairs
        diversity_loss = torch.tanh(diversity_loss)  # Bound between 0 and 1
        
        return self.diversity_loss_weight * diversity_loss

    def forward(self, input, teach_forcing):
        expert_outputs, expert_mask = self.compute_expert_outputs(input, teach_forcing)

        # Determine output shape from a non-dropped expert
        example_output = next(eo for eo in expert_outputs if eo is not None)
        dummy = torch.zeros_like(example_output)

        stacked_outputs = torch.stack([
            eo if eo is not None else dummy
            for eo in expert_outputs
        ], dim=0)  # [num_experts, batch, seq, vocab]

        # shape: [batch, seq, num_experts]
        gating_distribution, _ = self.compute_gating_distribution(input, teach_forcing, expert_mask)
        
        # Check for NaNs in gating distribution
        if torch.isnan(gating_distribution).any():
            gating_distribution = torch.ones_like(gating_distribution) / self.num_experts
            
        combined_output = torch.einsum('ebsv,bse->bsv', stacked_outputs, gating_distribution)

        if self.training:
            self.balance_loss = self.compute_balance_loss(gating_distribution)
            self.diversity_loss = self.compute_diversity_loss(expert_outputs)
            
            # Safety check for NaN losses
            if torch.isnan(self.balance_loss):
                self.balance_loss = torch.tensor(0.0, device=input.device)
            if torch.isnan(self.diversity_loss):
                self.diversity_loss = torch.tensor(0.0, device=input.device)

        return combined_output
    
    def build_lossfn(self):
        mtlossfn = mt.MultitaskTransformer.focal_lossfn(ignore_index=self.tokenizer.pad_idx)
        def lossfn(param, logit, output):
            main_loss = mtlossfn(param, logit, output)
            total_loss = main_loss + self.balance_loss + self.diversity_loss
            
            # Debug logging
            print(f"Loss components - Main: {main_loss:.4f}, Balance: {self.balance_loss:.4f}, Diversity: {self.diversity_loss:.4f}")
            print(f"Total loss: {total_loss:.4f}")
            
            return total_loss
        return lossfn
    
    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        
        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        torch.save(self.state_dict(), path / "mtransformer.pt")
        return path
    
    @staticmethod
    def load(dirpath = pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        state_dict = torch.load(dirpath / 'mtransformer.pt')
        
        hdim = state_dict['experts.0.encoder.layers.0.self_attn.in_proj_weight'].shape[0] // 3
        num_experts = max(int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('experts.')) + 1
        dim_feedforward = state_dict['experts.0.encoder.layers.0.linear1.weight'].shape[0]
        nhead = state_dict['experts.0.encoder.layers.0.self_attn.in_proj_weight'].shape[0] // (3 * hdim)
        
        model = MoE(tokenizer, num_experts=num_experts, hdim=hdim, 
                   dim_feedforward=dim_feedforward, nhead=nhead)
        model.load_state_dict(torch.load(dirpath / 'mtransformer.pt'))
        model.eval()
        return model