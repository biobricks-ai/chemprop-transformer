import pathlib, torch, torch.nn as nn, torch.nn.functional as F
from cvae.models.multitask_transformer import PositionalEncoding, generate_custom_subsequent_mask, MultitaskTransformer, SelfiesPropertyValTokenizer
import cvae.utils

class MoE(nn.Module):
    
    def __init__(self, tokenizer, num_experts=2, hdim=256, noise_factor=.1, top_k_percent=.25):
        super().__init__()
        self.tokenizer : SelfiesPropertyValTokenizer = tokenizer
        self.experts = nn.ModuleList([MultitaskTransformer(tokenizer, hdim) for _ in range(num_experts)])
        self.gating_network = MultitaskTransformer(tokenizer, hdim, output_size=num_experts)
        self.noise_factor = noise_factor
        self.top_k = int(num_experts * top_k_percent)

    def forward(self, input, teach_forcing):
        # Step 1: Calculate the outputs from each expert B x SEQUENCE x TOKENS
        expert_outputs = [expert(input, teach_forcing) for expert in self.experts]
        
        # Step 2: Stack outputs for gating NUM_EXPERTS x B x SEQUENCE x TOKENS
        stacked_outputs = torch.stack(expert_outputs, dim=0)
        
        # Step 3: Get gating scores
        gating_scores = self.gating_network(input, teach_forcing)
        
        # Add noise during training for better load balancing
        if self.training:
            noise = torch.randn_like(gating_scores) * self.noise_factor
            gating_scores = gating_scores + noise
        
        # Step 4: Get top-k experts per token
        top_k_scores, top_k_indices = torch.topk(gating_scores, self.top_k, dim=-1)
        
        # Create a mask for non-selected experts
        mask = torch.zeros_like(gating_scores).scatter_(-1, top_k_indices, 1.0)
        
        # Apply softmax only on selected experts
        gating_distribution = F.softmax(top_k_scores, dim=-1)
        
        # Distribute the gating weights back to full expert dimension
        gating_distribution = torch.zeros_like(gating_scores).scatter_(-1, top_k_indices, gating_distribution)
        
        # Step 5: Combine outputs from selected experts based on gating distribution
        combined_output = torch.einsum('ebsv,bsg->bsv', stacked_outputs, gating_distribution)
        return combined_output
    
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
        
        model = MoE(tokenizer, num_experts, hdim)
        model.load_state_dict(torch.load(dirpath / 'mtransformer.pt'))
        model.eval()
        return model