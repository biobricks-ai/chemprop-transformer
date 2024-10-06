import pathlib, torch, torch.nn as nn, torch.nn.functional as F
from cvae.models.multitask_transformer import PositionalEncoding, generate_custom_subsequent_mask, MultitaskTransformer, SelfiesPropertyValTokenizer
import cvae.utils

class MoE(nn.Module):
    
    def __init__(self, tokenizer, num_experts=2, hdim=256):
        super().__init__()
        self.tokenizer : SelfiesPropertyValTokenizer = tokenizer
        self.experts = nn.ModuleList([MultitaskTransformer(tokenizer, hdim) for _ in range(num_experts)])
        self.gating_network = MultitaskTransformer(tokenizer, hdim, output_size=num_experts)

    def forward(self, input, teach_forcing):
        # Assuming input is already the output of some layer or initial embedding
        # You might want to adapt this to your actual input processing
        
        # Step 1: Calculate the outputs from each expert B x SEQUENCE x TOKENS
        expert_outputs = [expert(input, teach_forcing) for expert in self.experts]
        
        # Step 2: Stack outputs for gating NUM_EXPERTS x B x SEQUENCE x TOKENS
        stacked_outputs = torch.stack(expert_outputs, dim=0)
        
        # Step 3: Use the last output to decide gating (simple version)
        gating_scores = self.gating_network(input,teach_forcing)
        gating_distribution = F.softmax(gating_scores, dim=-1) # N X SEQUENCE X NUM_EXPERTS
        
        # Step 4: Combine outputs from all experts based on gating distribution
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
        model = MoE(tokenizer)
        model.load_state_dict(torch.load(dirpath / 'mtransformer.pt'))
        model.eval()
        return model