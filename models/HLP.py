import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
)

class HLP_net(nn.Module):
    def __init__(self, hlp_model_name_or_path):
        super().__init__()
        base_model = AutoModel.from_pretrained(hlp_model_name_or_path)

    

