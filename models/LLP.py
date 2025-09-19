import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
)


class LLP_net(nn.Module):
    def __init__ (self, llp_model_name_or_path):
        super().__init__()
        base_model = AutoModel.from_pretrained(llp_model_name_or_path)

    def take_action():
        pass

    def take_action_eval():
        pass
