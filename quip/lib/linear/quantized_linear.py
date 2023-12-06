import torch
import torch.nn as nn
from lib.utils import get_hadK
from lib import codebook

def dtype_from_str(str):
    dtype_map = {
        'torch.int32': torch.int32,
        'torch.int16': torch.int16,
        'torch.uint8': torch.uint8,
    }
    return dtype_map[str]

class QuiPLinear(nn.Module):

    def __init__(self, model, key, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.outlier_channel_split = model.config.quip_params['outlier_channel_split']
        self.rescale_WH = model.config.quip_params['rescale_WH']
        self.idx_dtype = {
            'torch.int32': torch.int32,
            'torch.int16': torch.int16,
            'torch.uint8': torch.uint8,
            }[model.config.quip_params['idx_dtype']]
        self.codesz = model.config.quip_params['codesz']

        if self.outlier_channel_split: self.ocs_dupe_inds = torch.arange(in_features)

        self.scaleWH = torch.ones(in_features) if self.rescale_WH else None
        
        self.Qidxs = torch.zeros(
            out_features, in_features // self.codesz, dtype=self.idx_dtype)
        self.codebook_id = torch.tensor(0)
        self.SU = torch.ones(in_features)
        self.SV = torch.ones(out_features)
        self.Wscale = torch.ones(())

        self.built_codebook_class = False
        self.built_graph = False

        self.had_left, self.K_left = get_hadK(in_features)
        self.had_right, self.K_right = get_hadK(out_features)
        
    def forward(self, input):
        if not self.built_codebook_class:
            self.codebook_class = codebook.get_quantized_class(
                self.codebook_id.item())(self.Qidxs.device)
            self.built_codebook_class = True

        if self.outlier_channel_split:
            input = input[..., self.ocs_dupe_inds]

        return self.codebook_class(
            input,
            self.Qidxs, self.SU, self.SV, self.Wscale,
            self.had_left, self.had_right, self.K_left, self.K_right,
            rank=self.rank, A=self.A, B=self.B,
            rescale_WH=self.rescale_WH, scaleWH=self.scaleWH)
