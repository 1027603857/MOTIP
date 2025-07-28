from typing import List
import torch
import torch.nn as nn

from ..fast_reid.fastreid.config import get_cfg
from ..fast_reid.fastreid.modeling.meta_arch import build_model
from ..fast_reid.fastreid.utils.checkpoint import Checkpointer

def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.freeze()
    return cfg


class FastReID():
    def __init__(self, pre_reid, weights_path="weights/dance_sbs_S50.pth"):
        config_file = "models/fast_reid/configs/dancetrack/sbs_S50.yml"
        self.cfg = setup_cfg(config_file, ['MODEL.WEIGHTS', weights_path])
        self.model = build_model(self.cfg)
        self.model.eval()
        self.model.cuda()

        Checkpointer(self.model).load(weights_path)
        self.pH, self.pW = self.cfg.INPUT.SIZE_TEST

    def __call__(self, batch):
        # Uses half during training
        with torch.no_grad():
            return self.model(batch)

class nmODE(nn.Module):
    def __init__(self, in_dim, out_dim, delta=0.2, step=5):
        super().__init__()
        self.delta = delta
        self.step = step
        self.in_ln = nn.Sequential(
            nn.Linear(2048, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Linear(256, 256, bias=True),
        )
        self.sigma = nn.Sequential(
            nn.ReLU(inplace=True),
        )
        self.out_ln = nn.Sequential(
            nn.Linear(256, 256, bias=True),
        )

    def forward(self, x):
        x = self.in_ln(x)
        y = torch.zeros_like(x)
        for step in range(self.step):
            y = (1 - self.delta) * y + self.delta * self.sigma(y + x)
        return self.out_ln(y)

class FeatureExtractor(nn.Module):
    def __init__(
            self,
            pre_reid,
    ):
        super().__init__()
        self.ReIDNet = FastReID(pre_reid)
        self.nmODE = nmODE(2048, 256)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        pass

    def forward(self, annotations):
        if self.training:
            _B, _T = len(annotations), len(annotations[0])
            _N = annotations[0][0]["trajectory_id_labels"].shape[-1]
            _, _C, _H, _W = annotations[0][0]["feature"].shape
            _device = annotations[0][0]["feature"].device
            objects = torch.zeros((_B, _T, _N, _C, _H, _W), dtype=torch.float32, device=_device)
            for b in range(_B):
                for t in range(_T):
                    feat = annotations[b][t]["feature"]
                    n_t = feat.size(0)
                    objects[b, t, :n_t] = feat
            feature_embed = objects.flatten(0, -4)
            feature_embed = self.ReIDNet(feature_embed)
            feature_embed = self.nmODE(feature_embed)
            feature_embed = feature_embed.reshape(_B, _T, _N, -1)
        else:
            origin_shape = annotations.shape[:-3]
            feature_embed = annotations.flatten(0, -4)
            feature_embed = self.ReIDNet(feature_embed)
            feature_embed = self.nmODE(feature_embed)
            feature_embed = feature_embed.reshape(*origin_shape, -1)

        return feature_embed