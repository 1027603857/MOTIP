# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MOTIP(nn.Module):
    def __init__(
            self,
            only_detr: bool,
            feature_extractor: nn.Module,
            trajectory_modeling: nn.Module,
            id_decoder: nn.Module,
    ):
        super().__init__()
        self.only_detr = only_detr
        self.feature_extractor = feature_extractor
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder

        if self.id_decoder is not None:
            self.num_id_vocabulary = self.id_decoder.num_id_vocabulary
        else:
            self.num_id_vocabulary = 1000           # hack implementation

        return

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for MOTIP forward."
        match kwargs["part"]:
            case "feature_extractor":
                annotations = kwargs["annotations"]
                return self.feature_extractor(annotations)
            case "trajectory_modeling":
                seq_info = kwargs["seq_info"]
                return self.trajectory_modeling(seq_info)
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                use_decoder_checkpoint = kwargs["use_decoder_checkpoint"] if "use_decoder_checkpoint" in kwargs else False
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            case _:
                raise NotImplementedError(f"MOTIP forwarding doesn't support part={kwargs['part']}.")
