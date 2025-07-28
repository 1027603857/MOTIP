# Copyright (c) Ruopeng Gao. All Rights Reserved.
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

from torch.utils.data import Dataset
from utils.nested_tensor import nested_tensor_from_tensor_list

from utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh


class SeqDataset(Dataset):
    def __init__(
            self,
            seq_info,
            image_paths,
            max_shorter: int = 800,
            max_longer: int = 1536,
            size_divisibility: int = 0,
            dtype=torch.float32,
    ):
        self.seq_info = seq_info
        self.image_paths = image_paths
        self.max_shorter = max_shorter
        self.max_longer = max_longer
        self.size_divisibility = size_divisibility
        self.dtype = dtype

        self.transform = v2.Compose([
            v2.Resize(size=self.max_shorter, max_size=self.max_longer),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = self._load_image(self.image_paths[item])
        bbox = self._load_bbox(self.image_paths[item].replace('jpg', 'txt'))
        img_w, img_h  = image.size
        bbox[0][:, ::2] /= img_w + 1e-6
        bbox[0][:, 1::2] /= img_h + 1e-6
        transformed_image = self.transform(image)
        if self.dtype != torch.float32:
            transformed_image = transformed_image.to(self.dtype)
        transformed_image = nested_tensor_from_tensor_list([transformed_image], self.size_divisibility)
        return transformed_image, self.image_paths[item], bbox

    def seq_hw(self):
        return self.seq_info["height"], self.seq_info["width"]

    @staticmethod
    def _load_image(path):
        image = Image.open(path)
        return image

    @staticmethod
    def _load_bbox(path):
        """
        读取 per-frame txt ，行格式示例：
        frame,id,x,y,w,h,score,cls,-1,-1
              ↑ ↑ ↑ ↑ ↑ ↑   ↑    ↑
              0 1 2 3 4 5   6    7
        返回 (boxes_cxcywh, scores, labels)
        """
        path = Path(path)
        if not path.exists():
            empty = torch.zeros((0,), dtype=torch.float32)
            return empty.view(0, 4), empty, empty.to(torch.int64)

        boxes, scores, labels = [], [], []
        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                if len(parts) < 7:  # 行不完整直接跳过
                    continue
                # 解析字段
                x, y, w, h = map(float, parts[2:6])
                score = float(parts[6])
                cls = int(parts[7]) if len(parts) > 7 else -1

                boxes.append([x, y, w, h])
                scores.append(score)
                labels.append(0)

        if not boxes:  # 整帧无目标
            empty = torch.zeros((0,), dtype=torch.float32)
            return empty.view(0, 4), empty, empty.to(torch.int64)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # (x y w h) → (cx cy w h)
        boxes = box_xywh_to_xyxy(boxes)
        boxes = box_xyxy_to_cxcywh(boxes)

        return [boxes, scores, labels]
