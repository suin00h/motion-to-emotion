import torch
import ast
from torch.nn.utils.rnn import pad_sequence as _pad

__all__ = [
    'collate_PE',
    'collate_EP',
    'to_tensor',
    'norm_keys',
]

# custom collate function for variable-size input training PE model
def collate_PE(batch):
    return {
        'image': [item['image'] for item in batch],                         # list of tensors [B]: (tensor)[3, H, W]
        'seq': torch.tensor([item['seq'] for item in batch]),               # [B]
        'frame_idx': torch.tensor([item['frame_idx'] for item in batch]),   # [B]
        'action': torch.tensor([item['action'] for item in batch]),         # [B]
        'emotion': torch.tensor([item['emotion'] for item in batch]),       # [B]
        'bbox': torch.stack([item['bbox'] for item in batch]),              # [B, 4]
        'keypoints': torch.stack([item['keypoints'] for item in batch]),    # [B, 15, 3]
    }

# custom collate function for variable-size input training EP model
def collate_EP(batch):
    keys = _pad(
        sequences=[item[0] for item in batch],
        batch_first=True,
        padding_value=-1)
    action = torch.stack([item[1] for item in batch]).squeeze()
    emotion = torch.stack([item[2] for item in batch]).squeeze()
    
    return {
        'keys': keys,
        'action': action,
        'emotion': emotion}

def to_tensor(series):
    return torch.tensor(series.apply(ast.literal_eval).values.tolist(), dtype=float)

def norm_keys(bbox, keys):
    for i in range(bbox.shape[0]):
        w = bbox[i][2] - bbox[i][0]
        h = bbox[i][3] - bbox[i][1]

        keys[i, :, 0] = (keys[i, :, 0] - bbox[i][0]) / w
        keys[i, :, 1] = (keys[i, :, 1] - bbox[i][1]) / h

        for j in range(15):
            if keys[i, j, -1] == 0:
                keys[i, j] = torch.tensor([0, 0, 0], dtype=float)
    return keys