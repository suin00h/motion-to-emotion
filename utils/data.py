import torch
from torch.nn.utils.rnn import pad_sequence as _pad

__all__ = [
    'collate_PE',
    'collate_EP',
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