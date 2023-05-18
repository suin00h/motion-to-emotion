import torch

__all__ = [
    '_collate',
]

# custom collate function for variable-size input
def _collate(batch):
    return {
        'image': [item['image'] for item in batch],                         # list of tensors [B]: (tensor)[3, H, W]
        'seq': torch.tensor([item['seq'] for item in batch]),               # [B]
        'frame_idx': torch.tensor([item['frame_idx'] for item in batch]),   # [B]
        'action': torch.tensor([item['action'] for item in batch]),         # [B]
        'emotion': torch.tensor([item['emotion'] for item in batch]),       # [B]
        'bbox': torch.stack([item['bbox'] for item in batch]),              # [B, 4]
        'keypoints': torch.stack([item['keypoints'] for item in batch]),    # [B, 15, 3]
    }