import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

__all__ = [
    'my_collate',
    'visualize',
]

# custom collate function for variable-size input
def my_collate(batch):
    return {
        'image': [item['image'] for item in batch],                         # list of tensors [B]: (tensor) [3, H, W]
        'seq': torch.tensor([item['seq'] for item in batch]),               # [B]
        'frame_idx': torch.tensor([item['frame_idx'] for item in batch]),   # [B]
        'action': torch.tensor([item['action'] for item in batch]),         # [B]
        'emotion': torch.tensor([item['emotion'] for item in batch]),       # [B]
        'bbox': torch.stack([item['bbox'] for item in batch]),              # [B, 4]
        'keypoints': torch.stack([item['keypoints'] for item in batch]),    # [B, 15, 3]
    }
    
def visualize(
    num_image: int = 1,
    fig_size: int = 9,
    images: list = None,
    bbox: list = None,
    keypoints: list = None,
) -> None:
    fig = plt.figure(figsize=(fig_size, fig_size))
    cols = math.ceil(math.sqrt(num_image))
    rows = num_image // cols + 1

    for i in range(num_image):
        x1, y1, x2, y2 = bbox[i]
        key = keypoints[i]

        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(images[i].permute(1, 2, 0))
        ax.add_patch(patches.Rectangle(
            (x1, y1),
            x2-x1,
            y2-y1,
            color='w',
            alpha=0.2,
            fill=True,))
        ax.add_patch(patches.Rectangle(
            (x1, y1),
            x2-x1,
            y2-y1,
            color='c',
            linewidth=2,
            fill=False,))
        for k in key:
            if not k[-1]:
                continue
            ax.plot(k[0], k[1], 'ro', ms=3)
        ax.axis('off')