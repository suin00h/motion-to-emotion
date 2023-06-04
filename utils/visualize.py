import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from typing import List, Tuple
from torch import Tensor

__all__ = [
    'show_batch',
    'show_loss',
]

def show_batch(
    images: List[Tensor],
    bbox: List[Tensor],
    keypoints: List[Tensor],
    num_image: int = 1,
    fig_size: int = 9,
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
            color='violet',
            linewidth=2,
            fill=False,))
        for k in key:
            if not k[-1]:
                continue
            ax.plot(k[0], k[1], 'ro', ms=3)
        ax.axis('off')
        
    plt.show()
        
def show_loss(
    *loss: List[List],
    fig_size: Tuple[int, int] = (6, 4),    
) -> None:
    plt.figure(figsize=fig_size)
    
    for i, x in enumerate(['train', 'validation']):
        plt.plot(loss[i], label=x)
    
    plt.title('Loss Graph')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    
    plt.show()