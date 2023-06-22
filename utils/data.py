import torch
import ast
from torch.nn.utils.rnn import pad_sequence as _pad

__all__ = [
    'collate_PE',
    'collate_EP',
    'to_tensor',
    'norm_keys',
    'get_index',
]

# custom collate function for variable-size input training PE model
def collate_PE(batch):
    # WARNING! Not working on this version.
    image = [item[0] for item in batch]
    bbox = torch.stack([item[1] for item in batch]).squeeze()
    keys = torch.stack([item[2] for item in batch]).squeeze()
    
    return {
        'image': image,
        'bbox': bbox,
        'keys': keys
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


def get_index(tensors):
    def find_largest_index(tensor):
        _, indices = torch.max(tensor, dim=0)
        return indices.item()
    
    result=[]
    for tensor in tensors:
        val=find_largest_index(tensor)
        result.append(val)
    return result