import cv2
import numpy as np
from torch.utils.data import Dataset

class FaceDetectorDataset(Dataset):
    """
    Face detector dataset processor
    params:
        annotations: [ファイルパス, bbs]の行列
            - [
                ['file path', 2, 24, 55, 466, ...], 
                [], 
                ...
                ]
            - bbs: [[x1, y1, w, h], ...]
        transform: transform instance
            - batch を取るために、transformでrescaleは必要
    """
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
        
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_name = annotation[0]
        image = cv2.imread(img_name)
        bbs = np.array([annotation[i:i+4] for i in range(1, len(annotation), 4)])
        bbs = bbs.astype(np.float)
        sample = {'image': image, 'bbs': bbs}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


