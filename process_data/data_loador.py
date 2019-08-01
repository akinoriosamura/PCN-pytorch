import cv2
import numpy as np
from torch.utils.data import Dataset

class PCNDetectorDataset(Dataset):
    """
    Face detector dataset processor
    params:
        annotations: [ファイルパス, bboxes]の行列
            - imgpath: [
                ['file path', 2, 24, 55, 466, ...], 
                [], 
                ...
                ]
            - gt_cls: (-1, 0, 1)
            - bboxes: [[x1, y1, w, h], ...]
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
        gt_cls = np.array([annotation[1]]).astype(float)
        #if gt_cls[0] != 0:
        bboxes = np.array([annotation[i:i+4] for i in range(2, len(annotation)-3, 4)])
        bboxes = bboxes.astype(np.float)
        thetas = np.array([annotation[-3:]]).astype(float)
        #else:
        #    bboxes = np.array([])
        #    thetas = np.array([])
        sample = {'image': image, 'gt_cls': gt_cls, 'bboxes': bboxes, 'thetas': thetas}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


