import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data_loador import FaceDetectorDataset

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, bbs = sample['image'], sample['bbs']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))

        # h and w are swapped for bbs because for images,
        # x and y axes are axis 1 and 0 respectively
        # bbs = bbs * [new_w / w, new_h / h]
        x_scale = new_w / w
        y_scale = new_h / h

        newbb = []
        for bb in bbs:
            # original frame as named values
            (origLeft, origTop, origWidth, origHeight) = bb

            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            w = int(np.round(origWidth * x_scale))
            h = int(np.round(origHeight * y_scale))
            newbb.append([x, y, w, h])
        newbb = np.array(newbb)

        return {'image': img, 'bbs': newbb}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bbs = sample['image'], sample['bbs']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bbs': torch.from_numpy(bbs)}

# preprocess for dataset
annotation_file = './dataset/face_detection/WIDERFACE/anno_train.txt'
assert os.path.exists(annotation_file), 'Path does not exist: {}'.format(self.annotation_file)
annotations = []
with open(annotation_file, 'r') as f:
    annotations_set = f.readlines()
annotations = [annotation.rstrip().split(" ") for annotation in annotations_set]


# get dataset applyin transform
data_transform = transforms.Compose([
    Rescale((224, 224)),
    ToTensor()
])
face_dataset = FaceDetectorDataset(annotations, data_transform)

"""
# tensor cannot has key
for i in range(len(face_dataset)):
    img = face_dataset[i]["image"]
    bbs = face_dataset[i]["bbs"]
    for bb in bbs:
        cv2.rectangle(img, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (255, 0, 0))
    cv2.imwrite(os.path.join("sample_processed", str(i) + ".jpg"), img)
    if i == 10:
        break
"""


dataloader = DataLoader(face_dataset, batch_size=2, shuffle=True)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['bbs'].size())

    if i_batch == 3:
        break