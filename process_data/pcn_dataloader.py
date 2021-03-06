import cv2
import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as al

from data_loador import PCNDetectorDataset

"""
class Rescale(object):
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))

        # h and w are swapped for bboxes because for images,
        # x and y axes are axis 1 and 0 respectively
        # bboxes = bboxes * [new_w / w, new_h / h]
        x_scale = new_w / w
        y_scale = new_h / h

        newbb = []
        for bb in bboxes:
            # original frame as named values
            (origLeft, origTop, origWidth, origHeight) = bb

            x = int(np.round(origLeft * x_scale))
            y = int(np.round(origTop * y_scale))
            w = int(np.round(origWidth * x_scale))
            h = int(np.round(origHeight * y_scale))
            newbb.append([x, y, w, h])
        newbb = np.array(newbb)

        return {'image': img, 'bboxes': newbb}
"""

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt_cls, bboxes, thetas = sample['image'], sample['gt_cls'], sample['bboxes'], sample['thetas']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'gt_cls': torch.from_numpy(gt_cls),
                'bboxes': torch.from_numpy(bboxes),
                'thetas': torch.from_numpy(thetas)}

# preprocess for dataset
annotation_file = './dataset/anno_store/imglist_anno_24.txt'
assert os.path.exists(annotation_file), 'Path does not exist: {}'.format(annotation_file)
annotations = []
with open(annotation_file, 'r') as f:
    annotations_set = f.readlines()
annotations = [annotation.rstrip().split(" ") for annotation in annotations_set]

# get dataset applyin transform
data_transform = transforms.Compose([
    ToTensor()
])
face_dataset = PCNDetectorDataset(annotations, data_transform)

dataloader = DataLoader(face_dataset, batch_size=1, shuffle=True)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['gt_cls'], sample_batched['bboxes'], sample_batched['thetas'])
    # print(i_batch, sample_batched['image'].size(), sample_batched['gt_cls'].size(), sample_batched['bboxes'].size(), sample_batched['thetas'].size())
    if i_batch == 5:
        break

# tensor cannot has key
for i in range(len(face_dataset)):
    # import pdb; pdb.set_trace()
    img = face_dataset[i]["image"].numpy()
    img = img.transpose((1, 2, 0))
    print(img.shape)
    bboxes = face_dataset[i]["bboxes"].numpy()
    print("cls: ", face_dataset[i]["gt_cls"].numpy())
    print("thetas: ", face_dataset[i]["thetas"].numpy())
    print(bboxes)
    if not np.isnan(bboxes[0][0]):
        for bb in bboxes:
            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), (200, 0, 0), 1)
        cv2.imwrite(os.path.join("sample_processed", str(i) + ".jpg"), img)
        if i == 5:
            break
