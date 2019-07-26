import cv2
import os
import numpy as np
import torch

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

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, bboxes = sample['image'], sample['bboxes']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'bboxes': torch.from_numpy(bboxes)}