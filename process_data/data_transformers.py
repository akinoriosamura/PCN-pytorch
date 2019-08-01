import torch

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
