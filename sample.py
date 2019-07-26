from urllib.request import urlopen
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Resize,
    CenterCrop,
    RandomCrop,
    Crop,
    Rotate,
    Compose
)

# Functions to visualize bounding boxes and class labels on an image.
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name, Flag):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])),(200,0,0),2)
        # cv2.imshow('face detector', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        cv2.imwrite("./sample/" + Flag + str(idx) + ".jpg", img)

def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params={'format': 'coco', 'min_area': min_area, 'min_visibility': min_visibility, 'label_fields': ['category_id']})

image = cv2.imread("dataset/face_detection/WIDERFACE/WIDER_train/images/0--Parade/0_Parade_Parade_0_110.jpg")

# Annotations for image 386298 from COCO http://cocodataset.org/#explore?id=386298
bboxes = [[0, 486, 24, 32], [461, 493, 40, 63]]
faces = np.array(range(len(bboxes)))
annotations = {'image': image, 'bboxes': [[0, 486, 24, 32], [461, 493, 40, 63]], 'category_id': faces}
category_id_to_name = {}
for i in range(len(bboxes)):
    category_id_to_name[i] = 'face'

visualize(annotations, category_id_to_name, "default")

# aug = get_aug([VerticalFlip(p=1)])
import pdb; pdb.set_trace()
aug = get_aug([Rotate(limit=(-90,-90), p=1)])
augmented = aug(**annotations)
visualize(augmented, category_id_to_name, "rotate")
