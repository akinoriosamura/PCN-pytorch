import numpy as np
"""
def IoU(box, boxes):
    # box = (x1, y1, x2, y2)
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    
    # abtain the offset of the interception of union between crop_box and gt_box
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr"""


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (4, ): x1, y1, w, h
        input box
    boxes: numpy array, shape (n, 4): x1, y1, w, h
        input ground truth boxes
    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    # box = (x1, y1, x2, y2)
    box_area = (box[2] + 1) * (box[3] + 1)
    area = (boxes[:, 2] + 1) * (boxes[:, 3] + 1)
    
    # abtain the offset of the interception of union between crop_box and gt_box
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum((box[0] + box[2] - 1), (boxes[:, 0] + boxes[:, 2] - 1))
    yy2 = np.minimum((box[1] + box[3] - 1), (boxes[:, 1] + boxes[:, 3] - 1))

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square
    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox
    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox


def get_thetas(theta):
    """
    get theta label
    stage1: 
        0: 0° (-90 <= theta <= 90)
        1: 180° (-180 <= theta < -90, 90 < theta <= 180)
    stage2:
        0: 90° (-90 <= theta < -45)
        1: 0° (-45 <= theta <= 45)
        2: -90° (45 < theta <= 90)
    stage3:
        θ = theta
    """
    if -90 <= theta <= 90:
        th_1 = 0
        rotated_theta_1 = theta
        if rotated_theta_1 < -45:
            th_2 = 0
            rotated_theta_2 = rotated_theta_1 + 90
            th_3 = rotated_theta_2
        elif -45 <= rotated_theta_1 <= 45:
            th_2 = 1
            rotated_theta_2 = rotated_theta_1
            th_3 = rotated_theta_2
        else:
            th_2 = 2
            rotated_theta_2 = rotated_theta_1 - 90
            th_3 = rotated_theta_2
    else:
        th_1 = 1
        rotated_theta_1 = theta + 180 if theta < -90 else theta - 180
        if rotated_theta_1 < -45:
            th_2 = 0
            rotated_theta_2 = rotated_theta_1 + 90
            th_3 = rotated_theta_2
        elif -45 <= rotated_theta_1 <= 45:
            th_2 = 1
            rotated_theta_2 = rotated_theta_1
            th_3 = rotated_theta_2
        else:
            th_2 = 2
            rotated_theta_2 = rotated_theta_1 - 90
            th_3 = rotated_theta_2

    thetas = [th_1, th_2, th_3]

    return thetas