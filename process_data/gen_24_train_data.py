"""
    2018-10-20 15:50:20
    generate positive, negative, positive images whose size are 24*24 and feed into PNet
    get each annotation text files
    the each file line is
    [imgPath cls_label(pos:1, part:-1, neg:0) bboxes(n, x, y, w, h) thetas(180orNot, -90or0r90, theta)]
"""
import sys
import numpy as np
import cv2
import os
import random
sys.path.append(os.getcwd())
import albumentations as al

from process_data.utils import IoU, get_thetas


prefix = ''
anno_file = "./dataset/anno_store/anno_train.txt"
im_dir = "./dataset/face_detection/WIDERFACE/WIDER_train/images"
pos_save_dir = "./dataset/train/24/positive"
part_save_dir = "./dataset/train/24/part"
neg_save_dir = './dataset/train/24/negative'

if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

# store labels of positive, negative, part images
f1 = open(os.path.join('./dataset/anno_store', 'pos_24.txt'), 'w')
f2 = open(os.path.join('./dataset/anno_store', 'neg_24.txt'), 'w')
f3 = open(os.path.join('./dataset/anno_store', 'part_24.txt'), 'w')

# anno_file: store labels of the wider face training data
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)

p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0

img_size = 24

for annotation in annotations[:10]:
    annotation = annotation.strip().split(' ')
    # import pdb; pdb.set_trace()
    # im_path = os.path.join(prefix, annotation[0])
    im_path = annotation[0]
    print(im_path)
    bbox = list(map(float, annotation[1:]))
    # 画像に直立顔ラベルがない場合スキップ
    if bbox == []:
        continue
    # 1画像あたり10setのrotateデータ作成
    for i in range(10):
        theta = random.randint(-180, 180)
        thetas = get_thetas(theta)
        print("theta: ", theta)
        print("thetas: ", thetas)
        bboxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        img = cv2.imread(im_path)
        idx += 1
        if idx % 100 == 0:
            print(idx, "images done")

        # rotate image
        faces = np.array(range(len(bboxes)))
        annotation_dict = {'image': img, 'bboxes': bboxes, 'category_id': faces}
        category_id_to_name = {}
        for i in range(len(bboxes)):
            category_id_to_name[i] = 'face'    
        aug = al.Compose([al.Rotate(limit=(theta,theta), p=0.8)], bbox_params={'format': 'coco', 'min_area': 0., 'min_visibility': 0., 'label_fields': ['category_id']})
        augmented = aug(**annotation_dict)
        if augmented["bboxes"] == []:
            continue

        aug_img = augmented["image"]
        aug_bboxes = np.array(augmented["bboxes"]).astype(int)

        height, width, channel = aug_img.shape

        neg_num = 0
        # 一枚の画像につき、50setのnegative ラベルを作成。IoU0.3以下ならnegativeデータとして保存。
        while neg_num < 10:
            size = np.random.randint(img_size, min(width, height) / 2)
            nx = np.random.randint(0, width - size)
            ny = np.random.randint(0, height - size)
            crop_box = np.array([nx, ny, size, size])

            Iou = IoU(crop_box, aug_bboxes)

            cropped_im = aug_img[ny: ny + size, nx: nx + size, :]
            resized_im = cv2.resize(cropped_im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            np.max(Iou)
            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                # f2.write(save_file + ' 0\n')
                f2.write(save_file + ' 0 %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))

                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1

        for box in aug_bboxes:
            # box (x_left, y_top, w, h)
            x1, y1, w, h = box
            x2 = x1 + w
            y2 = y1 + h
            # x1, y1, x2, y2 = box
            # w = x2 - x1 + 1
            # h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth bboxes of small faces are not accurate
            if max(w, h) < 40 or min(w, h) < 5 or x1 < 0 or y1 < 0:
                continue

            # generate negative examples that have overlap with gt
            for i in range(10):
                size = np.random.randint(img_size, min(width, height) / 2)
                # delta_x and delta_y are offsets of (x1, y1)

                delta_x = np.random.randint(max(-size, -x1), w)
                delta_y = np.random.randint(max(-size, -y1), h)
                nx1 = max(0, x1 + delta_x)
                ny1 = max(0, y1 + delta_y)

                if nx1 + size > width or ny1 + size > height:
                    continue
                crop_box = np.array([nx1, ny1, size, size])
                Iou = IoU(crop_box, aug_bboxes)

                cropped_im = aug_img[ny1: ny1 + size, nx1: nx1 + size, :]
                resized_im = cv2.resize(cropped_im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

                if np.max(Iou) < 0.3:
                    # Iou with all gts must below 0.3
                    save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                    # f2.write(save_file + ' 0\n')
                    f2.write(save_file + ' 0 %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n' % (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                    cv2.imwrite(save_file, resized_im)
                    n_idx += 1

            # generate positive examples and part faces
            for i in range(5):
                size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

                # delta here is the offset of box center
                try:
                    delta_x = np.random.randint(-w * 0.2, w * 0.2)
                    delta_y = np.random.randint(-h * 0.2, h * 0.2)
                except:
                    import pdb; pdb.set_trace()

                nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
                ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                crop_box = np.array([nx1, ny1, size, size])

                # offset_x1 = (x1 - nx1) / float(size)
                offset_x1 = 0 if x1<nx1 else x1 - nx1
                offset_x1 *= img_size / float(size)
                # offset_y1 = (y1 - ny1) / float(size)
                offset_y1 = 0 if y1<ny1 else y1 - ny1
                offset_y1 *= img_size / float(size)
                # offset_x2 = (x2 - nx2) / float(size)
                offset_x2 = x2 - nx1 if x2<nx2 else size
                offset_x2 *= img_size / float(size)
                # offset_y2 = (y2 - ny2) / float(size)
                offset_y2 = y2 - ny1 if y2<ny2 else size
                offset_y2 *= img_size / float(size)
                offset_w = offset_x2 - offset_x1
                offset_h = offset_y2 - offset_y1

                cropped_im = aug_img[int(ny1): int(ny2), int(nx1): int(nx2), :]
                resized_im = cv2.resize(cropped_im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

                box_ = box.reshape(1, -1)
                # cv2.rectangle(resized_im,(int(offset_x1),int(offset_y1)),(int(offset_x2),int(offset_y2)),(200,0,0),2) 
                if IoU(crop_box, box_) >= 0.65:
                    # cv2.rectangle(aug_img,(int(x1),int(y1)),(int(x2),int(y2)),(200,0,0),2) 
                    # cv2.imshow('face detector', aug_img)                
                    # cv2.rectangle(resized_im,(int(offset_x1),int(offset_y1)),(int(offset_x2),int(offset_y2)),(200,0,0),1) 
                    # cv2.imshow('face detector', resized_im)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # import pdb; pdb.set_trace()

                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f %.1f %.1f %.1f\n' % (offset_x1, offset_y1, offset_w, offset_h, thetas[0], thetas[1], thetas[2]))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif IoU(crop_box, box_) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f %.1f %.1f %.1f\n' % (offset_x1, offset_y1, offset_w, offset_h, thetas[0], thetas[1], thetas[2]))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
            box_idx += 1
            print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

f1.close()
f2.close()
f3.close()
