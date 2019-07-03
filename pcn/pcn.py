import os
import cv2
import numpy as np
import torch

from .model import load_model

# global settings
EPS = 1e-5
minFace_ = 20 * 1.4
scale_ = 1.414
stride_ = 8
classThreshold_ = [0.37, 0.43, 0.97]
nmsThredHold_ = [0.8, 0.8, 0.3]
angleRange_ = 45
stable_ = 0

class Window2:
    def __init__(self, x, y, w, h, angle, scale, conf):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.angle = angle
        self.scale = scale
        self.conf = conf


def preprocess_img(img, dim=None):
    if dim:
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_NEAREST)
    # TODO:謎の引き算
    return img - np.array([104, 117, 123])

def resize_img(img, scale:float):
    h, w = img.shape[:2]
    h_, w_ = int(h / scale), int(w / scale)
    img = img.astype(np.float32)
    ret = cv2.resize(img, (w_, h_), interpolation=cv2.INTER_NEAREST)
    return ret

def pad_img(img:np.array):
    # height
    row = min(int(img.shape[0] * 0.2), 100)
    # width
    col = min(int(img.shape[1] * 0.2), 100)
    # add pad in aroud img: (src, top, bottom, left, right)
    ret = cv2.copyMakeBorder(img, row, row, col, col, cv2.BORDER_CONSTANT)
    return ret

def legal(x, y, img):
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        return True
    else:
        return False

def inside(x, y, rect:Window2):
    if rect.x <= x < (rect.x + rect.w) and rect.y <= y < (rect.y + rect.h):
        return True
    else:
        return False

def smooth_angle(a, b):
    if a > b:
        a, b = b, a
    diff = (b - a) % 360
    if diff < 180:
        return a + diff // 2
    else:
        return b + (360 - diff) // 2

# use global variable `prelist` to mimic static variable in C++
prelist = []
def smooth_window(winlist):
    global prelist

def IoU(w1:Window2, w2:Window2) -> float:
    


def NMS(winlist, local:bool, threshold:float):
    length = len(winlist)
    if length == 0:
        return winlist
    winlist.sort(key=lambda x: x.conf, reverse=True)
    flag = [0] * length
    for i in range(lenght):
        if flag[i]:
            continue
        for j in range(i+1, length):
            # 
            if local and abs(winlist[i].scale - winlist[j].scale) > EPS:
                continue
            if IoU(winlist[i], winlist[j] > threshold):
                flag[j] = 1
            

def set_input(img):
    if type(img) == list:
        # change from list to numpy
        # 0次元に新次元追加のつもり？？なってないけど
        # TODO: listだった場合のimg size確認
        img = np.stack(img, axis=0)
    else:
        # 0次元に新次元追加
        img = img[np.newaxis, :, :, :]
    # [b, h, w, c] -> [b, c, h, w]
    img = img.transpose((0, 3, 1, 2))
    return torch.FloatTensor(img)

def stage1(img, imgPad, net, thres):
    # 切り捨て除算によりpad分を計算
    row = (imgPad.shape[0] - img.shape[0]) // 2
    col = (imgPad.shape[1] - img.shape[1]) // 2
    winlist = []
    # TODO: 何これ？
    netSize = 24
    # net size > minFace_ほどresize縮小度小さく＝拡大
    # netsize = minFace_なら等倍
    curScale = minFace_ / netSize
    img_resized = resize_img(img, curScale)
    # netsizeよりも画像の縦か横が小さくなるまで
    while min(img_resized.shape[:2]) >= netSize:
        img_resized = preprocess_img(img_resized)
        # net forward
        net_input = set_input(img_resized)
        while torch.no_grad():
            net.eval()
            cls_prob, rotate, bbox = net(net_input)

        # w = minFace_
        w = netSize* curScale
        # TODO: これ、bboxの間違いじゃね？
        # TODO: ここら辺もっかい見る
        for i in range(cls_prob[2]): # cls_prob[2]->height
            for j in range(cls_prob.shape[3]): # cls_prob.shape[3]->width
                if cls_prob[0, 1, i, j].item() > thres:
                    sn = bbox[0, 0, i, j].item()
                    xn = bbox[0, 1, i, j].item()
                    yn = bbox[0, 2, i, j].item()
                    rx = int(j * curScale * stride_ - 0.5 * sn * w + sn * xn * w + 0.5 * w) + col
                    ry = int(i * curScale * stride_ - 0.5 * sn * w + sn * yn * w + 0.5 * w) + row
                    rw = int(w * sn)
                    if legal(rx, ry, imgPad) and legal(rx + rw - 1, ry + rw - 1, imgPad):
                        if rotate[0, 1, i, j].item() > 0.5:
                            winlist.append(Window2(rx, ry, rw, rw, 0, curScale, cls_prob[0, 1, i, j].item()))
                        else:
                            winlist.append(Window2(rx, ry, rw, rw, 180, curScale, cls_prob[0, 1, i, j].item()))
        img_resized = resize_img(img_resized, scale_)
        curScale = img.shape[0] / img_resized/shape[0]
    return winlist

def stage2(img, img180, net, thres, dim, winlist)



def detect(img, imgPad, nets):
    img180 = cv2.flip(imgPad, 0)
    img90 = cv2.transpose(imgPad, 0)
    imgNeg90 = cv2.flip(img90, 0)
    winlist = stage1(img, imgPad, nets[0], classThreshold_[0])
    winlist = NMS(winlist, True, nmsThredHold_[0])


def pcn_detect(img, nets):
    imgPad = pad_img(img)
    winlist = detect(img, imgPad, nets)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pcn.py path/to/img")
        sys.exit()
    else:
        imgpath = sys.argv[1]
    
    # network detection
    nets = load_model()
    img = cv2.imread(imgpath)
    faces = pcn_detect(img, nets)
