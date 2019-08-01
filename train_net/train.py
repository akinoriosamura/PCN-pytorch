import datetime
import os
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import mtcnn.core.image_tools as image_tools
import numpy as np

from pcn.image_reader import TrainImageReader
from pcn.models import PCN1, PCN2, PCN3, LossFn
from process_data.data_loador import PCNDetectorDataset
from process_data.data_transformers import ToTensor


def compute_accuracy(prob_cls, gt_cls):

    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    #we only need the detection which >= 0
    mask = torch.ge(gt_cls,0)
    #get valid element
    valid_gt_cls = torch.masked_select(gt_cls,mask)
    valid_prob_cls = torch.masked_select(prob_cls,mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls,0.6).float()
    right_ones = torch.eq(prob_ones,valid_gt_cls).float()

    ## if size == 0 meaning that your gt_labels are all negative, landmark or part

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))  ## divided by zero meaning that your gt_labels are all negative, landmark or part


def pcn1(model_store_path, end_epoch, annotation_file,
              batch_size,frequent=10,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = PCN1(is_train=True, use_cuda=use_cuda)
    net.train()

    if use_cuda:
        net.cuda()
    # optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    optimizer = torch.optim.SGD(net.parameters(), lr=base_lr)

    # preprocess for dataset
    assert os.path.exists(annotation_file), 'Path does not exist: {}'.format(annotation_file)
    annotations = []
    with open(annotation_file, 'r') as f:
        annotations_set = f.readlines()
    annotations = [annotation.rstrip().split(" ") for annotation in annotations_set]

    # get dataset applied transform
    data_transform = transforms.Compose([
        ToTensor()
    ])
    face_dataset = PCNDetectorDataset(annotations, data_transform)
    dataloader = DataLoader(face_dataset, batch_size=2, shuffle=True)

    frequent = 10
    for cur_epoch in range(1,end_epoch+1):
        # train_data.reset() # shuffle

        for batch_idx, batches in enumerate(dataloader):
            img = batches['image']
            gt_cls = batches['gt_cls']
            bboxes = batches['bboxes']
            thetas = batches['thetas']
            im_tensor = Variable(img)
            gt_cls = Variable(gt_cls)
            gt_bbox = Variable(bboxes)
            gt_theta = Variable(thetas)

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_cls = gt_cls.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_theta = gt_theta.cuda()

            cls_pred, theta, bbox = net(im_tensor)

            cls_loss = lossfn.cls_loss(gt_cls, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_cls, gt_bbox, bbox)
            theta_loss = lossfn.theta_loss(gt_cls, gt_theta, theta)

            all_loss = cls_loss*1.0 + box_offset_loss*0.5 + theta_loss*0.5

            if batch_idx % frequent==0:
                accuracy=compute_accuracy(cls_pred, gt_cls)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                # show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show5,base_lr))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(model_store_path,"pnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"pnet_epoch_model_%d.pkl" % cur_epoch))




def pcn2(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = RNet(is_train=True, use_cuda=use_cuda)
    net.train()
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,24,batch_size,shuffle=True)


    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()

        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):

            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*1.0+box_offset_loss*0.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                # show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(), cur_epoch, batch_idx, show1, show2, show3, show5, base_lr))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(model_store_path,"rnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"rnet_epoch_model_%d.pkl" % cur_epoch))


def pcn3(model_store_path, end_epoch,imdb,
              batch_size,frequent=50,base_lr=0.01,use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = ONet(is_train=True)
    net.train()
    print(use_cuda)
    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data=TrainImageReader(imdb,48,batch_size,shuffle=True)


    for cur_epoch in range(1,end_epoch+1):

        train_data.reset()

        for batch_idx,(image,(gt_label,gt_bbox,gt_landmark))in enumerate(train_data):
            # print("batch id {0}".format(batch_idx))
            im_tensor = [ image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred, landmark_offset_pred = net(im_tensor)

            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss*0.8+box_offset_loss*0.6+landmark_loss*1.5

            if batch_idx%frequent==0:
                accuracy=compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                show4 = landmark_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, landmark loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show4,show5,base_lr))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        torch.save(net.state_dict(), os.path.join(model_store_path,"onet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"onet_epoch_model_%d.pkl" % cur_epoch))