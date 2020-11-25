from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import transforms as T
from engine import train_one_epoch
import utils
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import h5py
import json

def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])

def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr[()][i].item()][()][0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr[()][0][0]]
        attrs[key] = values
    return attrs

class SVHN_train_dataset(Dataset):
    def __init__(self, root, transform):

        self.root = root
        self.transform = transform
        #load mat file(hdf5)
        self.hdf5_data = h5py.File(root + 'digitStruct.mat', 'r')

    def __len__(self):
        return self.hdf5_data['/digitStruct/name'].shape[0]

    def __getitem__(self, idx):
        #load labels
        bbox_dict = get_bbox(idx, self.hdf5_data)
        num_bbox = len(bbox_dict['label']) #length of label = numbers of bbox in image

        #build box
        img_bbox = []
        for i in range (num_bbox):
            bbox_left = bbox_dict['left'][i] #x0
            bbox_top = bbox_dict['top'][i] #y0
            bbox_right = bbox_left + bbox_dict['width'][i] #x1 = x0 + width
            bbpx_bottom = bbox_top + bbox_dict['height'][i] #y1 = y0 + height
            img_bbox.append([bbox_left, bbox_top, bbox_right, bbpx_bottom]) #format is [x0,y0,x1,y1]
        img_bbox = torch.as_tensor(img_bbox, dtype = torch.float32) #convert everything into a torch.Tensor

        #build labels
        img_label = torch.zeros(num_bbox, dtype = torch.int64) #creat int64tensor with shape(num_bbox, 1)
        for i in range(num_bbox):
            img_label[i] = int(bbox_dict['label'][i]) #fill in with int label

        #set image_id
        img_id = torch.tensor([idx])

        #calculate area
        bbox_area = (img_bbox[:, 3] - img_bbox[:, 1]) * (img_bbox[:, 2] - img_bbox[:, 0])

        #set iscrowd (suppose all instances are not crowd)
        iscrowd = torch.zeros(num_bbox, dtype = torch.int64)

        #open Image by PIL
        img_name = get_name(idx, self.hdf5_data)
        image = Image.open(self.root + img_name).convert("RGB")

        #build target dict
        target = {}
        target["boxes"] = img_bbox
        target["labels"] = img_label
        target["image_id"] = img_id
        target["area"] = bbox_area
        target["iscrowd"] = iscrowd

        #transform image and bbox
        if self.transform is not None:
                image, target = self.transform(image, target)
        
        return image, target

class SVHN_test_dataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.transform = transforms.ToTensor()
        self.test_names = [f for f in os.listdir(root) 
                                if os.path.isfile(os.path.join(root, f)) and f.endswith('.png')] 
    
    def __len__(self):
        return len(self.test_names)

    def __getitem__(self, idx):
        num = idx + 1
        file_name = str(num) + '.png'

        image = Image.open(self.root + file_name).convert("RGB")

        #transform image
        if self.transform is not None:
                image = self.transform(image)
               
        return image

def get_model_object_detection(num_classes):
    # load a object dectection model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model   

def get_transform(train):
    transform = []
    transform.append(T.ToTensor())
    if train:
        transform.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transform)

def to_answer_format(pre):
    #print(pre['labels'])
    num_label = len(pre['labels'])
    box_list = []
    score_list = []
    label_list = []
    for i in range(num_label):
        x0 = pre['boxes'][i][0].item()
        y0 = pre['boxes'][i][1].item()
        x1 = pre['boxes'][i][2].item()
        y1 = pre['boxes'][i][3].item()
        bbox = (y0, x0, y1, x1)
        
        label = pre['labels'][i].item()
        score = pre['scores'][i].item()

        box_list.append(bbox)
        label_list.append(label)
        score_list.append(score)

    return {"bbox": box_list, "label": label_list, "score": score_list}

def main(training):
    #os.environ["CUDA_VISIBLE_DEVICES"]="7"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_class = 11 #background + 10 number

    print("loading")
    train_dataset = SVHN_train_dataset('train/', get_transform(False))
    trainloader = DataLoader(train_dataset, batch_size = 2, shuffle = True, num_workers = 2, collate_fn = utils.collate_fn)

    test_dataset = SVHN_test_dataset('test/')
    #testloader = DataLoader(test_dataset, batch_size = 2, shuffle = False, num_workers = 2, collate_fn = utils.collate_fn)

    model = get_model_object_detection(num_class)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)

    num_epoch = 10
    fname_header = './SVHN_detection_'
    fname_tail = '.pth'
    if training:
        print("training")
        for epoch in range(num_epoch):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, trainloader, device, epoch, print_freq = 10)
            # update the learning rate
            lr_scheduler.step()

            #save model
            PATH = fname_header + str(epoch) + fname_tail
            torch.save(model.state_dict(), PATH)
    else:
        PATH = fname_header + str(num_epoch - 1) + fname_tail
        model.load_state_dict(torch.load(PATH))

    print("testing")
    ans = []
    with torch.no_grad():
        # For inference
        model.eval()
        for idx in range(len(test_dataset)):
            img = test_dataset[idx]
            #print(img.shape)
            img = img.to(device)

            predictions = model(img[None, ...])
            predictions = [{k: v.to("cpu") for k, v in p.items()} for p in predictions]
            #print(predictions)

            ans.append(to_answer_format(predictions[0]))
            if idx % 100 == 0:
                print(idx)

    json.dump(ans, open('answer.json', 'w'))

if __name__ == "__main__":
    main(False)