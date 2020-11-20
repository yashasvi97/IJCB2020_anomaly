import os
import numpy as np
import cv2
import sys
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torch.distributions
from utils import get_dataset_list, CustomFaceDataset, cycle, get_val_hter, TPC_loss, get_score_from_filelist
from vgg_face_dag import vgg_face_dag, spoof_model
import bob.measure
import pdb
import torchvision.utils as tutil
import argparse
import time
from parameters import *
import matplotlib.pyplot as plt
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset_name", help="Train Dataset name")
parser.add_argument("--dev_dataset_name", help="Devel Dataset name")
parser.add_argument("--test_dataset_name", help="Test Dataset name")
parser.add_argument("--exp_name", help="Experiment name")
args = parser.parse_args()

before_model_name = ""
if args.train_dataset_name == 'replayattack':
    before_model_name += "RA"
elif args.train_dataset_name == 'replaymobile':
    before_model_name += "RM"
elif args.train_dataset_name == 'rose_youtu':
    before_model_name += "RY"
elif args.train_dataset_name == 'oulu_npu':
    before_model_name += "ON"
elif args.train_dataset_name == 'spoof_in_wild':
    before_model_name += "SW"
elif args.train_dataset_name == 'msu_mfsd':
    before_model_name += "MM"

before_model_name += "_"

if args.dev_dataset_name == 'replayattack':
    before_model_name += "RA"
elif args.dev_dataset_name == 'replaymobile':
    before_model_name += "RM"
elif args.dev_dataset_name == 'rose_youtu':
    before_model_name += "RY"
elif args.dev_dataset_name == 'oulu_npu':
    before_model_name += "ON"
elif args.dev_dataset_name == 'spoof_in_wild':
    before_model_name += "SW"
elif args.dev_dataset_name == 'msu_mfsd':
    before_model_name += "MM"

print(before_model_name)


# Load dev data
dev_real_list, dev_classes = get_dataset_list(args.train_dataset_name, 'real', 'train')
if args.train_dataset_name == 'rose_youtu':
    dev_real_list2, dev_classes1 = get_dataset_list(args.train_dataset_name, 'enroll', 'train')
    for item in dev_real_list2:
        dev_real_list.append(item)

dev_attack_list, dev_classes2 = get_dataset_list(args.train_dataset_name, 'attack', 'train')
assert dev_classes == dev_classes2, " dev classes dont match"
if len(dev_real_list)%2 != 0:
    del dev_real_list[-1]
if len(dev_attack_list) % 2 != 0:
    del dev_attack_list[-1]
print("Number of dev samples real: {0}, attack samples: {1}".format(len(dev_real_list), len(dev_attack_list)))

# Load test data
real_data_list, num_classes = get_dataset_list(args.test_dataset_name, 'real', 'test')
if len(real_data_list) % 2 != 0:
    del real_data_list[-1]
attack_data_list, num_classes2 = get_dataset_list(args.test_dataset_name, 'attack', 'test')
assert num_classes == num_classes2, "test classes dont match"
if len(attack_data_list) % 2 != 0:
    del attack_data_list[-1]
print("No of real samples(test): {0}, attack samples(test): {1}".format(len(real_data_list), len(attack_data_list)))

# load model
train_param = []
if run_parameters['model'] == 'vggface':
    weight_pth= os.path.join('models', args.exp_name, before_model_name + '_face_model.pth')
    vgg_face = vgg_face_dag(weights_path=weight_pth, return_layer=run_parameters['return_layer'])

    for p in vgg_face.parameters():
        p.requires_grad = False

    face_model = vgg_face
if run_parameters['model'] == 'vgg16':
    vgg16 = torchvision.models().vgg16(pretrained=True)

    face_model = vgg16


face_model.eval()

spoof_classifier = spoof_model(input_dim=run_parameters['dimension'],\
                               weights_path=os.path.join('models', args.exp_name, before_model_name + '_spoof_classifier.pth'))

for p in spoof_classifier.parameters():
    p.requires_grad = False

spoof_classifier.eval()

if run_parameters['multi_gpu']:
    face_model = nn.DataParallel(face_model)
    spoof_classifier = nn.DataParallel(spoof_classifier)

face_model.cuda()
spoof_classifier.cuda()

inm = nn.InstanceNorm1d(1, affine=False)

hter = get_val_hter(face_model, spoof_classifier, real_data_list, attack_data_list, run_parameters['apply_inm'], num_classes, run_parameters['train_batch_size'])
print(hter)