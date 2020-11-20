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
import torch.distributions
from utils import get_dataset_list, CustomFaceDataset, cycle, get_val_hter, TPC_loss
from vgg_face_dag import vgg_face_dag, spoof_model
from parameters import *
import bob.measure
import pdb
import torchvision.utils as tutil
import torchvision
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset_name", help="Dataset name")
parser.add_argument("--dev_dataset_name", help="Dataset name")
parser.add_argument("--test_dataset_name", help="Dataset name")
parser.add_argument("--exp_name", help="Dataset name")
parser.add_argument("--vis_class", help="classes to visualize", type=int)
args = parser.parse_args()

if args.train_dataset_name != 'replayattack':
    print('Only replayattack implemented')
    exit(1)

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

# torch.manual_seed(0)
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# np.random.seed(0)

# Load train data
live_data_list = []
spoof_data_list = []
dataset_list, num_classes = get_dataset_list(args.train_dataset_name, 'real', 'train')
att_dataset_list, _ = get_dataset_list(args.train_dataset_name, 'attack', 'train')
for item in att_dataset_list:
    dataset_list.append(item)
print(num_classes)
if args.vis_class > num_classes:
    print('visualization class not in this part of dataset')
    exit(1)
else:
    live_data_list = []
    spoof_data_list = []
    for item in dataset_list:
        if item['class_id'] == args.vis_class:
            temp_dic = {}
            temp_dic['face'] = item['face']
            temp_dic['label_id'] = item['label_id']
            temp_dic['class_id'] = item['class_id']
            if item['label_id'] == 0:
                live_data_list.append(temp_dic)
            else:
                spoof_data_list.append(temp_dic)

face_model = None
if run_parameters['model'] == 'vggface':
    weight_pth=os.path.join('models', args.exp_name,  before_model_name + '_face_model.pth')

    print(weight_pth)
    vgg_face = vgg_face_dag(weights_path=weight_pth, return_layer=run_parameters['return_layer'])

    for p in vgg_face.parameters():
        p.requires_grad = False
    face_model = vgg_face
if run_parameters['model'] == 'vgg16':
    vgg16 = torchvision.models().vgg16(pretrained=True)
    face_model = vgg16

face_model.eval()

if run_parameters['multi_gpu']:
    face_model = nn.DataParallel(face_model)


face_model.cuda()

inm = nn.InstanceNorm1d(1, affine=False)

print("No of training samples: {0}, {1}".format(len(live_data_list), len(spoof_data_list)) )
# exit()
data_cell = {}
feat_cell = {}
# for k in args.vis_class.split(","):
    # i_k = int(k)
    # live_id_list = [item for item in live_data_list if item['label_id'] == i_k]
    # feat_cell[i_k] = []
k = args.vis_class
feat_cell[k] = []
live_faces_dataset = CustomFaceDataset(live_data_list)
live_face_loader = DataLoader(live_faces_dataset, batch_size=80, shuffle=True, num_workers=4)

print(len(live_data_list))


for it, (data, _, _) in enumerate(live_face_loader):
    data = data.cuda()
    # lab = lab.cuda()

    features = face_model(data)

    if run_parameters['apply_inm']:
        features = features.view(features.shape[0], 1, features.shape[-1])
        features = inm(features)
        features = features.view(features.shape[0], features.shape[-1])

    feat_cell[k].append(features.detach().cpu().numpy())


print(len(spoof_data_list))
data_cell[k] = (len(live_data_list), len(spoof_data_list))
spoof_faces_dataset = CustomFaceDataset(spoof_data_list)
spoof_face_loader = DataLoader(spoof_faces_dataset, batch_size=80, shuffle=True, num_workers=4)

for it, (data, _, _) in enumerate(spoof_face_loader):
    data = data.cuda()
    # lab = lab.cuda()

    features = face_model(data)

    if run_parameters['apply_inm']:
        features = features.view(features.shape[0], 1, features.shape[-1])
        features = inm(features)
        features = features.view(features.shape[0], features.shape[-1])

    feat_cell[k].append(features.detach().cpu().numpy())


feat_cell[k] = np.concatenate(feat_cell[k])
print(feat_cell[k].shape)


sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.ones(run_parameters['dimension']).cuda(), \
                                                                     run_parameters['std_dev'] * torch.eye(run_parameters['dimension']).cuda())

t = 20
all_feat = []
neg_feat = sampler.sample((t, ))
all_feat.append(neg_feat.detach().cpu().numpy())

# for k in args.vis_class.split(","):
all_feat.append(feat_cell[k])
all_feat = np.concatenate(all_feat)
print(all_feat.shape)




all_emb = TSNE(perplexity=10).fit_transform(all_feat)
# t = 500
plt.scatter(all_emb[:t, 0], all_emb[:t, 1], label='pseudo negative', color='b')
# for k in args.vis_class.split(","):
#     i_k = int(k)
(real_number, attack_numb) = data_cell[k]
plt.scatter(all_emb[t:t+real_number, 0], all_emb[t:t+real_number, 1], label='class {0} real data'.format(k), color='g')
t = t+ real_number
plt.scatter(all_emb[t:t+attack_numb, 0], all_emb[t:t+attack_numb, 1], label='class {0} attacked data'.format(k), color='r')
t = t + attack_numb
plt.legend()
plt.show()
