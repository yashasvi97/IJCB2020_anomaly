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
import bob.measure
import pdb
import torchvision.utils as tutil
import argparse
import torchvision
from parameters import *
import time
import scipy.io


parser = argparse.ArgumentParser()
parser.add_argument("--train_dataset_name", help="Train Dataset name")
parser.add_argument("--dev_dataset_name", help="Dev Dataset name")
parser.add_argument("--exp_name", help="Experiment name")
args = parser.parse_args()

np.set_printoptions(formatter='10.3f')
torch.set_printoptions(sci_mode=False, threshold=5000)

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

# for Fixing the seed
# torch.manual_seed(0)
#
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
#
# np.random.seed(0)

# Load train data
train_real_list, num_classes = get_dataset_list(args.train_dataset_name, 'real', 'train')
if args.train_dataset_name == 'rose_youtu':
    data_list1, num_classes1 = get_dataset_list(args.train_dataset_name, 'enroll', 'train')
    for item in data_list1:
        train_real_list.append(item)
train_attack_list, num_classes1 = get_dataset_list(args.train_dataset_name, 'attack', 'train')
assert num_classes == num_classes1, 'train classes dont match'
if len(train_real_list) %2 != 0:
    del train_real_list[-1]
if len(train_attack_list) % 2 != 0:
    del train_attack_list[-1]
print("No of training real samples: {0}, attack samples: {1}, num_classes: {2}".format(len(train_real_list), len(train_attack_list), num_classes) )

# Load test data
devel_real_list, test_num_classes1 = get_dataset_list(args.dev_dataset_name, 'real', 'devel')
if args.dev_dataset_name == 'rose_youtu':
    real_data_list1, tnc1 = get_dataset_list(args.dev_dataset_name, 'enroll', 'devel')
    for item in real_data_list1:
        devel_real_list.append(item)
devel_attack_list, test_num_classes2 = get_dataset_list(args.dev_dataset_name, 'attack', 'devel')
assert test_num_classes1 == test_num_classes2, "classes dont match in test data, check loader again"
test_num_classes = test_num_classes1
if len(devel_real_list) % 2 != 0:
    del devel_real_list[-1]
if len(devel_attack_list) % 2 != 0:
    del devel_attack_list[-1]
print("No of real samples(target): {0}, attack samples(target): {1}, number of dev classes: {2}".format(len(devel_real_list), len(devel_attack_list), test_num_classes))

# load model
train_param = []
face_model = None
if run_parameters['model'] == 'vggface':
    weight_pth = 'models/vgg_face_dag.pth'
    vgg_face = vgg_face_dag(weights_path=weight_pth, return_layer=run_parameters['return_layer'])

    for p in vgg_face.parameters():
        p.requires_grad = False
    print('Trainable parameters in vgg_face:', end=', ')
    for name, p in vgg_face.named_parameters():
        true_name = name.split(".")[0]
        if true_name in run_parameters['vgg_finetune'].split(","):
            print(true_name, end=', ')
            p.requires_grad = True
            train_param.append(p)
    print(" ")

    face_model = vgg_face
elif run_parameters['model'] == 'vgg16':
    vgg16 = torchvision.models().vgg16(pretrained=True)

    face_model = vgg16

# set train mode
face_model.train()

spoof_classifier = spoof_model(run_parameters['dimension'])
spoof_classifier.train()

if run_parameters['multi_gpu']:
    face_model = nn.DataParallel(face_model)
    spoof_classifier = nn.DataParallel(spoof_classifier)


if torch.cuda.is_available():
    face_model.cuda()
    spoof_classifier.cuda()

vgg_optim = optim.Adam(train_param, lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
spoof_optim = optim.Adam(spoof_classifier.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)

inm = nn.InstanceNorm1d(1, affine=False)

live_faces_dataset = CustomFaceDataset(train_real_list)
live_face_loader = DataLoader(live_faces_dataset, batch_size=run_parameters['train_batch_size'], shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss()
criterion.cuda()
best_hter = 100.0

f = 0
for ep in range(run_parameters['epoch']):
    if ep % run_parameters['print_epoch'] == 0:
        hter = get_val_hter(face_model, spoof_classifier, devel_real_list, devel_attack_list, run_parameters['apply_inm'], test_num_classes, run_parameters['test_batch_size'])
        if hter < best_hter:
            best_hter = hter
            f = 1
            save_name = os.path.join('models', args.exp_name, before_model_name + '_face_model.pth')
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            torch.save(face_model.state_dict(), save_name)
            save_name = os.path.join('models', args.exp_name, before_model_name + '_spoof_classifier.pth')
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            torch.save(spoof_classifier.state_dict(), save_name)

        print("Epoch {0}>>> HTER: {1}, best ACER: {2}".format(ep, hter, best_hter))


    face_model.train()
    spoof_classifier.train()

    for it, (data, _, _) in enumerate(live_face_loader):
        data = data.cuda()

        vgg_optim.zero_grad()
        spoof_optim.zero_grad()

        features = face_model(data)

        if run_parameters['apply_inm']:
            features = features.view(features.shape[0], 1, features.shape[-1])
            features = inm(features)
            features = features.view(features.shape[0], features.shape[-1])

        # spoof work
        if run_parameters['white_noise']:
            # push from origin
            sampler = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(run_parameters['dimension']).cuda(),  \
                                                                                 run_parameters['std_dev'] * torch.eye(run_parameters['dimension']).cuda())
        else:
            # push from shifted mean cluster
            if it == 0:
                old_mean = torch.zeros(run_parameters['dimension']).cuda()
            else:
                old_mean = mean_vector
            mean_vector = torch.mean(features, axis=0)
            # like running avg
            new_mean = run_parameters['alpha'] * old_mean + (1 - run_parameters['alpha']) * mean_vector

            if f == 1:
                save_name = os.path.join('models', args.exp_name, before_model_name + '_mean_vector.npy')
                scipy.io.savemat(save_name, {'mean':new_mean})
                f = 0
            sampler = torch.distributions.multivariate_normal.MultivariateNormal(new_mean,  \
                                                                                 run_parameters['std_dev'] * torch.eye(run_parameters['dimension']).cuda())
        # sample from pseudo-negative gaussian
        noise = sampler.sample((features.shape[0],))
        noise = noise.cuda()
        spoof_input = torch.cat([features, noise], dim=0)

        spoof_output = spoof_classifier(spoof_input)

        spoof_label = torch.cat([torch.zeros(features.shape[0]), torch.ones(noise.shape[0])], dim=0)
        spoof_label = spoof_label.cuda()
        spoof_label = spoof_label.long()

        if run_parameters['use_pc']:
            # Calculate TPC loss
            tpc_loss = TPC_loss(features)

        # Spoof Loss for classifier
        spoof_loss = criterion(spoof_output, spoof_label)

        if run_parameters['use_pc']:
            loss = run_parameters['lambda1'] * tpc_loss + run_parameters['lambda2'] * spoof_loss
        else:
            loss = run_parameters['lambda2'] * spoof_loss

        loss.backward()

        vgg_optim.step()
        spoof_optim.step()
        if it % 10 == 0:
            if run_parameters['use_pc']:
                print("TPC Loss: {0}, Spoof Loss: {1}, Total Loss: {2}".format(tpc_loss.item(), spoof_loss.item(), loss.item()))
            else:
                print("Spoof Loss: {0}, Total Loss: {1}".format(spoof_loss.item(), loss.item()))
