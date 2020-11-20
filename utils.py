import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from vgg_face_dag import Vgg_face_dag as vgg_face_class
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import bob.measure
import scipy.io
import pdb
import dlib
from imutils.face_utils import FaceAligner, rect_to_bb
from parameters import dataset_parameters as data_param
torch.set_printoptions(sci_mode=False, threshold=5000)

vgg_face = vgg_face_class(return_layer='fc6')

detector = dlib.get_frontal_face_detector()

def get_dataset_list(dataset, type_, mode):
    data_dir = data_param[dataset]['path']
    if not os.path.exists(data_dir):
        print(data_dir)
        print("No path")
        exit()

    data_list = []
    num_classes = len(data_param[dataset][mode]['id_list'])
    id_list = data_param[dataset][mode]['id_list']
    id_dic = data_param[dataset][mode]['id_dic']
    if dataset == 'replayattack' or dataset == 'replaymobile':
        data_dir = os.path.join(data_dir, mode, type_)
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                if 'DS_Store' in filename:
                    continue
                filepath = os.path.join(root, filename)
                ind = filepath.find('client')
                ind += 6
                class_id = int(filepath[ind:ind+3])
                face = plt.imread(filepath)
                lab_id = 1 if (type_ == 'attack') else 0
                data_list.append({'face': face, 'label_id': lab_id, 'class_id': id_dic[class_id]})
    elif dataset == 'rose_youtu':
        for id_ in id_list:
            mat_pth = os.path.join(data_dir, str(id_), type_+'.mat')
            mat = scipy.io.loadmat(mat_pth)
            paths = mat['paths']
            faces = mat['data'][0]
            for dat, path in zip(faces, paths):
                lab_id = 1 if (type_ == 'attack') else 0
                data_list.append({'face': dat, 'label_id': lab_id})
    elif dataset == 'oulu_npu':
        if mode == 'train':
            data_dir = os.path.join(data_dir, 'Train_faces')
        elif mode == 'devel':
            data_dir = os.path.join(data_dir, 'Dev_faces')
        else:
            data_dir = os.path.join(data_dir, 'Test_faces')

        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                img_type = int(filename.split("_")[-1].split(".")[0])
                if (type_ == 'enroll' or type_ == 'real') and img_type == 1:
                    mat = scipy.io.loadmat(filepath)['Faces'][0]
                    for dat in mat:
                        lab_id = 1 if (type_ == 'attack') else 0
                        data_list.append({'face': dat, 'label_id': lab_id})
                elif type_ == 'attack' and img_type != 1:
                    mat = scipy.io.loadmat(filepath)['Faces'][0]
                    for dat in mat:
                        lab_id = 1 if (type_ == 'attack') else 0
                        data_list.append({'face': dat, 'label_id': lab_id})
    elif dataset == 'spoof_in_wild':
        if mode == 'train' or mode == 'devel':
            data_dir = os.path.join(data_dir, 'Train')
        else:
            data_dir = os.path.join(data_dir, 'Test')

        if type_ == 'real':
            data_dir = os.path.join(data_dir, 'live')
        elif type_ == 'attack':
            data_dir = os.path.join(data_dir, 'spoof')

        id_list = sorted([int(i) for i in os.listdir(data_dir)])
        num_classes = len(id_list)
        for idx, id_ in enumerate(id_list):
            id_dic[id_] = idx
        for root, dirs, filenames in os.walk(data_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                if 'real' in filename or 'enroll' in filename:
                    continue
                # id_ = int(filename.split("-")[0])
                faces = scipy.io.loadmat(filepath)['faces']
                if len(faces) > 0:
                    faces = faces[0]
                    for face_enum, dat in enumerate(faces):
                        if face_enum %4 == 0:
                            if dat.shape[0] > 0:
                                lab_id = 1 if (type_ == 'attack') else 0
                                data_list.append({'face': dat, 'label_id':lab_id})
    elif dataset == 'msu_mfsd':
        if type_ == 'real':
            data_dir = os.path.join(data_dir, 'real')
        elif type_ == 'attack':
            data_dir = os.path.join(data_dir, 'attack')

        folders = os.listdir(data_dir)
        for fold in folders:
            id_ = int(fold.split('_')[1][6:])
            if id_ in id_list:
                orig_ = len(data_list)
                # print(orig_)
                for root, dirs, filenames in os.walk(os.path.join(data_dir, fold)):
                    for idx, filename in enumerate(filenames):
                        if idx %2 == 0:
                            filepath = os.path.join(root, filename)
                            img = plt.imread(filepath)
                            if idx == 0:
                                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                                rects = detector(gray, 2)
                                if len(rects) != 0:
                                    (x, y, w, h) = rect_to_bb(rects[0])
                                    face = img[y:y+h, x:x+w]
                            else:
                                face = img[y:y+h, x:x+w]
                            lab_id = 1 if (type_ == 'attack') else 0
                            if type_ == 'attack':
                                if idx %4 == 0:
                                    data_list.append({'face': face, 'label_id': lab_id})
                            else:
                                data_list.append({'face': face, 'label_id': lab_id})

    else:
        print("not implemented :( !")
        exit()

    return data_list, num_classes

def load_face(faceitem):
    face = faceitem['face']
    class_id = faceitem['class_id']
    lab = faceitem['label_id']
    return face, lab, class_id

class CustomFaceDataset(Dataset):
    def __init__(self, data_list):
        self.facelist = data_list
        #use 0-255 as opposed to 0-1 for this pretrained vggface network
        normalize = transforms.Normalize(mean=vgg_face.meta['mean'], std=vgg_face.meta['std'])

        trans_list = [transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(size=(224, 224))]

        trans_list += [transforms.ToTensor()]
        trans_list += [lambda x: x * 255.0]

        trans_list.append(normalize)
        self.transform = transforms.Compose(trans_list)

    def __getitem__(self, index):
        facedata, facelabel, class_label = load_face(self.facelist[index])
        facedata = self.transform(facedata)
        facedata = facedata.type(torch.FloatTensor)
        # return facedata, self.facelist[index]['filepath']
        return facedata, facelabel, class_label

    def __len__(self):
        return len(self.facelist)

def cycle(dataloader):
    while True:
        for data, lab in dataloader:
            yield data, lab

def get_val_hter(vgg_face, spoof_classifier, real_data_list, attack_data_list, apply_inm, num_classes, test_batch_size):
    spoof_classifier.eval()
    vgg_face.eval()

    inm = nn.InstanceNorm1d(1, affine=False)

    tmp_bch_sz = test_batch_size

    i_list = []
    for item in real_data_list:
        tmp_dic = {}
        tmp_dic['face'] = item['face']
        tmp_dic['class_id'] = item['class_id']
        tmp_dic['label_id'] = 0
        i_list.append(tmp_dic)
    for item in attack_data_list:
        tmp_dic = {}
        tmp_dic['face'] = item['face']
        tmp_dic['class_id'] = item['class_id']
        tmp_dic['label_id'] = 1
        i_list.append(tmp_dic)

    face_dataset = CustomFaceDataset(i_list)
    val_loader = DataLoader(face_dataset, batch_size=tmp_bch_sz, num_workers=4)

    all_scores = np.zeros((len(i_list), 2))
    for idx, (data, lab, _) in enumerate(val_loader):
        data = data.cuda()
        out = vgg_face(data)
        if apply_inm:
            out = out.view(out.shape[0], 1, out.shape[-1])
            out = inm(out)
            out = out.view(out.shape[0], out.shape[-1])

        prob = spoof_classifier(out)

        sc_val = prob[:, 0].detach().cpu().numpy()
        lb_val = lab[:].numpy()

        lower = idx * tmp_bch_sz
        upper = lower + data.shape[0]
        if upper > all_scores.shape[0]:
            upper = all_scores.shape[0]

        all_scores[lower:upper, 0] = sc_val
        all_scores[lower:upper, 1] = lb_val

    all_values = np.array([p for p, _ in all_scores])
    all_labels = np.array([q for _, q in all_scores])

    ind0 = all_labels == 0
    ind1 = all_labels == 1
    positives = all_values[ind0]
    negatives = all_values[ind1]

    positives = np.array(positives, dtype=np.float)
    negatives = np.array(negatives, dtype=np.float)
    positives.sort()
    negatives.sort()

    # normalize
    scores = np.concatenate((positives, negatives), axis=0)
    scores = (scores - np.min(scores)) / float(np.max(scores) - np.min(scores))

    positives = scores[:positives.shape[0]]
    negatives = scores[positives.shape[0]:]

    T = bob.measure.min_hter_threshold(negatives, positives, is_sorted=True)
    fpr, fnr = bob.measure.fprfnr(negatives, positives, T)
    hter = (fpr + fnr) / 2.0
    hter = 100.0 * float(hter)

    return hter

def TPC_loss(features):
    # loss taken from https://github.com/abhimanyudubey/confusion
    batch_size = features.shape[0]
    assert batch_size % 2 == 0, "batch not even"
    left_batch = features[:int(0.5*features.shape[0])]

    right_batch = features[int(0.5*features.shape[0]):]

    int_norm = torch.norm((left_batch - right_batch).abs(), 2, 1)
    int_prod = int_norm
    l = int_prod.sum() / float(batch_size)
    return l


'''
# not sure if this function useful, use with caution, haven't tested
def get_score_from_filelist(vgg_face, spoof_classifier, i_list, apply_inm, test_batch_size):
    spoof_classifier.eval()
    vgg_face.eval()

    inm = nn.InstanceNorm1d(1, affine=False)

    tmp_bch_sz = test_batch_size

    face_dataset = CustomFaceDataset(i_list)
    val_loader = DataLoader(face_dataset, batch_size=tmp_bch_sz, num_workers=4, shuffle=False)

    all_scores = np.zeros((len(i_list), 2))

    p = np.arange(len(i_list))
    all_scores[:, 0] = p

    for idx, (data, lab, _) in enumerate(val_loader):
        # pdb.set_trace()
        data = data.cuda()
        out = vgg_face(data)
        if apply_inm:
            out = out.view(out.shape[0], 1, out.shape[-1])
            out = inm(out)
            out = out.view(out.shape[0], out.shape[-1])

        prob = spoof_classifier(out)
        sc_val = prob[:, 0].detach().cpu().numpy()

        lower = idx * tmp_bch_sz
        upper = lower + data.shape[0]
        if upper > all_scores.shape[0]:
            upper = all_scores.shape[0]

        all_scores[lower:upper, 1] = sc_val

    return all_scores
'''