from CNN_ISOCC_copy import isocc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
tf = transforms.ToTensor()

import numpy as np
import librosa
import pandas as pd

from crs_lib import *
from tmr import *
from UNet_utils import *

import os
import PIL.Image
from copy import deepcopy as dc
import shutil
import csv
import pdb
import time
import random
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--epo', type=int, default=30)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--sch', type=float, default=0.99)

parser.add_argument('--dp', type=float, default=0.2)
parser.add_argument('--ich', type=int, default=16)

# Options: 44100 (or None), 44000, 16000, 8000
# If the default datasets are in 16kHz, then the max available sample rate is 16kHz
parser.add_argument('--sr', type=int, default=4000)
parser.add_argument('--nfft', type=int, default=256)

parser.add_argument('--pin', type=bool, default=True)
parser.add_argument('--nw', type=int, default=0)

args = parser.parse_args()

sr_set = args.sr
nfft = args.nfft
pin_setting = args.pin
nw_set = args.nw
cur_lr = args.lr
ich = args.ich

if torch.cuda.is_available():
    device = "cuda"
    if args.gpu != 0:
        torch.cuda.set_device(args.gpu)
else:
    device = "cpu"

# Master directory
# master_dir = 'D:/D-Drive Files/000_data_backup'
# master_dir = 'E:/chlee'
master_dir = '../../../project/hw-team/chlee'

write_time = record_start('time_record')
train_ratio = 7

cur_dataset_dir = opj(master_dir, 'esc10_rename_vol_norm_4k')
classes = os.listdir(cur_dataset_dir)
classes.sort()
num_classes = len(classes)

single_file_num = len(os.listdir(opj(cur_dataset_dir, classes[0])))

sample_audio_dir = opj(cur_dataset_dir, 'chainsaw/chainsaw0.wav')
sample_audio, _ = librosa.load(sample_audio_dir, sr=sr_set)
ov_audio_len = sample_audio.shape[0]
none_audio = audio_STFT_lib(sample_audio, nfft)
num_ch, sz_y, sz_x = none_audio.shape
sample_audio = []
none_audio = []

print("Input image info: %d x %d x %d" % (num_ch, sz_x, sz_y))
print()


def img_loader(in_address):
    converted, _ = librosa.load(in_address, sr=sr_set)
    return audio_STFT_lib(converted, nfft)
    

def cls_list(in_classes):
    out_list = torch.zeros(num_classes)
    for in_cls in in_classes:
        in_cls_num = classes.index(in_cls)
        out_list[in_cls_num] = 1
    return out_list
    


img_train = []
img_test = []
ratio_cnt = 0
load_cnt = 0
for cn in range(num_classes):
    for cur_file in range(single_file_num):
        the_address = opj(cur_dataset_dir, classes[cn], classes[cn] + str(cn) + '.wav')
        
        if ratio_cnt < train_ratio:
            img_train.append((img_loader(the_address), cls_list([classes[cn]])))
            ratio_cnt += 1
        elif ratio_cnt < 9:
            img_test.append((img_loader(the_address), cls_list([classes[cn]])))
            ratio_cnt += 1
        else:
            img_train.append((img_loader(the_address), cls_list([classes[cn]])))
            ratio_cnt = 0
        load_cnt += 1
        print("Loading %d/%d" % (load_cnt, num_classes * single_file_num), end='\r')

tr_dl = DataLoader(img_train, batch_size=80, pin_memory=pin_setting, shuffle=True, num_workers=nw_set)
tr_len = len(tr_dl.dataset)
te_dl = DataLoader(img_test, batch_size=len(img_test), pin_memory=pin_setting, num_workers=nw_set)
te_len = len(te_dl.dataset)

model = isocc(in_dim=num_ch, in_y=sz_y, in_x=sz_x, num_out=num_classes, init_out=ich).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: args.sch ** epoch)
loss_fn = nn.BCELoss().to(device)


def list_ans(in_list, in_num):
    t_list = in_list.tolist()
    out_ans = []
    for _ in range(in_num):
        cur_idx = t_list.index(max(t_list))
        out_ans.append(cur_idx)
        t_list[cur_idx] = -1
    return out_ans


def train_test(in_label_num, in_epo=args.epo):
    global model, tr_dl, te_dl
    model.train()
    in_losses = []
    in_accs = []
    for cur_epo in range(in_epo):
        for data_in, data_out in tr_dl:
            data_in = data_in.to(device)
            data_out = data_out.to(device)
            
            predicted = model(data_in)
            #pdb.set_trace()
            cur_loss = loss_fn(predicted, data_out)
            in_losses.append(cur_loss.item())
            
            for ii in range(len(data_out)):
                if list_ans(predicted[ii], in_label_num) == list_ans(data_out[ii], in_label_num):
                    in_accs.append(1)
                else:
                    in_accs.append(0)
            
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
        
        print("Epoch %d - Accuracy: %.3f, Loss: %.3f" % (cur_epo + 1, sum(in_accs)/len(in_accs)*100, sum(in_losses)/len(in_losses)))
    
    model.eval()
    in_losses = []
    in_accs = []
    with torch.no_grad():
        for data_in, data_out in te_dl:
            data_in = data_in.to(device)
            data_out = data_out.to(device)
            
            predicted = model(data_in)
            
            cur_loss = loss_fn(predicted, data_out)
            in_losses.append(cur_loss.item())
            
            for ii in range(len(data_out)):
                if list_ans(predicted[ii], in_label_num) == list_ans(data_out[ii], in_label_num):
                    in_accs.append(1)
                else:
                    in_accs.append(0)
    
    print("Testing Accuracy: %.3f, Loss: %.3f" % (sum(in_accs)/len(in_accs)*100, sum(in_losses)/len(in_losses)))
    
    return sum(in_accs)/len(in_accs)*100, sum(in_losses)/len(in_losses)
            

print()
print("Single-label Training and Testing Started")
training_timer = time.time()
single_acc, single_loss = train_test(1)
print("Spent -", end(training_timer))

torch.save(model.state_dict(), './isocc_classification/isocc_followed_single.pth')

print(model.parameters())
