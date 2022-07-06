from asc_model import asc
# from asc_model_4 import asc
# from asc_model_4_lstm import asc
# from asc_model_BLSTM import asc
# from asc_model_BLSTM_orig import asc
# from asc_model_LSTM_orig import asc
# from asc_model_LSTM import asc
# from asc_model_LSTM_4layers import asc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import librosa
import pandas as pd

from crs_lib import *
from tmr import *
from UNet_utils import *

import os
from copy import deepcopy as dc
import shutil
import csv
import pdb
import time
import random
import argparse
parser = argparse.ArgumentParser()

# Options: 44100 (or None), 44000, 16000, 8000
# If the default datasets are in 16kHz, then the max available sample rate is 16kHz
parser.add_argument('--sr', type=int, default=4000)
# Options: 1, 2, 3, 4, 5
parser.add_argument('--set', type=int, default=2)
parser.add_argument('--over', type=int, default=5)
parser.add_argument('--nfft', type=int, default=256)
parser.add_argument('--vn', type=bool, default=True)

parser.add_argument('--epo', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--val', type=int, default=10)
parser.add_argument('--sch', type=float, default=0.99)
parser.add_argument('--bs', type=int, default=200)

parser.add_argument('--dp', type=float, default=0.1)
parser.add_argument('--ich', type=int, default=8)

parser.add_argument('--full', type=bool, default=False)
parser.add_argument('--pin', type=bool, default=True)
parser.add_argument('--nw', type=int, default=0)

args = parser.parse_args()

sr_set = args.sr
cur_lr = args.lr
nfft = args.nfft
pin_setting = args.pin
nw_set = args.nw


if torch.cuda.is_available():
    device = "cuda"
    if args.gpu != 0:
        torch.cuda.set_device(args.gpu)
else:
    device = "cpu"


# Master directory
# master_dir = 'D:/D-Drive Files/000_data_backup'
master_dir = 'E:/chlee'
# master_dir = '../../../project/hw-team/chlee'

write_time = record_start('time_record')

# Dataset setting
if sr_set == 4000:
    if args.vn:
        multi_dataset = opj(master_dir, 'esc10_2_all_mixes_ver_vol_norm_4k')
        single_dataset = opj(master_dir, 'esc10_rename_vol_norm_4k')
    else:
        multi_dataset = opj(master_dir, 'esc10_2_all_mixes')
        single_dataset = opj(master_dir, 'esc10_rename_4k')
elif sr_set == 8000:
    multi_dataset = opj(master_dir, 'esc10_2_all_mixes_ver_vol_norm_8k')
    single_dataset = opj(master_dir, 'esc10_rename_8k_vol_norm')
elif sr_set == 16000:
    multi_dataset = opj(master_dir, 'esc10_2_all_mixes_ver_vol_norm_16k')
    single_dataset = opj(master_dir, 'esc10_rename_16k_vol_norm')
elif sr_set == 44100:
    multi_dataset = opj(master_dir, 'esc10_2_all_mixes_ver_vol_norm')
    single_dataset = opj(master_dir, 'esc10_rename_vol_norm')
else:
    # Just use 44.1kHz
    multi_dataset = opj(master_dir, 'esc10_2_all_mixes_ver_vol_norm')
    single_dataset = opj(master_dir, 'esc10_rename_vol_norm')
max_num_mixture = 2

# Class setting
classes = os.listdir(single_dataset)
classes.sort()
num_class = len(classes)
cls_file_len = len(os.listdir(opj(single_dataset, 'chainsaw')))
all_file_len = cls_file_len * num_class

# Size Check
sample_audio_dir = opj(single_dataset, 'chainsaw/chainsaw0.wav')
sample_audio, _ = librosa.load(sample_audio_dir, sr=sr_set)
ov_audio_len = sample_audio.shape[0]
none_audio = audio_STFT_lib(sample_audio, nfft)
sz_ch, sz_y, sz_x = none_audio.shape
sample_audio = []
none_audio = []

print("Audio Specification: %ds (%d), %dHz" % (ov_audio_len/sr_set, ov_audio_len, sr_set))
print("STFT Specification: %d x %d x %d" % (sz_ch, sz_y, sz_x))
print()

# K-fold settings
set_name = 'set' + str(args.set)
k_fold_sets_dir = 'esc10_811div_txts'

r = open(opj(k_fold_sets_dir, set_name, set_name + '_train.txt'), 'r')
train_nums = list(map(int, r.readline().split(', ')))
single_train_len = len(train_nums)
r.close()

r = open(opj(k_fold_sets_dir, set_name, set_name + '_val.txt'), 'r')
val_nums = list(map(int, r.readline().split(', ')))
val_single_len = len(val_nums)
r.close()

# Load single audio
# 0 indicates Single-label audio
single_audio_train = []
single_audio_val = []
for cur_cls in range(num_class):
    access_cls = opj(single_dataset, classes[cur_cls])

    for cur_file in range(cls_file_len):
        file_original = classes[cur_cls] + str(cur_file) + '.wav'
        access_file = opj(access_cls, file_original)

        if cur_file in train_nums:
            single_audio_train.append((access_file, 0))
        
        elif cur_file in val_nums:
            single_audio_val.append((access_file, 0))

print("Single-label audio: %d-train, %d-validation" % (len(single_audio_train), len(single_audio_val)))

train_oversample = len(single_audio_train) * args.over
print("Single-label Training Audio Oversampling: %d" % (train_oversample))

# Multi-label (2-class)
multi_csv = './esc10_mixes_v2.csv'
mf = pd.read_csv(multi_csv)
cur_data = np.array(mf['filename'])
mix_a_cls = np.array(mf['A_class'])
mix_a_num = np.array(mf['A_num'])
mix_b_cls = np.array(mf['B_class'])
mix_b_num = np.array(mf['B_num'])
set_info = np.array(mf[set_name])
all_mix_len = len(cur_data)

"""
1: Train, 2: Validation, 3: Test, -1: None
1 indicates 2-class mixture
"""
mix_train_filenames = []
mix_val_filenames = []
for data_num in range(all_mix_len):
    if set_info[data_num] == 1 or set_info[data_num] == 2:
        access_file = opj(multi_dataset, cur_data[data_num] + '.wav')
        
        if set_info[data_num] == 1:
            mix_train_filenames.append((access_file, 1))
        else:
            mix_val_filenames.append((access_file, 1))
            
print("Multi-label audio: %d-train, %d-validation" % (len(mix_train_filenames), len(mix_val_filenames)))

# Undersample
mix_unselected = []
mix_selected = []
called_nums = []
for _ in range(train_oversample):
    picked_num = random.randint(0, len(mix_train_filenames)-1)
    while picked_num in called_nums:
        picked_num = random.randint(0, len(mix_train_filenames)-1)
    called_nums.append(picked_num)

for rn in called_nums:
    mix_selected.append(dc(mix_train_filenames[rn]))
for us in range(len(mix_train_filenames)):
    if us not in called_nums:
        mix_unselected.append(dc(mix_train_filenames[us]))
        

for _ in range(args.over):
    mix_selected.extend(dc(single_audio_train))


# Returns the lists of (input, output) data
def uploader(in_info):
    upload = []
    in_len = len(in_info)
    for fn in range(in_len):
        get_audio, _ = librosa.load(in_info[fn][0], sr=sr_set)
        audio_stft = audio_STFT_lib(get_audio, nfft)
        upload.append((audio_stft, in_info[fn][1]))
        print("Loading %d/%d" % (fn+1, in_len), end="\t\t\r")
    print()
    
    return upload


# Loads the final data
tr_timer = time.time()
tr_dl = DataLoader(uploader(mix_selected), batch_size=args.bs, pin_memory=pin_setting, shuffle=True, num_workers=nw_set)
mix_train_filenames = []
print("Training dataset Loading time - ", end(tr_timer))

va_timer = time.time()
va_dl = DataLoader(uploader(mix_val_filenames), batch_size=len(mix_val_filenames), pin_memory=pin_setting, num_workers=nw_set)
mix_val_filenames = []
va_dl_2 = DataLoader(uploader(single_audio_val), batch_size=len(single_audio_val), pin_memory=pin_setting, num_workers=nw_set)
single_audio_val = []
print("Validation dataset Loading time - ", end(va_timer))

# Model settings
model = asc(in_dim=sz_ch, in_y=sz_y, in_x=sz_x, num_out=max_num_mixture, in_dp=args.dp, init_out=args.ich).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: args.sch ** epoch)
loss_fn = nn.CrossEntropyLoss().to(device)

print()
print("Training started")
training_timer = time.time()
val_rec = 0
val_sets = [0, 0]
if args.gpu == 0:
    model_filename = './asc_output_models/' + write_time + '_model.pth'
else:
    model_filename = './asc_output_models_other_gpu/' + write_time + '_model.pth'
    
for cur_epo in range(args.epo):
    model.train()
    tr_accs = []
    tr_losses = []
    for tr_in, tr_out in tr_dl:
        tr_in = tr_in.to(device)
        tr_out = tr_out.to(device)

        tr_pred = model(tr_in)
        
        tr_loss = loss_fn(tr_pred, tr_out)
        tr_losses.append(tr_loss.item())
        
        tr_cor = (tr_pred.argmax(1) == tr_out).type(torch.float).sum().item()
        tr_cor /= tr_pred.shape[0]
        tr_accs.append(tr_cor)
        
        optimizer.zero_grad()
        tr_loss.backward()
        optimizer.step()
            
    print("Epoch %d - Accuracy: %.3f, Loss: %.3f" % (cur_epo + 1, sum(tr_accs)/len(tr_accs)*100, sum(tr_losses)/len(tr_losses)))
    
    if (cur_epo + 1) % args.val == 0:
        model.eval()
        with torch.no_grad():
            temp_val = 0
            temp_sets = [0, 0]
            for cur_tag, cur_dl in [('Multi', va_dl), ('Single', va_dl_2)]:
                ov_losses = []
                ov_accs = []
                for v_in, v_out in cur_dl:
                    v_in = v_in.to(device)
                    v_out = v_out.to(device)

                    v_pred = model(v_in)
                    
                    ov_losses.append(loss_fn(v_pred, v_out).item())

                    ov_cor = (v_pred.argmax(1) == v_out).type(torch.float).sum().item()
                    ov_cor /= v_pred.shape[0]
                    ov_accs.append(ov_cor)
                
                temp_val += sum(ov_accs)/len(ov_accs)*50
                
                if cur_tag == 'Multi':
                    temp_sets[1] = sum(ov_accs)/len(ov_accs)*100
                else:
                    temp_sets[0] = sum(ov_accs)/len(ov_accs)*100
                
                print("%s Validation %d - Accuracy: %.3f, Loss: %.3f" % (cur_tag, (cur_epo + 1)//args.val, sum(ov_accs)/len(ov_accs)*100, sum(ov_losses)/len(ov_losses)))
            
            if temp_val > val_rec:
                val_rec = temp_val
                val_sets = temp_sets
                torch.save(model.state_dict(), model_filename)
                
        print()

print("The best average of Validation:", val_rec)
print("Single:", val_sets[0])
print("Multi:", val_sets[1])
print("Total training time:", end(training_timer))
print()

tr_in = []
tr_out = []
tr_pred = []
v_in = []
v_out = []
v_pred = []

"""
While undersampling, some of multi-label audio is not selected for training.
The following part tests with unselected multi-label audio files, to verify whether undersampling affects the overall performance.
"""
if args.full:
    model = asc(in_dim=sz_ch, in_y=sz_y, in_x=sz_x, num_out=max_num_mixture, in_dp=args.dp, init_out=args.ich).to(device)
    model.load_state_dict(torch.load(model_filename))

    tr_in = []
    tr_out = []
    tr_pred = []
    v_in = []
    v_out = []
    v_pred = []
    va_dl = DataLoader(uploader(mix_unselected), batch_size=100, pin_memory=pin_setting, num_workers=nw_set)
    model.eval()
    ov_losses = []
    ov_accs = []
    with torch.no_grad():
        for v_in, v_out in va_dl:
            v_in = v_in.to(device)
            v_out = v_out.to(device)

            v_pred = model(v_in)
            
            ov_losses.append(loss_fn(v_pred, v_out).item())

            ov_cor = (v_pred.argmax(1) == v_out).type(torch.float).sum().item()
            ov_cor /= v_pred.shape[0]
            ov_accs.append(ov_cor)

        print("Unselected Multi-label - Accuracy: %.3f, Loss: %.3f" % (sum(ov_accs)/len(ov_accs)*100, sum(ov_losses)/len(ov_losses)))

# Testing phase
r = open(opj(k_fold_sets_dir, set_name, set_name + '_test.txt'), 'r')
test_nums = list(map(int, r.readline().split(', ')))
single_test_len = len(test_nums)
r.close()

# Load single audio
single_audio_test = []
for cur_cls in range(num_class):
    access_cls = opj(single_dataset, classes[cur_cls])
    for cur_file in range(cls_file_len):
        if cur_file in test_nums:
            file_original = classes[cur_cls] + str(cur_file) + '.wav'
            access_file = opj(access_cls, file_original)
            single_audio_test.append((access_file, 0))

"""
1: Train, 2: Validation, 3: Test, -1: None
1 indicates 2-class mixture
"""
mix_test_filenames = []
for data_num in range(all_mix_len):
    if set_info[data_num] == 3:
        access_file = opj(multi_dataset, cur_data[data_num] + '.wav')
        mix_test_filenames.append((access_file, 1))

test_timer = time.time()
test_dl_single = DataLoader(uploader(single_audio_test), batch_size=len(mix_test_filenames), pin_memory=pin_setting, num_workers=nw_set)
test_dl_multi = DataLoader(uploader(mix_test_filenames), batch_size=len(mix_test_filenames), pin_memory=pin_setting, num_workers=nw_set)
mix_test_filenames = []
print("Testing dataset Loading time - ", end(test_timer))

print()
print("Testing started")
    
model.eval()
with torch.no_grad():
    for cur_tag, test_dl in [('Single', test_dl_single), ('Multi', test_dl_multi)]:
        ov_losses = []
        ov_accs = []
        for bn, (v_in, v_out) in enumerate(test_dl):
            v_in = v_in.to(device)
            v_out = v_out.to(device)

            v_pred = model(v_in)
            
            ov_losses.append(loss_fn(v_pred, v_out).item())
            
            ans_preds = v_pred.argmax(1)
            
            for chk in range(v_pred.shape[0]):
                cur_pred = ans_preds[chk].item()
                cur_ans = v_out[chk].item()
                
                check_ans = "correct" if cur_pred == cur_ans else "incorrect"
                
                # print("%d %s - Pred %d, Actual %d" % (chk, check_ans, cur_pred, cur_ans))
                
            ov_cor = (v_pred.argmax(1) == v_out).type(torch.float).sum().item()
            ov_cor /= v_pred.shape[0]
            ov_accs.append(ov_cor)
        
        print("%s Accuracy: %.3f, Loss: %.3f" % (cur_tag, sum(ov_accs)/len(ov_accs)*100, sum(ov_losses)/len(ov_losses)))
