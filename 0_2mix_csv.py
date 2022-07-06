import csv
import os

import pdb

from crs_lib import *

# Master directory
# master_dir = 'D:/D-Drive Files/000_data_backup'
# master_dir = 'E:/chlee'
master_dir = './dataset'

single_dataset = opj(master_dir, 'esc10_rename')
classes = os.listdir(single_dataset)
classes.sort()
num_class = len(classes)
cls_file_len = len(os.listdir(opj(single_dataset, classes[0])))

csv_file = open('esc10_mixes_v2.csv', 'w', newline='')
wr = csv.writer(csv_file)

# Load set division information
k_fold_sets_dir = 'esc10_811div_txts'
k_fold_sets = os.listdir(k_fold_sets_dir)
k_fold_sets.sort()

"""
Each row contains information about each mixture audio

Filename, A-class, A-class file number, B-class, B-class file number, set 1 ~ 5
"""
write_row = ['filename', 'A_class', 'A_num', 'B_class', 'B_num']
write_row.extend(k_fold_sets)
wr.writerow(write_row)

trs = []
vas = []
tes = []

p_cnt = 1

for a_cls in range(num_class - 1):
    a_cls_dir = opj(single_dataset, classes[a_cls])

    for b_cls in range(a_cls + 1, num_class):
        b_cls_dir = opj(single_dataset, classes[b_cls])

        for a_file in range(cls_file_len):
            for b_file in range(cls_file_len):

                file_name = classes[a_cls] + str(a_file) + '_' + classes[b_cls] + str(b_file)

                write_row = [file_name, classes[a_cls], a_file, classes[b_cls], b_file]

                for cur_set in k_fold_sets:
                    r = open(opj(k_fold_sets_dir, cur_set, cur_set + '_train.txt'), 'r')
                    train_nums = list(map(int, r.readline().split(', ')))
                    r.close()

                    r = open(opj(k_fold_sets_dir, cur_set, cur_set + '_val.txt'), 'r')
                    val_nums = list(map(int, r.readline().split(', ')))
                    r.close()

                    r = open(opj(k_fold_sets_dir, cur_set, cur_set + '_test.txt'), 'r')
                    test_nums = list(map(int, r.readline().split(', ')))
                    r.close()

                    if a_file in train_nums and b_file in train_nums:
                        trs.append(file_name)
                        write_row.append(1)
                    elif a_file in val_nums and b_file in val_nums:
                        vas.append(file_name)
                        write_row.append(2)
                    elif a_file in test_nums and b_file in test_nums:
                        tes.append(file_name)
                        write_row.append(3)
                    else:
                        write_row.append(-1)
                
                """
                Each row contains
                - Information of what the mixture audio is consists of, including the file number.
                - Whether the mixture audio is used for train (1), validation (2), test (3), or none (-1)
                """
                wr.writerow(write_row)
                
                print(p_cnt, end='\r')
                p_cnt += 1
