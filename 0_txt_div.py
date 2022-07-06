import shutil
import os
import itertools

from crs_lib import *

import pdb

# Setting the train, validation, and test ratio for ESC10
train_ratio, vali_ratio, test_ratio = 8, 1, 1

vali_test_len = vali_ratio + test_ratio
vali_test_order = 10//vali_test_len

master_dir = 'E:/chlee'

input_dir = opj(master_dir, 'esc10_rename')

# The directory name for the dataset division
name_tag = 'esc10_' + str(train_ratio) + str(vali_ratio) + str(test_ratio) + 'div_txts'
total_set_output_dir = opj('.', name_tag)
refresh_dir(total_set_output_dir)

for set_num in range(vali_test_order):
    set_name = 'set' + str(set_num + 1)
    
    # Each division set is made separately
    out_set = opj(total_set_output_dir, set_name)
    refresh_dir(out_set)
    
    # Each text file is opened separately
    tr_txt = open(opj(out_set, set_name + '_train.txt'), 'w')
    va_txt = open(opj(out_set, set_name + '_val.txt'), 'w')
    te_txt = open(opj(out_set, set_name + '_test.txt'), 'w')

    cur_tr = []
    cur_va = []
    cur_te = []
    
    # Determine which distribution (train/validation/test) is associated with the specific number
    for cur_num in range(40):
        vali_test_start = set_num * vali_test_len
        vali_end = vali_test_start + vali_ratio
        test_end = vali_end + test_ratio

        cur_id = cur_num % 10

        if cur_id < vali_test_start or test_end <= cur_id < 10:
            cur_tr.append(cur_num)
        elif cur_id < vali_end:
            cur_va.append(cur_num)
        elif cur_id < test_end:
            cur_te.append(cur_num)
    
    """
    map converts each item into a specific data format
    ***.join returns with a specific character inserted between each item in the list
    ***.write records the variable inside the associated text file.
    """
    tr_txt.write(', '.join(map(str, cur_tr)))
    va_txt.write(', '.join(map(str, cur_va)))
    te_txt.write(', '.join(map(str, cur_te)))
    
    """
    Close each text file for the next operation.
    """
    tr_txt.close()
    va_txt.close()
    te_txt.close()
