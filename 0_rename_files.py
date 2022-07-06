import shutil
import os

from crs_lib import *

master_dir = './dataset'

get_from = opj(master_dir, 'esc10')
target_dir = opj(master_dir, 'esc10_rename')
refresh_dir(target_dir)

for cur_cls in os.listdir(get_from):
    # Accessing each class folder in esc10 folder
    access_cls = opj(get_from, cur_cls)
    
    # Make each class folder for the new directory
    access_folder = opj(target_dir, cur_cls)
    refresh_dir(access_folder)
    
    """
    Copy the file from esc10 to the new folder.
    When copying, each file are renamed with each corresponding class and the number
    Ex: 1st chainsaw audio file -> chainsaw0.wav
    """
    for cur_num, cur_file in enumerate(os.listdir(access_cls)):
        cpy_file = opj(access_cls, cur_file)
        cpy_to = opj(access_folder, cur_cls + str(cur_num) + '.wav')

        shutil.copyfile(cpy_file, cpy_to)
