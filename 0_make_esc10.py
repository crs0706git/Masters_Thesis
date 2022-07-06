import shutil
import os
import csv

from crs_lib import *

# Directory of where all the datasets are saved
master_dir = './dataset'

"""
opj: os.path.join - returns full directory by combining each element inside.
esc50: ESC50 dataset directory
esc50_csv: ESC50's csv file directory (esc50.csv)
esc50_audio: ESC50's audio directory

Inside ESC50 audio directory, the files are named with specific digits and characters, which each indicate class and class number, a record took number, etc.
The esc50.csv file contains the information of each audio file.
"""
esc50 = opj(master_dir, 'ESC-50')
esc50_csv = opj(esc50, 'meta', 'esc50.csv')
esc50_audio = opj(esc50, 'audio')

"""
esc10: New directory to save ESC10 dataset.
refresh_dir: A function from crs_lib.py. If the directory exists, the directory is erased and made again. Else, the directory is made.
"""
esc10 = opj(master_dir, 'esc10')
refresh_dir(esc10)

"""
esc50.csv is used to check the classes and to made ESC10 dataset.
"""
cls_called = []
with open(esc50_csv, 'r', newline='') as rc:
    reading_line = csv.DictReader(rc)
    
    for row in reading_line:
        # If 'esc10' is recorded in True, it means that the corresponding class was used in ESC10.
        if row['esc10'] == 'True':
            # Making new directory in ESC10 dataset directory
            save_at = opj(esc10, row['category'])
            
            # To make sure that the directory is made or not. It mades the class directory if not exists.
            if row['category'] not in cls_called:
                cls_called.append(row['category'])
                refresh_dir(save_at)
            
            # shutil.copyfile is a function that literally copies a source file to a destination
            cpy_from = opj(esc50_audio, row['filename'])
            cpy_to = opj(save_at, row['filename'])
            shutil.copyfile(cpy_from, cpy_to)
