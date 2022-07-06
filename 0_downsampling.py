import scipy.io as wv
import scipy.io.wavfile as siw
from scipy.io import wavfile
import soundfile as sf

import librosa
import librosa.display

import os
from crs_lib import *
import pdb

# Master directory
# master_dir = 'D:/D-Drive Files/000_data_backup'
# master_dir = 'E:/chlee'
# master_dir = '../audio_datasets'
master_dir = '../../../project/hw-team/chlee'

# Source directory
data_source = opj(master_dir, 'esc10_3_all_mixes_vol_norm_set2_Only')
cls_lists = os.listdir(data_source)
cls_lists.sort()

# Setting into the specific sample rate
cur_set = 8000

# Final output directory, putting the output sample rate on the end of the source directory name.
output_dir = data_source + '_' + str(cur_set//1000) + 'k'
refresh_dir(output_dir)

cnt = 1
print()
print("Started for", output_dir)
for cur_cls in cls_lists:
    # Accessing each class and the copy of subdirectories
    access_cls = opj(data_source, cur_cls)
    cls_files = os.listdir(access_cls)
    cls_files.sort()

    output_cls = opj(output_dir, cur_cls)
    refresh_dir(output_cls)

    for cur_file in cls_files:
        # Access each audio file in each subdirectory,
        access_file = opj(access_cls, cur_file)

        save_dir = opj(output_cls, cur_file)
        
        """
        Loading each audio file into specific sample rate.
        "sr" parameter sets the sample rate when loading
        The outputs of "librosa.load": audio, sample rate
        """
        y, _ = librosa.load(access_file, sr=cur_set)
        
        """
        Saving the audio into the new directory
        
        Directory name, audio (in list), sample rate
        """
        sf.write(save_dir, y, cur_set)
        
        #
        """
        Indicates how many files have been converted
        '\t' indicates tab
        '\r' indicates remove
        When printing, the previous message is removed and replace the new message
        """
        print(cnt, end='\t\t\r')
        cnt += 1
print()
