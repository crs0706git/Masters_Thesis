import scipy.io as wv
import scipy.io.wavfile as siw
from scipy.io import wavfile
import soundfile as sf

import librosa
import librosa.display
import matplotlib.pyplot as plt
# For plotting headlessly
# https://stackoverflow.com/questions/52432731/store-the-spectrogram-as-image-in-python/52683474
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
from numpy import save
import os
import math
from crs_lib import *
import pdb

from mfcc_function import mel_converter

# Master directory
# master_dir = 'D:/D-Drive Files/000_data_backup'
master_dir = 'E:/chlee'
# master_dir = '../audio_datasets'

data_source = opj(master_dir, 'esc10_rename_vol_norm_4k')

mixed_output_name = 'esc10_2_all_mixes_ver_vol_norm_4k'
mixed_output_location = opj(master_dir, mixed_output_name)
refresh_dir(mixed_output_location)

class_lists = os.listdir(data_source)
class_lists.sort()
num_class = len(class_lists)

"""
After Volume NormalizatioDetermine the maximum decibel value among the audio in ESC10.
abs_max determines the maximum decibel value, regardless of the "very-positive" or "very-negative" value of the signal - the signal fluctuates from top to bottom, going positive and negative.n, 1D matrix addition is performed between the audio to mix with.
"""
max_amp = 0
abs_max = lambda in_list: max(abs(min(in_list)), abs(max(in_list)))

for cur_cls in class_lists:
    access_cls = opj(data_source, cur_cls)
    cls_files = os.listdir(access_cls)
    cls_files.sort()

    for cur_file in cls_files:
        access_file = opj(access_cls, cur_file)
        
        _, y = siw.read(access_file)
        
        # Saves only the absolute maximum decibel value among the files.
        max_amp = max(max_amp, abs_max(y))

"""
Calculates the total number of files by mixing two audio of different classes.
"""
total_combinations = int(math.factorial(num_class) / (math.factorial(num_class-2) * math.factorial(2))) * 40 * 40
cnt_conv = 1

print("Started for", mixed_output_location)
# Select two audio files of two different classes
for a_cls in range(num_class - 1):
    a_dir = opj(data_source, class_lists[a_cls])

    for b_cls in range(a_cls + 1, num_class):
        b_dir = opj(data_source, class_lists[b_cls])

        for a_num in range(40):
            for b_num in range(40):
                a_filename = class_lists[a_cls] + str(a_num) + '.wav'
                a_file = opj(a_dir, a_filename)
                sr, a_data = wavfile.read(a_file)
                np_a_data = np.array(a_data, dtype=np.float64)

                b_filename = class_lists[b_cls] + str(b_num) + '.wav'
                b_file = opj(b_dir, b_filename)
                _, b_data = wavfile.read(b_file)
                np_b_data = np.array(b_data, dtype=np.float64)

                # max dB
                max_A = abs_max(a_data)
                max_B = abs_max(b_data)
                
                """
                Volume Normalization is done for each audio, dividing each audio with its highest decibel and then multiply with the highest decibel among the dataset.
                """
                np_mix = np_a_data / max_A * max_amp + np_b_data / max_B * max_amp
                np_mix = np.array(np_mix, dtype=np.int16)
                
                # The final filename contains which classes were mixed
                mix_filename = class_lists[a_cls] + str(a_num) + '_' + class_lists[b_cls] + str(b_num) + '.wav'
                mix_file = opj(mixed_output_location, mix_filename)

                wavfile.write(mix_file, sr, np_mix)

                print("%d/%d" % (cnt_conv, total_combinations), end='\t\t\r')
                cnt_conv += 1
print()
