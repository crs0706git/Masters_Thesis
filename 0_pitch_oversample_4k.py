import scipy.io as wv
import scipy.io.wavfile as siw
from scipy.io import wavfile
from scipy import signal
import soundfile as sf

import librosa
import librosa.display

import numpy as np
from numpy import save
import os
import math
from crs_lib import *
import pdb

# Master directory
# master_dir = 'D:/D-Drive Files/000_data_backup'
master_dir = 'E:/chlee'
# master_dir = '../audio_datasets'
# master_dir = '../../../project/hw-team/chlee'

data_source = opj(master_dir, 'esc10_rename_vol_norm')
cls_lists = os.listdir(data_source)
cls_lists.sort()

# The setting of output downsampling rate
f_sample_out = 4000
sample_tag = '_' + str(f_sample_out//1000) + 'k'


"""
For this research, the shifting rate is set to small to make the least variation of frequency, avoiding distortion. To do this, the maximum octave is set to high number.

"do_rates" decides the final augmentation amount. For instance, Rate 5 makes 5 times more for the output file.
Ex: Rate 5 -> 320 * 5 = 1,600 files

With the max_octave_rate and each "do_rates", the final maximum octave value is determined.
-> Ex: Rate 5 -> max_octave_rate * do_rate = 100 * 5 = 500
500 data points for the octave frequency.

** Octave: Twice or half the frequency.
"""
max_octave_rate = 100

do_rates = [5, 20, 60, 100, 144]


for cur_shifting_rate in do_rates:
    output_dir = data_source + "_p" + str(cur_shifting_rate) + sample_tag
    refresh_dir(output_dir)
    
    # The maximum octave value is determined here
    max_octave = cur_shifting_rate * max_octave_rate

    print()
    print("Started for", output_dir)
    for cur_cls in cls_lists:
        access_cls = opj(data_source, cur_cls)
        cls_files = os.listdir(access_cls)
        cls_files.sort()

        output_cls = opj(output_dir, cur_cls)
        make_dir(output_cls)

        for file_num, cur_file in enumerate(cls_files):
            # Load original file
            cur_audio, sr = librosa.load(opj(access_cls, cur_file), sr=f_sample_out)
            
            # Ex: Rate 5 -> 0 ~ 4 pitch shifting
            for cur_pitch in range(cur_shifting_rate):
                """
                librosa.effects.pitch_shift
                - Shifts pitch higher or lower.
                - n_steps: Number of shifting pitch
                - bins_per_octave: The maximum value for making octave frequency, for the given audio. Greater or equal to such a number doubles or halves the pitch.
                """
                pitch_audio = librosa.effects.pitch_shift(cur_audio, f_sample_out, n_steps=-cur_pitch, bins_per_octave=max_octave)
                
                """
                Remain the original filename
                Else, add the name tag of how much pitch is shifted
                """
                if cur_pitch == 0:
                    name_tag = ""
                else:
                    name_tag = "_Down" + str(cur_pitch)

                output_path = opj(output_cls, cur_file.replace('.wav', '') + name_tag + '.wav')
                sf.write(output_path, pitch_audio, f_sample_out)

                print("%s, %d, %d" % (cur_cls, file_num + 1, cur_pitch), end='\t\t\r')

    print()
