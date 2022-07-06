import torch
import numpy as np
import librosa
import scipy.io.wavfile as wavfile

import pdb


def audio_STFT_torch(input_audio, nfft):
    converted = torch.from_numpy(np.array(input_audio, dtype=np.float32))
    converted = torch.stft(converted, n_fft=nfft)
    a, b, c = converted.shape
    converted = torch.reshape(converted, (c, a, b))
    return converted


def STFT_audio_torch(input_stft, nfft, in_len):
    c, a, b = input_stft.shape
    converted = torch.reshape(input_stft, (a, b, c))
    converted = torch.istft(converted, n_fft=nfft, length=in_len)
    converted = converted.numpy()
    converted = np.array(converted, dtype=np.float32)
    return converted


def audio_STFT_lib(input_audio, nfft):
    converted = np.abs(librosa.stft(input_audio, n_fft=nfft))
    converted = np.array(converted, dtype=np.float32)
    converted = torch.from_numpy(converted)
    converted = torch.unsqueeze(converted, 0)
    return converted
    
    
def audio_STFT_lib_upNorm(input_audio, nfft, up_rate=10):
    converted = input_audio * up_rate
    converted = np.abs(librosa.stft(converted, n_fft=nfft))
    converted = np.array(converted, dtype=np.float32)
    converted = torch.from_numpy(converted)
    converted = torch.unsqueeze(converted, 0)
    return converted
    
    
def audio_mel_lib(input_audio, in_sr, nfft):
    converted = librosa.feature.melspectrogram(input_audio, sr=in_sr, n_fft=nfft, fmax=5000)
    converted = np.array(converted, dtype=np.float32)
    converted = torch.from_numpy(converted)
    converted = torch.unsqueeze(converted, 0)
    return converted


def STFT_audio_lib(in_stft, nfft, in_len):
    in_stft = in_stft.detach().cpu().numpy()
    converted = librosa.istft(in_stft, n_fft=nfft, length=in_len)
    return converted


def audio_scipy(in_dir, in_sr):
    sr, y = wavfile.read(in_dir)
    
    if in_sr != sr:
        y = np.array(y, dtype=np.float32)
        y = librosa.resample(y, orig_sr=sr, target_sr=in_sr)
        y = np.array(y, dtype=np.int16)
        
    return y
