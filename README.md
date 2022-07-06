The overall flow is described as follows:

![ov_structure](https://user-images.githubusercontent.com/67090206/175464226-b1ccce96-fb58-445d-93c2-2238f0596480.png)



# Requirements
- Python 3.7.11
- torch 1.11 (CUDA 11.3)
  - Recommend the version in which "same" padding option is available for Conv2d
- librosa 0.9.1
- numpy 1.21.5
- scipy 1.7.3
- SoundFile 0.10.3
- Dataset: ESC50
  - https://github.com/karolpiczak/ESC-50

# 0. Preprocessings
Code:
- 0_make_esc10.py
- 0_rename_files.py
- 0_txt_div.py
- 0_downsampling.py
- 0_pitch_oversample_4k.py
- 0_2mix_norm_vol.py
- 0_2mix_csv.py

## 0_make_esc10.py
Before ESC50, ESC10 was made first. ESC10 consists of 10 classes and 40 audio files for each class. Each audio file is recorded at a 44.1kHz sample rate. <br/>
Inside ESC50 folder, downloaded via the GitHub link provided, there is an audio folder and CSV file. The audio files inside the audio folder are named in specific digits and characters, indicating class, class number, a record taken number, etc. <br/>
The CSV file (esc50.csv) has information like the corresponding class (category) for each audio file. Also, the CSV file contains information on whether the file was in ESC10 or not. With the CSV file, the following code makes the ESC10 dataset by checking if each file and class was in ESC10.

## 0_rename_files.py
With ESC10 made with 0_make_esc10.py, the new directory is made. The subdirectories named in class types are made in the same way, but the audio files in the subdirectories are renamed with class names and the specific number.
Ex: 1st chainsaw audio file -> chainsaw0.wav

## 0_txt_div.py
The following code operates train, validation, and test dataset division, by designating each distribution with specific digits. The division is operated for each class. The division text files are saved to "esc10_811div_txts", which also exists in this repository. The purpose of the specific division is to equal the distribution of each class file of ESC10 since the dataset is small. Any equally distribution method will work. <br/>
Refer to the image below for the specific flow:

![div](https://user-images.githubusercontent.com/67090206/175455235-6e1ba2ff-70d7-416b-a828-9e684f20492d.png)

## 0_downsampling.py
The purposes of the audio downsampling are:
- Reduce the number of input layers of the further model.
- Reduce the dimension of STFT format

To downsample, "librosa.load" function was used. "librosa.load" can load an audio file with specific sample rate, automatically. If the sample rate is not declared, then the audio loads into 22050Hz. <br/>
The alternate methods for downsampling are:
- librosa.resample
- scipy.signal.resample

## 0_pitch_oversample_4k.py
The following code augments only on ESC10 dataset. This is done to make a balance between the 1-class label (single-label) dataset and the 2-class label (multi-label, made by 0_2mix_norm_vol.py). Specifically for this research, the ratio between the 1-class label training dataset and the 2-class label training dataset is 1:144 (320 files: 46080 files). To overcome such a data imbalance situation, the dataset oversampling method was used on the 1-class label dataset. Among the several methods of oversampling, an augmentation method was used. <br/>
Among several augmentation methods for audio files, a pitch shifting augmentation method was used for this code. The pitch shifting augmentation method augments the audio into several shifted pitch, either higher or lower. Refer to the link and the images below for the specific process of this code: <br/>
Link: https://medium.com/@makcedward/data-augmentation-for-audio-76912b01fdf6 <br/>

![pitch](https://user-images.githubusercontent.com/67090206/175459109-2165de11-61cd-4a30-ad72-7662548cb6c5.png)

![aug](https://user-images.githubusercontent.com/67090206/175458512-a0c6fb31-8b71-4a8e-a209-cc62ec94bdf6.png)


## 0_2mix_norm_vol.py
The following code prepares the 2-class label dataset. Since ESC10 dataset only contains audio that each is recorded single-source sound. Two main processes are done for mixing:
1. Volume Normalization (entire dataset) <br/>
Before mixing, the volume of each audio are considered. When mixing, certain mixture audio had in case of one audio being louder than the other audio and so causing the other audio to be harder to distinguish. Such a situation defects the model detection performance, so it is considered an impossible case to deal with. The situation is neglected for this research and applied the alternate method, called Volume Normalization, to resolve the situation.<br/>
For this research, the volume normalization was done to the entire dataset. Among the audio files, the code finds the loudest audio. Then, the code normalize each audio to the loudest volume value.

2. 1D Matrix Addition
Calculates the total number of files by mixing two audio of different classes. After Volume Normalization, 1D matrix addition is performed between the audio to mix with. The audio files mixed are associated with different classes to each other file.

![volnormpng](https://user-images.githubusercontent.com/67090206/175461638-ef6e6e1b-248d-4cad-98c8-cb55419b5061.png)

Note: The code also includes audio downsampling.

## 0_2mix_csv.py
To precisely and easily browse the information of the mixture information, the CSV file was used. Using the same method of mixing (as 0_2mix_norm_vol.py), the following code writes on a CSV file about what each mixture audio is consists of and whether it is associated to train, validation, or test dataset based on the dataset division on the single-label dataset. For example, if the mixture audio is mixed with only the training files, then it is considered a training file. Else if the mixture audio is mixed with the different sets, like training and testing files, then it is not used. The corresponding division is based on the set number and the distributed set (train/validation/test).

# 1. Audio Source Counter (ASC)
Code:
- asc_model.py
- UNet_utils.py
- audio_source_counter.py

*Note: The "UNet_utils.py" and "audio_source_counter.py" includes STFT conversion process. The conversion function is defined in "UNet_utils.py". <br/>

## 1.1 asc_model.py
The model of ASC is referred from Attention RNN: https://github.com/douglas125/SpeechCmdRecognition <br/>
The Attention RNN model is a CRNN structure that uses CNN layers at first for extracting local relations features and bidirectional long short-term memory (BLSTM) layers, a type of RNN layers, next for extracting long-term dependencies features. <br/>
- CNN: Extract unique features of audio for each kernel, but remains the output feature map size with the STFT converted dimension size.
- BLSTM (RNN): The sound activation event(s) is happened in chronological order. By this factor, the RNN-type layer is used for treating a seqeuntial input.

The overall model is described as follows:

![asc_model_structure](https://user-images.githubusercontent.com/67090206/175464994-c75b3c2a-f5bb-4b61-9ab0-bada0c999147.png)


## 1.2 UNet_utils.py
The following code contains functions of STFT conversion and inverse STFT conversion. For this research, STFT conversion with librosa library is used, in the end, which the function name is "audio_STFT_lib". <br/>
The specific usage for STFT conversion needs to be studied about basic STFT conversion and external Python Audio library usage, to understand the expected output properly: <br/>
Librosa Link: https://librosa.org/doc/main/generated/librosa.stft.html

```
def audio_STFT_lib(input_audio, nfft):
    """
    "librosa.stft" converts audio into STFT format.
    n_fft: Determines the width of the output format.
    """
    converted = np.abs(librosa.stft(input_audio, n_fft=nfft))
    
    # The matrix is refined into a specific number type, to be used in the model computation.
    converted = np.array(converted, dtype=np.float32)
    
    # Converts the data type to torch.
    converted = torch.from_numpy(converted)
    
    # The model perceives the input in several channels. Each channel of the audio represents spatial information (like Stereo). For this research, the audio is considered mono, so the channel is one.
    converted = torch.unsqueeze(converted, 0)
    
    return converted
```

## 1.3 audio_source_counter.py
The following code trains and tests the ASC model. <br/>
Process:
1. Determine the directories of audio files to use as an input and the corresponding number of labels as an output.
- Single-label: <br/>
  Input: Single-label audio directory <br/>
  Output: 0 (indicates 1 label exists in the input)
- 2-class label: <br/>
  Input: 2-class label audio directory <br/>
  Output: 1 (indicates 2 label exists in the input)

2. Dataset Oversampling and Undersampling is applied
- Oversampling: <br/>
  Select the specific augmented single-label dataset
- Undersampling: <br/>
  As much as oversampled single-label dataset, the code randomly selects multi-label audio files. The following method is called Random Undersampling, refer to the link below: <br/>
  Link: https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis

3. Load the selected files, done by "uploader" function

4. Training and Testing are done for each dataset.
- Can select the option of testing unselected multi-label audio files, to verify whether the model can deal with the untrained files.

# 2. Classifier
Code:
- CNN_ISOCC_copy.py
- sep_opt_isocc_single_volnorm_STFT.py / sep_opt_isocc_multi_random_volnorm_STFT.py

## CNN_ISOCC_copy.py
The following code is the model that is exactly the same with the previous audio multi-label research, proposed on ISOCC 2021: <br/>
Link: https://ieeexplore.ieee.org/document/9613899 <br/>

The overall structure is described as follows:

![isocc_model](https://user-images.githubusercontent.com/67090206/175468383-c4149cbe-40ea-4d04-9215-c815f2917a59.png)

The model is in simple CNN structure. For classifying audio, CNN structure distinguishes the label(s) associated to the input audio well enough.

## sep_opt_isocc_single_volnorm_STFT.py / sep_opt_isocc_multi_random_volnorm_STFT.py
The following code train and test with the model for classifying label(s) of the input audio. <br/>
- sep_opt_isocc_single_volnorm_STFT.py <br/>
  The following code trains and tests the 1-class label classifier.
- sep_opt_isocc_multi_random_volnorm_STFT.py <br/>
  The following code trains and tests the 2-class label classifier.

# 9. Utilities
Code:
- crs_lib.py
- tmr.py

## crs_lib.py
The following code contains functions for making a directory and path-joining function

## tmr.py
The following code contains functions for recording operation time and starting operation time. Simply, just use "datatime" library which is a basic Python library given when installed.
