import numpy as np
from itertools import chain
import os
import tensorflow as tf
import librosa
from audiomentations import Compose, AddBackgroundNoise, SpecFrequencyMask, TimeMask, Resample
from tqdm import tqdm



def load_files(data_dir='data'):
    """
    Loads the speech command data directory and returns the files and the labels
    :param data_dir:  path to directory containing audio files
    :return: filenames, commands
    """
    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    not_needed = ['README.md', 'speech_commands_v0.01.tar.gz', 'validation_list.txt', 'testing_list.txt', 'LICENSE',
                  '.DS_Store', '_background_noise_']
    commands = [j for j in commands if j not in not_needed]
    filenames = []
    for i in range(len(commands)):
        filenames.append(tf.io.gfile.glob(str(data_dir) + "/" + commands[i] + "/*.wav"))
    filenames = list(chain.from_iterable(filenames))
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)
    print('Number of total examples:', num_samples)
    # print('Number of examples per label:',
    #       len(tf.io.gfile.listdir(os.path.join(data_dir, commands[0]))))
    # print('Example file tensor:', filenames[0])
    return filenames, commands


def get_spectrogram(wav):
    """
    Takes a wav file as input and returns a melspectrogram
    :param wav: audio file in wav format
    :return: melspectrogram
    """
    D = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=480, hop_length=160, win_length=480, center=False)
    return D


def get_label(file_path):
    """
    Returns label of an audio file
    :param file_path:
    :return: Label
    """
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]


def get_mfccs(file_path):
    """
    Takes an audio file and returns mfcc of the audio file. This will be used to generate
    the mfcc of validation audio files and test audio files
    :param file_path: audio file in wav format
    :return: mfcc
    """
    signal, sr = librosa.load(file_path, sr=16000)
    edited_signal = librosa.util.fix_length(signal, 16000)
    log_spect = np.log(get_spectrogram(edited_signal) + 1e-6)

    mfccs = librosa.feature.mfcc(y=edited_signal, n_mfcc=40, sr=sr, S=log_spect)
    return mfccs


def get_mfccs_augmented(file_path, data_dir):
    """
    Takes an audio file and returns augmented mfcc of the audio file. This will be used to
    generate the mfcc of training set audio files
    :param file_path: audio file in wav format
    :param data_dir:
    :return:
    """
    signal, sr = librosa.load(file_path, sr=16000)
    edited_signal = librosa.util.fix_length(signal, 16000)
    start_ = int(np.random.uniform(-1600, 1600))
    if start_ >= 0:
        wav_time_shift = np.r_[edited_signal[start_:], np.random.uniform(-0.001, 0.001, start_)]
    else:
        wav_time_shift = np.r_[np.random.uniform(-0.001, 0.001, -start_), edited_signal[:start_]]

    augment = Compose([
        Resample(min_sample_rate=16000 * 0.85, max_sample_rate=16000 * 1.15),
        TimeMask(min_band_part=0.0, max_band_part=0.255),
        AddBackgroundNoise(sounds_path=os.path.join(data_dir, '_background_noise_'))
    ])
    edited_signal = augment(samples=wav_time_shift, sample_rate=16000)
    edited_signal = librosa.util.fix_length(edited_signal, 16000)
    spect = get_spectrogram(edited_signal)
    spect = SpecFrequencyMask(min_mask_fraction=0, max_mask_fraction=0.175)(spect)

    log_spec = np.log(spect + 1e-6)

    mfccs = librosa.feature.mfcc(y=edited_signal, n_mfcc=40, sr=sr, S=log_spec)
    return mfccs


def load_train_test_val_files(filenames, data_dir):
    """
    Takes a list of files as input and returns list of train files, val files and test files
    :param filenames: List of audio files
    :param data_dir: Path to audio files
    :return: train files, test files and val files
    """
    with open(os.path.join(data_dir, 'validation_list.txt'), 'r') as f:
        list_of_val_data = f.read().splitlines()

    with open(os.path.join(data_dir, 'testing_list.txt'), 'r') as f:
        list_of_test_data = f.read().splitlines()

    train_files = []
    val_files = []
    test_files = []
    for i in filenames:
        current_file = i.numpy().decode('utf-8')
        main_dir, label, data = current_file.split('/')
        input_data = '/'.join((label, data))

        if input_data in list_of_val_data:
            input_data = "/".join((data_dir, input_data))
            val_files.append(input_data)
        elif input_data in list_of_test_data:
            input_data = "/".join((data_dir, input_data))
            test_files.append(input_data)
        else:
            input_data = "/".join((data_dir, input_data))
            train_files.append(input_data)
    print("Completed loading train files, test files and validation files")
    return train_files, test_files, val_files


def create_example(fp, commands, data_dir, trainable=True):
    """

    :param fp: file path
    :param commands: list of labels
    :param data_dir: directory of audio files
    :param trainable: bool parameter for creating train set or test set and val set
    :return:mfcc, label
    """
    file_path = fp
    if trainable:
        mfccs = get_mfccs_augmented(file_path, data_dir)
    else:
        mfccs = get_mfccs(file_path)
    label = get_label(file_path)
    label = label.numpy().decode('utf-8')
    label_id = [i for i in range(len(commands)) if commands[i] == label][0]
    if mfccs.shape != (40, 98):
        print(mfccs.shape)
        # print("not padded")
    return np.expand_dims(mfccs, axis=-1), label_id



def load_data(train_files, test_files, val_files, commands, data_dir):
    """
    creates training, testing and validation data and labels
    :param train_files:
    :param test_files:
    :param val_files:
    :param commands:
    :param data_dir:
    :return: train_data, train_label, test_data, test_label, val_data, val_label
    """

    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []

    for fp in tqdm(train_files):

        data, label = create_example(fp, commands, data_dir,trainable=True)
        train_data.append(data)
        train_label.append(label)

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    print(f"Train data shape: {train_data.shape}, train label shape: {train_label.shape}")

    for fp in tqdm(val_files):

        data, label = create_example(fp, commands, data_dir,trainable=False)
        val_data.append(data)
        val_label.append(label)

    val_data = np.array(val_data)
    val_label = np.array(val_label)
    print(f"Validation data shape: {val_data.shape}, validation label shape: {val_label.shape}")

    for fp in tqdm(test_files):

        data, label =  create_example(fp, commands, data_dir,trainable=False)
        test_data.append(data)
        test_label.append(label)

    test_data = np.array(test_data)
    test_label = np.array(test_label)

    print(f"Test data shape: {test_data.shape}, test label shape: {test_label.shape}")
    return train_data, train_label, test_data, test_label, val_data, val_label


def collect_data(batch_size, data_dir):
    """
    Takes batch size and data directory as input and returns train_ds, test_ds and val_Ds
    :param batch_size: batch size of train set and val set
    :param data_dir:
    :return: train_ds, test_ds, val_ds, commands
    """
    filenames, commands = load_files()
    train_files, test_files, val_files = load_train_test_val_files(filenames, data_dir)
    train_data, train_label, test_data, test_label, val_data, val_label = load_data(train_files, test_files, val_files, commands, data_dir)
    train_ds = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    val_ds = tf.data.Dataset.from_tensor_slices((val_data, val_label))
    test_ds = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, test_ds, val_ds, commands