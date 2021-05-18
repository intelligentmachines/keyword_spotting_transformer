import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from argparse import ArgumentParser
import librosa
import librosa.display



commands =['off', 'bed', 'dog', 'one', 'zero', 'happy', 'visual', 'cat', 'six', 'house', 'left',
            'yes', 'backward', 'marvin', 'no', 'bird', 'go', 'up', 'learn', 'forward', 'two', 'wow', 'nine', 'on', 'right', 'seven',
            'tree', 'sheila', 'five', 'stop', 'eight', 'down', 'four', 'follow', 'three']


def predict(path, model_path):
    mfcc = get_mfccs(path)
    mfcc = tf.expand_dims(mfcc, axis=0)
    mfcc = tf.expand_dims(mfcc, axis=-1)

    model = tf.keras.models.load_model(model_path, custom_objects={"optimizer": tfa.optimizers.AdamW(
        learning_rate=0.001, weight_decay=0.0001
    )})

    y_pred = np.argmax(model.predict(mfcc))
    output = commands[y_pred]
    print(f"The keyword spoken is: {output}")


def get_spectrogram(wav):
    D = librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=480, hop_length=160, win_length=480, center=False)
    return D


def get_mfccs(file_path):
    signal, sr = librosa.load(file_path, sr=16000)
    edited_signal = librosa.util.fix_length(signal, 16000)
    log_spect = np.log(get_spectrogram(edited_signal) + 1e-6)

    mfccs = librosa.feature.mfcc(y=edited_signal, n_mfcc=40, sr=sr, S=log_spect)
    return mfccs


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", default="KWS_transformer/1")
    parser.add_argument("--file_path", default='data/cat/0a2b400e_nohash_0.wav')
    args = parser.parse_args()
    predict(args.file_path, args.model_dir)
