import matplotlib.pyplot as plt
import librosa
import librosa.display
from librosa import effects
import numpy as np
from sklearn.preprocessing import normalize

import python_speech_features

from scipy.io import wavfile

wavdir = "/tmp/SirayaTest/feat/"
files = [wavdir + "S-438.wav"]
files = files + ["/tmp/sample.wav", "/tmp/proc.wav"]
for file in files:
    signal, sr = librosa.load(file, sr=None)
    print(len(signal),sr)
    signal = (librosa.resample(signal, sr, 16000))
    sr = 16000
    print(len(signal), sr)
    signal, _ = effects.trim(signal[int(0.1*sr):], top_db=5, frame_length=128, hop_length=64)


    librosa.display.waveplot(signal, sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(file)
    plt.show()

    fft = np.fft.fft(signal)

    magnitude = np.abs(fft)
    magnitude = magnitude[:int(len(magnitude)/2)]
    frequency = np.linspace(0, sr, len(magnitude))

    plt.plot(frequency, magnitude)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title(file)

    plt.show()

    n_fft = 512
    hop_length = 128

    stft = librosa.core.stft(signal, n_fft, hop_length)
    spectrogram = np.abs(stft)

    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    librosa.display.specshow(log_spectrogram)
    plt.colorbar()
    plt.xlabel("Time")
    plt.title(file)
    plt.show()

    # MFFCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=40)
    # librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
    # plt.title("mfcc " + file)
    # plt.show()

    data = python_speech_features.logfbank(signal, sr, nfilt=40, lowfreq=50, highfreq=8000)
    print(np.shape(data))
    normalized = normalize(data, 'l2', 1)
    librosa.display.specshow(data, sr=sr)
    plt.title("euh " + file)
    plt.show()

