import numpy as np
from pathlib import Path

import librosa
import soundfile as sf


corpus_dir = Path("/tmp/SirayaTest3")
wav_dir = Path(corpus_dir, "wav")
label_dir = Path(corpus_dir, "label")
new_sample_per_length= 200
lengths = [2,3,4]
common_rate = None

def buildIndex():
    return [f.name[:-4] for f in wav_dir.iterdir()]

def read_label(sample_name):
    with open(Path(label_dir, sample_name + ".ipa")) as f:
        return f.readline().strip()

def read_wav(sample_name):
    global common_rate
    (sig, rate) = librosa.load(str(Path(wav_dir, sample_name + ".wav")), sr=None)
    if common_rate is None:
        common_rate = rate
    assert(rate == common_rate)
    return sig

def write_label(new_name, ipa):
    with open(Path(label_dir, new_name + ".ipa"), "w") as f:
        f.write(ipa + "\n")


idx = buildIndex()
N = len(idx)
k = 2

counter = 0
for l in lengths:
    for s in range(new_sample_per_length):
        counter += 1
        base_name = "N-" + str(counter)
        samples = [idx[int(i*N)] for i in np.random.sample(l)]
        ipa = " ".join([read_label(sample) for sample in samples])
        print(ipa)
        signals = [read_wav(sample) for sample in samples]
        sig = np.concatenate(signals)
        sf.write(str(Path(wav_dir, base_name + ".wav")), sig, common_rate)
        write_label(base_name, ipa)
