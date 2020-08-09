from flask import Flask, jsonify

from pathlib import Path
import numpy as np


import sounddevice as sd
from scipy.io.wavfile import write


from persephone import siraya

#corpus = corpus.Corpus("fbank", "ortho", "/tmp/SirayaTest")
#reader = corpus_reader.CorpusReader(corpus, num_train=512, batch_size=16)

model_path = "/home/pierre/SRC/Siraya/Models/15/model/model_best.ckpt"


app = Flask(__name__)
def load_model():
    reader = siraya.FakeReader(Path("/tmp/SirayaTest"), 123, 19)
    model = siraya.SirayaModel("/tmp/exp", reader, num_layers=1, hidden_size=512)
    return model


@app.route("/", methods=["GET"])
def root():
    # record sound
    fs = 16000  # Sample rate
    seconds = 3  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('/tmp/sample.wav', fs, myrecording)  # Save as WAV file
    # model = siraya.MyModel("/tmp/exp", reader, num_layers=1, hidden_size=512)
    #model = siraya.SirayaModel("/tmp/exp", reader, num_layers=1, hidden_size=512)
    model = load_model()
    prediction, scores = model.transcribeOne("/tmp/sample.wav", restore_model_path=model_path)
    best = scores[np.argmax([-s for w,s in  scores])][0]
    return jsonify({"best":best, "prediction":prediction, "scores": sorted(scores, key= lambda x: x[1])})



