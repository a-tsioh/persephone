from flask import Flask, jsonify, send_from_directory, request

import os

from pathlib import Path

from pathlib import Path
import numpy as np

import json

import sounddevice as sd
from scipy.io.wavfile import write


from persephone import siraya

#corpus = corpus.Corpus("fbank", "ortho", "/tmp/SirayaTest")
#reader = corpus_reader.CorpusReader(corpus, num_train=512, batch_size=16)

model_path = Path(os.environ["MODEL"]) #  "/home/pierre/SRC/Siraya/Models/21/model/model_best.ckpt"
corpus_file = Path(model_path, "corpus.p")
model_file = str(Path(model_path, "model/model_best.ckpt"))
model_description = json.load(open(Path(model_path, "model_description.json")))


app = Flask(__name__)
def load_model():
    reader = siraya.FakeReader(corpus_file, 123, 23)
    model = siraya.SirayaModel("/tmp/exp", reader, num_layers=model_description["num_layers"], hidden_size=model_description["hidden_size"])
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
    prediction, scores = model.transcribeOne("/tmp/sample.wav", restore_model_path=model_file)
    best = scores[np.argmax([-s for w,s in  scores])][0]
    return jsonify({"best":best, "prediction":prediction, "scores": sorted(scores, key= lambda x: x[1])})


@app.route('/static/<path>', methods=["GET"])
def statics(path):
    print(path)
    return send_from_directory('/home/pierre/SRC/persephone/statics/', path)

@app.route('/upload', methods=["POST"])
def upload_wav():
    file = request.files['audio_data']
    file.save("/tmp/sample_up.ogg")
    model = load_model()
    prediction, scores = model.transcribeOne("/tmp/sample_up.ogg", restore_model_path=model_file)
    #print(request.form['audio_data'])
    best = scores[np.argmax([-s for w, s in scores])][0]
    return jsonify({"best": best, "prediction": prediction, "scores": sorted(scores, key=lambda x: (x[1],-len(x[0])))})

