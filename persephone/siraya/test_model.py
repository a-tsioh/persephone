from pathlib import Path
from persephone import siraya


reader = siraya.FakeReader(Path("/tmp/SirayaTest"), 123, 19)
print(reader.corpus)

model = siraya.SirayaModel("/tmp/exp", reader, num_layers=1, hidden_size=512)

model_path = "/home/pierre/SRC/Siraya/Models/17/model/model_best.ckpt"


dir = Path("/home/pierre/Corpora/Siraya/Recording-laptop")
for f in dir.iterdir():
    print(f.absolute())
    prediction, scores = model.transcribeOne(str(f.absolute()), restore_model_path=model_path)
    print(prediction)
    print(sorted(scores, key=lambda x: x[1]))
