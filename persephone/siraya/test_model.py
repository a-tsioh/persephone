from pathlib import Path
from persephone import siraya, corpus, corpus_reader



corpus_dir="/tmp/SirayaTest3"
corpus = corpus.Corpus("fbank", "ipa", corpus_dir)
#reader = corpus_reader.CorpusReader(corpus, num_train=1024, batch_size=32)

reader = siraya.FakeReader(Path("/tmp/SirayaTest3/corpus.p"), 123, 23)
model = siraya.SirayaModel("/tmp/exp", reader, num_layers=3, hidden_size=256)



model_path = "/home/pierre/SRC/Siraya/Models/22/model/model_best.ckpt"




dir = Path("/home/pierre/Corpora/Siraya/Recording-laptop")
for f in dir.iterdir():
    print(f.absolute())
    prediction, scores = model.transcribeOne(str(f.absolute()), restore_model_path=model_path)
    print(prediction)
    print(sorted(scores, key=lambda x: x[1]))
print(corpus.vocab_size, corpus.num_feats)