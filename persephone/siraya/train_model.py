from persephone import corpus, corpus_reader, experiment
from persephone import siraya

# corpus_dir = "/pbs/home/m/magistry/SirayaTest"
corpus_dir = "/tmp/SirayaTest"
exp_dir = experiment.prep_exp_dir()



# Todo : normalized fbank ? (cf preprocess.py)
corpus = corpus.Corpus("fbank", "ortho", corpus_dir)
#corpus = corpus.Corpus("mfcc13_d", "ortho", corpus_dir)
reader = corpus_reader.CorpusReader(corpus, num_train=512, batch_size=32)
model = siraya.MyModel(exp_dir, reader, num_layers=4, hidden_size=256)
model.train(min_epochs=20, max_epochs=1000, early_stopping_steps=5, max_valid_ler=0.15, max_train_ler=0.2)
