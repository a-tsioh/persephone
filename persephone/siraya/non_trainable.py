from typing import Sequence, List

from pathlib import Path
import pickle

from persephone import corpus


class FakeCorpus(corpus.Corpus):

    def __init__(self, num_feats, vocab_size):
        self._nf = num_feats
        self.vocab_size = vocab_size

    @property
    def num_feats(self):
        return self._nf

    @classmethod
    def from_pickle(cls, corpus_file: Path):
        pickle_path = corpus_file
        # logger.debug("Creating Corpus object from pickle file path %s", pickle_path)
        with pickle_path.open("rb") as f:
            c = pickle.load(f)
            c.__class__ = cls
            return c



class FakeReader:

    def __init__(self, pickled, num_feats, vocab_size):
        self.corpus = FakeCorpus.from_pickle(pickled)
        self.corpus._nf = num_feats
        self.corpus.vocab_size = vocab_size

    def human_readable(self, dense_repr: Sequence[Sequence[int]]) -> List[List[str]]:
        """ Returns a human readable version of a dense representation of
        either or reference to facilitate simple manual inspection.
        """

        transcripts = []
        for dense_r in dense_repr:
            non_empty_phonemes = [phn_i for phn_i in dense_r if phn_i != 0]
            transcript = self.corpus.indices_to_labels(non_empty_phonemes)
            transcripts.append(transcript)

        return transcripts
