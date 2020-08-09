import os
from pathlib import Path
from typing import Union, Optional

import logging

import tensorflow as tf
from persephone.config import ENCODING

import numpy as np

import librosa
from librosa import effects
import python_speech_features

import soundfile as sf


from persephone.exceptions import PersephoneException
from persephone.model import allow_growth_config
from persephone import rnn_ctc


logger = logging.getLogger(__name__)  # type: ignore


TARGET_LEXICON = list(map(lambda x: x.split(" "), [
"a g u a n g",
"a y a m",
"d a r a n g",
"f n a n g",
"m u s u h a p a",
"p a r a n a x",
"p u r a r a y",
"t m i k o g",
"v a w u n g",
"v u k i n",
"v u r i g a n",
"w a g i"
]))

class SirayaModel(rnn_ctc.Model):

    def __init__(self, exp_dir: Union[str, Path], corpus_reader, num_layers: int = 3,
                 hidden_size: int=250, beam_width: int = 100,
                 decoding_merge_repeated: bool = True) -> None:
        super().__init__(exp_dir, corpus_reader, num_layers, hidden_size, beam_width, decoding_merge_repeated)

    def fbank_of_file(self, wav_path, flat=True):
        """ Currently grabs log Mel filterbank, deltas and double deltas."""
        (sig, rate) = librosa.load(wav_path, sr=None)
        sig = librosa.to_mono(librosa.resample(sig, rate, 16000))
        rate = 16000
        sig, _ = effects.trim(sig[int(0.1 * rate):], top_db=20, frame_length=128, hop_length=64)
        sf.write("/tmp/proc.wav", sig, rate)
        if len(sig) == 0:
            logger.warning("Empty wav: {}".format(wav_path))
        fbank_feat = python_speech_features.logfbank(sig, rate, nfilt=40, lowfreq=250, highfreq=6000)
        mfcc = python_speech_features.mfcc(sig, rate, appendEnergy=True)
        energy_row_vec = mfcc[:, 0]
        energy = energy_row_vec[:, np.newaxis]
        feat = np.hstack([energy, fbank_feat])
        delta_feat = python_speech_features.delta(feat, 2)
        delta_delta_feat = python_speech_features.delta(delta_feat, 2)
        all_feats = [feat, delta_feat, delta_delta_feat]
        if not flat:
            all_feats = np.array(all_feats)
            # Make time the first dimension for easy length normalization padding
            # later.
            all_feats = np.swapaxes(all_feats, 0, 1)
            all_feats = np.swapaxes(all_feats, 1, 2)
        else:
            all_feats = np.concatenate(all_feats, axis=1)

        # Log Mel Filterbank, with delta, and double delta
        feat_fn = wav_path[:-3] + "fbank.npy"
        return all_feats

    def transcribeOne(self, wavfile: str, restore_model_path: Optional[str]=None):
        """ Transcribes an untranscribed dataset. Similar to eval() except
        no reference translation is assumed, thus no LER is calculated.
        """
        feats = self.fbank_of_file(wavfile)
        saver = tf.train.Saver()
        with tf.Session(config=allow_growth_config) as sess:
            if restore_model_path:
                saver.restore(sess, restore_model_path)
            else:
                if self.saved_model_path:
                    saver.restore(sess, self.saved_model_path)
                else:
                    raise PersephoneException("No model to use for transcription.")

            #batch_gen = self.corpus_reader.untranscribed_batch_gen()
            #assert(len(batch_gen) == 1)

            hyp_batches = []
            batch_x = [feats]
            batch_x_lens = [len(feats)]
            feat_fn_batch = "coucou"
            #for batch_i, batch in enumerate(batch_gen):
            #print("b", batch_i, batch)
            #batch_x, batch_x_lens, feat_fn_batch = batch
            feed_dict = {self.batch_x: [batch_x[0]],
                         self.batch_x_lens: [batch_x_lens[0]]}

            [dense_decoded] = sess.run([self.dense_decoded], feed_dict=feed_dict)
            hyps = self.corpus_reader.human_readable(dense_decoded)

            # Prepare dir for transcription
            hyps_dir = os.path.join(self.exp_dir, "transcriptions")
            if not os.path.isdir(hyps_dir):
                os.mkdir(hyps_dir)

            hyp_batches.append((hyps,feat_fn_batch))
            from persephone import distance
            prediction = hyp_batches[0][0][0]
            return prediction, list(zip(["".join(x) for x in TARGET_LEXICON], [distance.min_edit_distance(prediction,x) for x in TARGET_LEXICON]))
