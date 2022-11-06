from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.utterance import Utterance
from pathlib import Path
import os
import pickle
# Contains the set of utterances of a single speaker

root_pickle = './data'
class Speaker:
    def __init__(self, root: Path, speaker):
        self.root = root
        self.name = speaker
        self.utterances = None
        self.utterance_cycler = None
        
    def _load_utterances(self):
        # speaker 불러오기
        speaker = ""
        rt = f'{self.name}_sources.txt'
        with open(f'{os.path.join(root_pickle,rt)}', 'rb') as f:
            sources = f
            self.utterances = [Utterance(*list(map(lambda x: self.root.joinpath(x), line.decode().split(',')))) for line in sources]
        self.utterance_cycler = RandomCycler(self.utterances)
               
    def random_partial(self, count, n_frames):
        """
        Samples a batch of <count> unique partial utterances from the disk in a way that all 
        utterances come up at least once every two cycles and in a random order every time.
        
        :param count: The number of partial utterances to sample from the set of utterances from 
        that speaker. Utterances are guaranteed not to be repeated if <count> is not larger than 
        the number of utterances available.
        :param n_frames: The number of frames in the partial utterance.
        :return: A list of tuples (utterance, frames, range) where utterance is an Utterance, 
        frames are the frames of the partial utterances and range is the range of the partial 
        utterance with regard to the complete utterance.
        """
        if self.utterances is None:
            self._load_utterances()

        utterances = self.utterance_cycler.sample(count)

        a = [(u,) + u.random_partial(n_frames) for u in utterances]

        return a
