from dataclasses import asdict
from pathlib import Path
import os
from glob import glob
from tqdm import tqdm
import pickle
root = '../data/train/'
class make_source:
    def __init__(self, root = root):
        self.root = root
        #self.name = root.name
    def track_wav_path(self,wf_):
        wf_path = os.path.join(self.root,wf_)
        wfs = []
        for env in os.listdir(wf_path):
            now_env = os.path.join(self.root,env)
            for days in os.listdir(now_env):
                now_days = os.path.join(days)
                for wf in os.listdir(now_days):
                    wfs.append(wf)
        return wfs
    def pair(self,wfs):
        wav_ids = [wav[i].split('.')[0]  for i in range(wav)]
        frame_ids = [frame[i].split('.')[0] for i in range(frame)]
        frame_ids
        zip_file = []
        for name in wav_ids:
            if name in frame_ids:
                zip_file.append((f'{name}.wav',f'{frame_ids[frame_ids.index(name)]}.npy'))
        return zip_file

if __name__ == "__main__":
    ms = make_source()
    wav,frame = ms.track_wav_path(os.path.join(root,'wav')),ms.track_wav_path(os.path.join(root,'mel'))
    zip_file = ms.pair(wav,frame)
    wf_dict = dict()
    for i ,j in zip_file:
        wf_dict[i] = j
    with open('_source.pickle', 'wb') as f:
        pickle.dump(wf_dict, f, pickle.HIGHEST_PROTOCOL)
