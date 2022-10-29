import pickle

from encoder.data_objects.random_cycler import RandomCycler
from encoder.data_objects.speaker_batch import SpeakerBatch
from encoder.data_objects.speaker import Speaker
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# TODO: improve with a pool of speakers for data efficiency

class SpeakerVerificationDataset(Dataset):
    def __init__(self, datasets_root, speaker_list_path):
        self.root = datasets_root
        with open(f'{speaker_list_path}/train_speakers.pkl', 'rb') as f:
            speaker_lst = pickle.load(f) # 각 speaker 폴더 불러옴
        if len(speaker_lst) == 0:
            raise Exception("No speakers found. Make sure you are pointing to the directory "
                            "containing all preprocessed speaker directories.")
        self.speakers = [Speaker(self.root, speaker) for speaker in speaker_lst] # 각 speaker 폴더 경로를 Speaker에 입력
        self.speaker_cycler = RandomCycler(self.speakers)

    def __len__(self):
        return int(1e10)
        
    def __getitem__(self, index):
        return next(self.speaker_cycler)
    
    def get_logs(self):
        log_string = ""
        for log_fpath in self.root.glob("*.txt"):
            with log_fpath.open("r") as log_file:
                log_string += "".join(log_file.readlines())
        return log_string
    
    
class SpeakerVerificationDataLoader(DataLoader):
    def __init__(self, cfg, dataset, speakers_per_batch, utterances_per_speaker, sampler=None, 
                 batch_sampler=None, num_workers=0, pin_memory=False, timeout=0, 
                 worker_init_fn=None):
        
        self.cfg = cfg
        
        self.utterances_per_speaker = utterances_per_speaker

        super().__init__(
            dataset=dataset, 
            batch_size=speakers_per_batch, 
            shuffle=False, 
            sampler=sampler, 
            batch_sampler=batch_sampler, 
            num_workers=num_workers,
            collate_fn=self.collate, 
            pin_memory=pin_memory, 
            drop_last=False, 
            timeout=timeout, 
            worker_init_fn=worker_init_fn
        )

    def collate(self, speakers):
        return SpeakerBatch(speakers, self.utterances_per_speaker, self.cfg.data.audio.partials_n_frames) 
    