class Mel_Dataset(torch.utils.data.Dataset):
    def __init__(self, files, file_batch_size:int=64, last_file_path:str=None):
        super().__init__()
        self.files = sorted(glob(files))
        self.file_batch_size = file_batch_size
        if last_file_path == None:
            self.last_file = files[-1]
        else:
            self.last_file = last_file_path
            
    def __len__(self):
        return (len(self.files)-1) * self.file_batch_size + torch.load(self.last_file).shape[0]

    def __getitem__(self, idx):
        num1 = idx // self.file_batch_size
        num2 = idx % self.file_batch_size
        return torch.load(self.files[num1])[num2]