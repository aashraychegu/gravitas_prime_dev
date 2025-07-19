import torch


class FileSource:
    def __init__(self, path, device="cpu", eos_to_index_map=None) -> None:
        self.spectrograms, self.parameters = torch.load(path, map_location=device)
        self.length = len(self.spectrograms)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.parameters[int(index[0])][-1] = index[1]
        return (self.spectrograms[int(index[0])], self.parameters[int(index[0])])
