from torch.utils.data import Dataset
from PIL import Image
import torch

class FineTuneDataset(Dataset):
    
    def __init__(self, signatures, transform):
        self.signatures = signatures
        self.transform = transform
        self.pairs, self.labels = self.create_pairs()
        
    def create_pairs(self):
        pairs = []
        labels = []
        for i in range(len(self.signatures)):
            for j in range(i, len(self.signatures)):
                pairs.append((self.signatures[i], self.signatures[j]))
                labels.append(1)
        
        return pairs, labels
    
    def __getitem__(self, index):
        img0, img1 = self.pairs[index]
        img0 = Image.fromarray(img0)
        img1 = Image.fromarray(img1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.tensor(self.labels[index], dtype=torch.float32)
    
    def __len__(self):
        return len(self.pairs)        