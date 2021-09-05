import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SentenceOrderingDataset(Dataset):

    def __init__(self, filename):
        
        id_, context1, context2 = [], [], []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                sents = line.strip().split('\t')
                if len(sents) > 1:
                    id_.append(i)
                    context1.append(sents)
                    context2.append(' '.join(sents))
                            
        self.id = id_
        self.context1 = context1
        self.context2 = context2
        
    def __len__(self):
        return len(self.context1)

    def __getitem__(self, index): 
        i = self.id[index]
        c1 = self.context1[index]
        c2 = self.context2[index]
        return i, c1, c2
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]
    
    
def SentenceOrderingLoader(filename, batch_size, shuffle):
    dataset = SentenceOrderingDataset(filename)
    loader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return loader