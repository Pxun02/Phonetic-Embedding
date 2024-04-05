import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class toDataset(Dataset):
    def __init__(self, inputs, labels, character):
        self.inputs = inputs
        self.labels = labels
        self.character = character
        
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        label = self.labels[idx]
        return input, label
    
def create_dataset(inputs, labels, character, flatten=True):
    inp = inputs
    if flatten:
        inp = flatten_data(inputs)
    X_train, X_val, y_train, y_val, cc_train, cc_val = train_test_split(inp, labels, character, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test, cc_val, cc_test = train_test_split(X_val, y_val, cc_val, test_size=0.5, random_state=42)

    train_dataset = toDataset(X_train, y_train, cc_train)
    val_dataset = toDataset(X_val, y_val, cc_val)
    test_dataset = toDataset(X_test, y_test, cc_test)
    
    return train_dataset, val_dataset, test_dataset

def create_dataloader(train_dataset, val_dataset, batch_size=128, shuffle=False):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader

def flatten_data(dataset):
    temp = []
    for d in dataset:
        temp.append(np.hstack(d))
    return np.array(temp)