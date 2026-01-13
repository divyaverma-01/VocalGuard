from torch.utils.data import DataLoader
from src.data.sequence_dataset import SequenceDataset

def create_sequence_dataloaders(sequence_csv:str, batch_size:int = 8, num_workers:int = 2,):
    
    train_ds = SequenceDataset(sequence_csv, split="train")
    val_ds = SequenceDataset(sequence_csv, split="val")
    test_ds = SequenceDataset(sequence_csv, split="test")

    train_loader = DataLoader(train_ds, batch_size= batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size= batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size= batch_size, num_workers=num_workers, shuffle=False)

    return train_loader,val_loader,test_loader