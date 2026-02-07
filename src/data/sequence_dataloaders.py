from torch.utils.data import DataLoader
from src.data.sequence_dataset import SequenceDataset
from src.data.balanced_batch_sampler import BalancedBatchSampler

def create_sequence_dataloaders(sequence_csv:str, batch_size:int = 8, num_workers:int = 2, target_mode: str = "sequence",):
    
    train_ds = SequenceDataset(sequence_csv, split="train", target_mode=target_mode)
    val_ds = SequenceDataset(sequence_csv, split="val", target_mode=target_mode)
    test_ds = SequenceDataset(sequence_csv, split="test", target_mode=target_mode)

    if target_mode == "escalation":
        # Balanced sampler for escalation mode
        sampler = BalancedBatchSampler(labels=train_ds.escalation_labels, batch_size=batch_size)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers)
    else: 
        # Normal training for sequence_label
        train_loader = DataLoader(train_ds, batch_size= batch_size, num_workers=num_workers, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size= batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size= batch_size, num_workers=num_workers, shuffle=False)

    return train_loader,val_loader,test_loader