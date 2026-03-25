import torch
from torch.utils.data import Dataset

class IndianNamesDataset(Dataset):
    """
    Handles character-level tokenization and sequence padding 
    for the Indian Names dataset.
    """
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.names = [line.strip().lower() for line in f if line.strip()]
            
        # Extract unique characters and add special tokens
        unique_chars = sorted(list(set(''.join(self.names))))
        self.pad, self.sos, self.eos = '<PAD>', '<SOS>', '<EOS>'
        self.vocab = [self.pad, self.sos, self.eos] + unique_chars
        
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.max_seq_len = max(len(name) for name in self.names) + 2

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        # Build sequence: <SOS> name <EOS>
        tokens = [self.sos] + list(name) + [self.eos]
        indices = [self.char_to_idx[token] for token in tokens]
        
        # Apply padding to ensure uniform batch sizes
        padded = indices + [self.char_to_idx[self.pad]] * (self.max_seq_len - len(indices))
        
        # Shifted input and target for next-character prediction
        x_tensor = torch.tensor(padded[:-1], dtype=torch.long)
        y_tensor = torch.tensor(padded[1:], dtype=torch.long)
        
        return x_tensor, y_tensor