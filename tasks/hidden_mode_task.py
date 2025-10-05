import torch
from torch.utils.data import Dataset
import pandas as pd

class HiddenModeConfig:
    """Configuration for the Hidden-Mode task."""
    def __init__(self, url: str, seed: int = 42):
        self.url = url
        self.seed = seed


class HiddenModeDataset(Dataset):
    """Dataset for the Hidden-Mode task from Jackson Petty's 'SSM Illusion' post."""
    def __init__(self, config: HiddenModeConfig, max_seq_len: int = 64):
        self.config = config
        self.max_seq_len = max_seq_len
        self.data = self._load_and_process_data()
        self.pad_token = "<pad>"
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.pad_value = self.stoi[self.pad_token]

    def _load_and_process_data(self):
        """Loads data from the URL and splits it."""
        df = pd.read_csv(self.config.url)
        # The CSV has 'input' and 'target' columns
        df = df[['input', 'target']]
        # The target is just a number, convert it to string
        df['target'] = df['target'].astype(str)
        return df
    def _build_vocab(self):
        """Builds a vocabulary from the characters in the dataset."""
        chars = set()
        for _, row in self.data.iterrows():
            chars.update(list(row['input']))
            chars.update(list(row['target']))
        
        # Add pad token to vocab
        vocab = sorted(list(chars))
        vocab.append(self.pad_token)
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data.iloc[idx]['input']
        completion = self.data.iloc[idx]['target']

        # The input to the model is the prompt, the target is the completion
        # For sequence models, we often concatenate them for training
        full_sequence = prompt + completion
        
        # Encode the sequence
        encoded_sequence = [self.stoi[char] for char in full_sequence]

        # Pad or truncate the sequence to max_seq_len
        seq_len = len(encoded_sequence)
        if seq_len < self.max_seq_len + 1:
            # Pad sequence
            padding = [self.pad_value] * (self.max_seq_len + 1 - seq_len)
            encoded_sequence.extend(padding)
        elif seq_len > self.max_seq_len + 1:
            # Truncate sequence
            encoded_sequence = encoded_sequence[:self.max_seq_len + 1]

        x = torch.tensor(encoded_sequence[:-1], dtype=torch.long)
        y = torch.tensor(encoded_sequence[1:], dtype=torch.long)
        
        return x, y
