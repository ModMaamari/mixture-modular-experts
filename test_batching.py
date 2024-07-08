from torch.utils.data import IterableDataset, DataLoader
from collections import defaultdict, Counter
import torch
import random

# Mock Data: Creating simple mock datasets for 'en', 'fr', 'de', and 'py'
def create_mock_dataset(size, language_code):
    return [{"input_ids": torch.randint(0, 100, (10,)), "language": language_code} for _ in range(size)]

mock_datasets = {
    'en': create_mock_dataset(1000, 0),
    'fr': create_mock_dataset(1000, 1),
    'de': create_mock_dataset(303, 2),
    'py': create_mock_dataset(100, 3),
}

class BalancedLanguageBatchDataset(IterableDataset):
    def __init__(self, datasets, batch_size, device, random_batches=False):
        self.datasets = datasets
        self.device = device
        self.batch_size = batch_size
        self.batch_size_per_lang = batch_size // 4
        self.random_batches = random_batches
        self.iterators = {lang: iter(ds) for lang, ds in datasets.items()}
        self.dataset_sizes = {lang: len(ds) for lang, ds in datasets.items()}
        self.max_dataset_size = max(self.dataset_sizes.values())
        self.total_iterations_needed = self.max_dataset_size // self.batch_size_per_lang  # Total iterations needed to cover the largest dataset
        
    def restart_iterator(self, lang):
        """Restart an iterator for a specific language."""
        self.iterators[lang] = iter(self.datasets[lang])

    def __iter__(self):
        self.current_iteration = 0
        return self

    def __next__(self):
        if self.current_iteration >= self.total_iterations_needed:
            raise StopIteration
        batch = defaultdict(list)

        if self.random_batches:
            for _ in range(self.batch_size):
                lang = random.choice(list(self.datasets.keys()))
                try:
                    item = next(self.iterators[lang])
                except StopIteration:
                    self.restart_iterator(lang)
                    item = next(self.iterators[lang])
                
                for key in item:
                    batch[key].append(item[key])
        
        else:
            for lang in self.iterators.keys():
                for _ in range(self.batch_size_per_lang):
                    try:
                        item = next(self.iterators[lang])
                    except StopIteration:
                        self.restart_iterator(lang)
                        item = next(self.iterators[lang])
                    
                    for key in item:
                        batch[key].append(item[key])

        batch = {k: torch.stack([torch.tensor(x) if not isinstance(x, torch.Tensor) else x for x in v], dim=0).to(self.device) for k, v in batch.items()}
        self.current_iteration += 1
        return batch

def create_balanced_dataloaders(train_datasets_separated, batch_size):
    train_balanced_ds = BalancedLanguageBatchDataset(train_datasets_separated, batch_size, device='cpu', random_batches=True)
    train_dataloader = DataLoader(train_balanced_ds, batch_size=None)  # batch_size=None because dataset yields batches directly
    return train_dataloader

# Define batch size
batch_size = 64

# Create the dataloader
train_dataloader = create_balanced_dataloaders(mock_datasets, batch_size)

# Check a few batches
num_batches_to_check = 5

for i, batch in enumerate(train_dataloader):
    if i <= num_batches_to_check:
        print(f"Batch {i+1}")
        print("Input IDs:", batch['input_ids'])
        print("Languages:", batch['language'])
        print("Counts per language:", Counter(batch['language'].tolist()))
        print("\n")
    
    
    print(f" Batch{i} : Counts per language: {Counter(batch['language'].tolist())}" )
