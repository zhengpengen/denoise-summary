import torch
import datasets
import logging
import absl.logging
import utils
import os

# --- Silence absl logging ---
# logging.root.removeHandler(absl.logging._absl_handler)
# absl.logging._warn_preinit_stderr = False
# absl.logging.set_verbosity(absl.logging.ERROR)

LOGGER = utils.get_logger(__name__)

class SummarizationDataset(torch.utils.data.Dataset):
    """Dataset for extractive summarization using pre-computed binary masks."""

    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.max_sentences = self.config.summarization.max_sentences
        self.dataset = self._load_and_process_dataset()

    def __len__(self):
        return len(self.dataset)

    def _preprocess_function(self, examples):
        all_article_sents = []
        all_summary_masks = []
        all_attention_masks = []

        # Iterate through the batch
        # We assume the dataset has 'sentences' (list of str) and 'labels' (list of 0/1 ints)
        for sentences, mask in zip(examples['src'], examples['labels']):
            
            # 1. Truncate strictly to max_sentences
            # The dataset provides a list of strings, so no tokenization needed.
            article_sents = sentences[:self.max_sentences]
            mask_subset = mask[:self.max_sentences]
            
            num_sents = len(article_sents)

            # 2. Convert Mask to Tensor
            # The 'labels' field is already a binary mask [0, 1, 0, 1...]
            mask_tensor = torch.tensor(mask_subset, dtype=torch.long)

            # 3. Padding Logic
            pad_len = self.max_sentences - num_sents
            
            # Pad sentences with empty strings (handled by embedder later)
            padded_sents = list(article_sents) + [""] * pad_len
            
            # Pad summary mask with 0
            padded_mask = torch.cat([mask_tensor, torch.zeros(pad_len, dtype=torch.long)])
            
            # Attention mask: 1 for real sentences, 0 for padding
            attention_mask = torch.zeros(self.max_sentences, dtype=torch.long)
            attention_mask[:num_sents] = 1

            all_article_sents.append(padded_sents)
            all_summary_masks.append(padded_mask)
            all_attention_masks.append(attention_mask)

        return {
            'article_sents': all_article_sents,
            'summary_mask': all_summary_masks,
            'attention_mask': all_attention_masks
        }

    def _load_and_process_dataset(self):
        LOGGER.info(f"Loading dataset {self.config.data.dataset_name} for split {self.split}")
        
        # Load the dataset
        dataset = datasets.load_dataset(
            self.config.data.dataset_name,
            split=self.split,
            cache_dir=self.config.data.cache_dir,
            trust_remote_code=True
        )

        # Optimization: Use all cores
        # Since we removed ROUGE, this mapping is very lightweight and fast.
        max_proc = os.cpu_count()
        LOGGER.info(f"Formatting dataset with {max_proc} workers...")
        
        processed_dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            # Larger batch size is fine now that the heavy compute is gone
            batch_size=self.config.loader.preprocessing_batch_size, 
            num_proc=max_proc,
            remove_columns=dataset.column_names,
            desc=f"Formatting {self.split} set",
            load_from_cache_file=True
        )
        
        # processed_dataset.set_format(type='torch', columns=['summary_mask', 'attention_mask'])
        
        return processed_dataset

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'article_sents': item['article_sents'], # List of strings (passes through)
            'summary_mask': torch.tensor(item['summary_mask'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long)
        }

def get_summarization_dataloaders(config):
    train_set = SummarizationDataset(config, split='train')
    valid_set = SummarizationDataset(config, split='validation') 
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=config.loader.num_workers,
        pin_memory=True, 
        persistent_workers=True 
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=config.loader.eval_batch_size,
        shuffle=False, 
        num_workers=config.loader.num_workers,
        pin_memory=True,  
        persistent_workers=True
    )
    return train_loader, valid_loader