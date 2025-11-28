import nltk
import os
import torch
from rouge_score import rouge_scorer
import logging
import absl.logging # Import absl logging
import datasets
import utils

# --- FIX 1: Silence the absl logger ---
# This stops the "Using default tokenizer" spam
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
absl.logging.set_verbosity(absl.logging.ERROR)

LOGGER = utils.get_logger(__name__)

# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    LOGGER.info("NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt')

class SummarizationDataset(torch.utils.data.Dataset):
    """Dataset for extractive summarization framed as a diffusion task."""

    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.max_sentences = self.config.summarization.max_sentences
        self.dataset = self._load_and_process_dataset()

    def __len__(self):
        return len(self.dataset)

    def _create_oracle_summary_mask(self, article_sents, summary_sents, rouge_scorer_instance):
        """Greedily select sentences from article to maximize ROUGE-L with summary."""
        selected_indices = []
        summary_text = " ".join(summary_sents)
        
        # Performance: Limit source sentences immediately
        article_sents = article_sents[:self.max_sentences]
        
        # Optimization: Pre-calculate the base score only once per loop
        current_summary_text = ""
        base_rouge = 0.0

        while True:
            best_sent_idx = -1
            best_rouge_gain = 0.0
            
            # Optimization: Create a list of candidates to score
            # (Note: ROUGE is expensive. We only score sentences not yet selected)
            for i, sent in enumerate(article_sents):
                if i in selected_indices:
                    continue

                # Construct candidate summary
                candidate_summary = (current_summary_text + " " + sent).strip()
                
                # specific optimization: Only fetch the fmeasure float, don't return the full dict
                score = rouge_scorer_instance.score(target=summary_text, prediction=candidate_summary)
                new_rouge = score['rougeL'].fmeasure
                
                rouge_gain = new_rouge - base_rouge

                if rouge_gain > best_rouge_gain:
                    best_rouge_gain = rouge_gain
                    best_sent_idx = i

            if best_sent_idx != -1:
                selected_indices.append(best_sent_idx)
                # Update base for next iteration
                current_summary_text = (current_summary_text + " " + article_sents[best_sent_idx]).strip()
                base_rouge += best_rouge_gain
            else:
                break 

        mask = torch.zeros(len(article_sents), dtype=torch.long)
        if selected_indices:
            selected_indices.sort()
            mask[selected_indices] = 1
        return mask

    def _preprocess_function(self, examples):
        all_article_sents = []
        all_summary_masks = []
        all_attention_masks = []

        # Instantiate scorer once per batch (per process)
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        for article, summary in zip(examples['article'], examples['highlights']):
            # Robustness: Handle empty inputs
            if not article or not summary:
                article = ""
                summary = ""

            article_sents = nltk.sent_tokenize(article)
            summary_sents = nltk.sent_tokenize(summary)

            article_sents = article_sents[:self.max_sentences]
            num_sents = len(article_sents)

            oracle_mask = self._create_oracle_summary_mask(list(article_sents), list(summary_sents), scorer)

            # Padding logic
            padded_sents = list(article_sents) + [""] * (self.max_sentences - len(article_sents))
            padded_sents = padded_sents[:self.max_sentences] # Ensure strictly cut

            padded_mask = torch.zeros(self.max_sentences, dtype=torch.long)
            attention_mask = torch.zeros(self.max_sentences, dtype=torch.long)
            
            padded_mask[:num_sents] = oracle_mask
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
        dataset = datasets.load_dataset(
            self.config.data.dataset_name,
            self.config.data.version,
            split=self.split,
            cache_dir=self.config.data.cache_dir,
        )

        # --- FIX 2: Maximize CPU Usage ---
        # Greedy ROUGE is CPU-bound. 4 workers is too low.
        # We use os.cpu_count() to use all available cores.
        max_proc = os.cpu_count()
        LOGGER.info(f"Preprocessing dataset with {max_proc} workers...")
        
        processed_dataset = dataset.map(
            self._preprocess_function,
            batched=True,
            # Smaller batch size per process helps keep memory in check with many workers
            batch_size=min(100, self.config.loader.preprocessing_batch_size), 
            num_proc=max_proc, 
            remove_columns=dataset.column_names,
            desc=f"Preprocessing {self.split} set",
            load_from_cache_file=True # Ensure caching is enabled
        )
        processed_dataset.set_format(type='torch', columns=['summary_mask', 'attention_mask'])
        # Note: 'article_sents' is strings, usually can't be set to torch format directly
        # You might need to leave it as python objects or handle it in collate_fn
        
        return processed_dataset

    def __getitem__(self, idx):
        return self.dataset[idx]

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