import datasets
import nltk
import torch
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

import utils

LOGGER = utils.get_logger(__name__)

# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    LOGGER.info("NLTK 'punkt' model not found. Downloading...")
    nltk.download('punkt')


class SummarizationDataset(torch.utils.data.Dataset):
    """Dataset for extractive summarization framed as a diffusion task."""

    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.max_sentences = self.config.summarization.max_sentences

        LOGGER.info(f"Loading dataset {self.config.data.train} for split {self.split}")
        self.dataset = datasets.load_dataset(
            self.config.data.train, '3.0.0',
            split=self.split,
            cache_dir=self.config.data.cache_dir
        )

        LOGGER.info("Loading sentence embedding model...")
        self.sentence_embedder = SentenceTransformer(
            self.config.summarization.sentence_embedder_name_or_path
        )
        self.embedding_dim = self.sentence_embedder.get_sentence_embedding_dimension()

        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def __len__(self):
        return len(self.dataset)

    def _create_oracle_summary_mask(self, article_sents, summary_sents):
        """Greedily select sentences from article to maximize ROUGE-L with summary."""
        selected_indices = []
        # ROUGE-L is calculated over the full text, not sentence-by-sentence.
        summary_text = " ".join(summary_sents)

        # Limit to max_sentences to avoid excessive computation
        article_sents = article_sents[:self.max_sentences]

        # Greedily add sentences that improve ROUGE-L the most
        while True:
            best_sent_idx = -1
            best_rouge_gain = 0.0  # We only want to add sentences that provide a positive gain.

            current_summary_text = " ".join([article_sents[i] for i in selected_indices])
            # If there's no current summary, base ROUGE is 0
            base_rouge = self.rouge.score(target=summary_text, prediction=current_summary_text)['rougeL'].fmeasure if current_summary_text else 0.0

            for i, sent in enumerate(article_sents):
                if i in selected_indices:
                    continue

                # Ensure there's a space between sentences
                new_summary_text = (current_summary_text + " " + sent).strip()
                new_rouge = self.rouge.score(target=summary_text, prediction=new_summary_text)['rougeL'].fmeasure
                rouge_gain = new_rouge - base_rouge

                if rouge_gain > best_rouge_gain:
                    best_rouge_gain = rouge_gain
                    best_sent_idx = i

            # If we found a sentence that improves the score, add it and continue.
            if best_sent_idx != -1:
                selected_indices.append(best_sent_idx)
            else:
                break  # No more improvement possible

        mask = torch.zeros(len(article_sents), dtype=torch.long)
        if selected_indices:
            # Sort indices to maintain original sentence order in the mask
            selected_indices.sort()
            mask[selected_indices] = 1
        return mask

    def __getitem__(self, idx):
        item = self.dataset[idx]
        article = item['article']
        summary = item['highlights']

        article_sents = nltk.sent_tokenize(article)
        summary_sents = nltk.sent_tokenize(summary)

        # Truncate if necessary
        article_sents = article_sents[:self.max_sentences]
        num_sents = len(article_sents)

        # 1. Create oracle mask (x_0)
        oracle_mask = self._create_oracle_summary_mask(list(article_sents), list(summary_sents))

        # 2. Embed article sentences
        sentence_embeddings = self.sentence_embedder.encode(
            article_sents, convert_to_tensor=True,
            normalize_embeddings=True
        )

        # 3. Pad to max_sentences
        padded_embeddings = torch.zeros(self.max_sentences, self.embedding_dim)
        padded_mask = torch.zeros(self.max_sentences, dtype=torch.long)
        attention_mask = torch.zeros(self.max_sentences, dtype=torch.long)

        padded_embeddings[:num_sents] = sentence_embeddings
        padded_mask[:num_sents] = oracle_mask
        attention_mask[:num_sents] = 1

        return {
            'sentence_embeddings': padded_embeddings,
            'summary_mask': padded_mask,
            'attention_mask': attention_mask
        }

def get_summarization_dataloaders(config):
    train_set = SummarizationDataset(config, split='train')
    # For validation, cnn_dailymail uses the 'validation' split. For other datasets, it might be 'test'.
    # It's good practice to check which splits are available.
    # For cnn_dailymail, splits are 'train', 'validation', 'test'.
    valid_set = SummarizationDataset(config, split='validation') 
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.loader.batch_size,
        shuffle=True,
        num_workers=config.loader.num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=config.loader.eval_batch_size,
        shuffle=False, num_workers=config.loader.num_workers
    )
    return train_loader, valid_loader