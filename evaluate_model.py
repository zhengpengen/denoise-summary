import os
import torch
import hydra
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from omegaconf import DictConfig

# Import from your existing files
from dataloader import SummarizationDataset
from summarization_model import SummarizationDenoiser
from train_summarizer import linear_beta_schedule
import utils

LOGGER = utils.get_logger(__name__)

@torch.no_grad()
def p_sample_loop(model, sentence_embeddings, attention_mask, alphas, alphas_cumprod, config):
    device = sentence_embeddings.device
    timesteps = config.diffusion.timesteps
    batch_size, seq_len = attention_mask.shape

    current_mask = torch.zeros((batch_size, seq_len), device=device).long()

    for t_step in tqdm(reversed(range(timesteps)), desc="Generating Summary", total=timesteps, leave=False):
        t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

        # Predict clean mask
        predicted_logits = model(sentence_embeddings, current_mask, t, attention_mask)
        predicted_x0_prob = torch.softmax(predicted_logits, dim=-1)

        # --- D3PM Math ---
        alpha_t = alphas[t].view(-1, 1)
        alpha_bar_t = alphas_cumprod[t].view(-1, 1)
        if t_step > 0:
            alpha_bar_t_prev = alphas_cumprod[t-1].view(-1, 1)
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device).view(-1, 1)

        p_x0_is_0 = predicted_x0_prob[..., 0]
        p_x0_is_1 = predicted_x0_prob[..., 1]
        x_t_is_0 = (current_mask == 0).float()
        x_t_is_1 = (current_mask == 1).float()

        # Posterior probability: q(x_{t-1}=1 | x_t, x_0)
        term1 = (alpha_t * x_t_is_1 + (1 - alpha_t) * x_t_is_0) * p_x0_is_1 * alpha_bar_t_prev
        term2 = ((1 - alpha_t) * x_t_is_1 + alpha_t * x_t_is_0) * p_x0_is_0 * (1 - alpha_bar_t_prev)
        prob_xt_prev_is_one = term1 + term2

        # Normalize
        denominator = (alpha_bar_t * p_x0_is_1 + (1 - alpha_bar_t) * p_x0_is_0) * x_t_is_1 + \
                      ((1 - alpha_bar_t) * p_x0_is_1 + alpha_bar_t * p_x0_is_0) * x_t_is_0
        
        prob_xt_prev_is_one = prob_xt_prev_is_one / (denominator + 1e-8)

        # Robustness fixes
        prob_xt_prev_is_one = torch.nan_to_num(prob_xt_prev_is_one, nan=0.0)
        prob_xt_prev_is_one = torch.clamp(prob_xt_prev_is_one, min=0.0, max=1.0)

        # Sample
        current_mask = torch.bernoulli(prob_xt_prev_is_one).long()

    return current_mask

def reconstruct_text(sentences, mask):
    """Helper to turn a binary mask + list of sentences into a single string."""
    selected = [sent for sent, m in zip(sentences, mask) if m == 1]
    return " ".join(selected)

@hydra.main(version_base=None, config_path='.', config_name='config')
def evaluate(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # --- 1. Load Data (Test Split) ---
    LOGGER.info("Loading Test Dataset...")
    test_dataset = SummarizationDataset(config, split='test')
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.loader.eval_batch_size,
        shuffle=False,
        num_workers=config.loader.num_workers,
        collate_fn=None 
    )

    # --- 2. Load Model & Embedder ---
    checkpoint_path = config.eval.checkpoint_path 
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    LOGGER.info(f"Loading model from {checkpoint_path}...")
    model = SummarizationDenoiser(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    LOGGER.info("Loading Sentence Transformer...")
    sentence_model = SentenceTransformer(config.summarization.sentence_embedder_name_or_path, device=device)

    # --- 3. Prepare Diffusion Schedule ---
    betas = linear_beta_schedule(config.diffusion.timesteps, config.diffusion.beta_start, config.diffusion.beta_end).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # --- 4. Metrics Storage ---
    all_results = []
    
    # Accumulators for dataset-wide averages
    total_f1 = 0
    total_prec = 0
    total_rec = 0
    total_acc = 0
    total_batches = 0

    LOGGER.info("Starting Evaluation Loop...")
    
    # --- 5. Evaluation Loop ---
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            
            # A. Prepare Embeddings
            transposed_sents = batch['article_sents']
            raw_sents_rows = list(zip(*transposed_sents))
            flat_sents = [sent for doc in raw_sents_rows for sent in doc]
            
            flat_embeddings = sentence_model.encode(flat_sents, convert_to_tensor=True, device=device, show_progress_bar=False)
            
            B = len(raw_sents_rows)
            L_max = len(transposed_sents)
            D = flat_embeddings.shape[1]
            sentence_embeddings = flat_embeddings.view(B, L_max, D)

            # B. Prepare Masks
            gt_mask = batch['summary_mask'].to(device) 
            attention_mask = batch['attention_mask'].to(device)
            
            # Apply mask to embeddings for padding
            sentence_embeddings = sentence_embeddings * attention_mask.unsqueeze(-1)

            # C. Run Diffusion Sampling
            pred_mask_tensor = p_sample_loop(
                model, 
                sentence_embeddings, 
                attention_mask, 
                alphas, 
                alphas_cumprod, 
                config
            )

            # D. Compute Metrics per Sample
            pred_mask_np = pred_mask_tensor.cpu().numpy()
            gt_mask_np = gt_mask.cpu().numpy()
            attn_mask_np = attention_mask.cpu().numpy()

            batch_f1 = []
            batch_prec = []
            batch_rec = []
            batch_acc = []

            for i in range(B):
                # Only evaluate on real sentences (ignore padding)
                doc_len = int(attn_mask_np[i].sum())
                
                p_slice = pred_mask_np[i, :doc_len]
                g_slice = gt_mask_np[i, :doc_len]
                
                # Calculate metrics for this specific document
                f1 = f1_score(g_slice, p_slice, zero_division=0)
                prec = precision_score(g_slice, p_slice, zero_division=0)
                rec = recall_score(g_slice, p_slice, zero_division=0)
                acc = accuracy_score(g_slice, p_slice)

                batch_f1.append(f1)
                batch_prec.append(prec)
                batch_rec.append(rec)
                batch_acc.append(acc)

                # Store textual representation for qualitative review
                curr_sents = list(raw_sents_rows[i])[:doc_len]
                pred_text = reconstruct_text(curr_sents, p_slice)
                gt_text = reconstruct_text(curr_sents, g_slice)
                
                # Capture raw data for output file
                all_results.append({
                    'full_source_text': " ".join(curr_sents).replace('\n', ' '), # Ensure single line
                    'ground_truth_mask': str(g_slice.tolist()), 
                    'predicted_mask': str(p_slice.tolist()),
                    'mask_f1': f1,
                    'mask_precision': prec,
                    'mask_recall': rec
                })

            # Update totals
            total_f1 += np.mean(batch_f1)
            total_prec += np.mean(batch_prec)
            total_rec += np.mean(batch_rec)
            total_acc += np.mean(batch_acc)
            total_batches += 1
            
            break

    # --- 6. Aggregate Results ---
    avg_f1 = total_f1 / total_batches
    avg_prec = total_prec / total_batches
    avg_rec = total_rec / total_batches
    avg_acc = total_acc / total_batches
    
    # --- 7. Write to Text File ---
    txt_output_filename = "evaluation_predictions.txt"
    LOGGER.info(f"Saving text-formatted results to {txt_output_filename}...")
    
    with open(txt_output_filename, "w", encoding="utf-8") as f:
        for item in all_results:
            # Line 1: Source Text
            f.write(f"{item['full_source_text']}\n")
            # Line 2: Ground Truth Mask
            f.write(f"{item['ground_truth_mask']}\n")
            # Line 3: Predicted Mask
            f.write(f"{item['predicted_mask']}\n")

    print("\n" + "="*40)
    print(f"EVALUATION COMPLETE. Results saved to {txt_output_filename}")
    print("="*40)
    print(f"Accuracy:  {avg_acc:.4f}")
    print(f"F1 Score:  {avg_f1:.4f}")
    print(f"Precision: {avg_prec:.4f}")
    print(f"Recall:    {avg_rec:.4f}")
    print("="*40 + "\n")

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    evaluate()