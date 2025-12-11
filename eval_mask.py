import os
import math
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
# from dit_model import SummarizationDenoiser
import utils

LOGGER = utils.get_logger(__name__)

# --- 1. Define Scheduler locally to match train_mask.py ---
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    Matches train_mask.py implementation.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

@torch.no_grad()
def p_sample_loop(model, sentence_embeddings, attention_mask, alphas_cumprod, config):
    """
    Reverse process for Absorbing State Diffusion (Masking).
    Forward: 0 (Data) -> 1 (Mask).
    Reverse: 1 (Mask) -> 0 (Data) based on model prediction.
    """
    device = sentence_embeddings.device
    timesteps = config.diffusion.timesteps
    batch_size, seq_len = attention_mask.shape

    # Start with pure noise (all 1s / fully masked)
    # In absorbing diffusion, the prior is all 1s.
    current_mask = torch.ones((batch_size, seq_len), device=device).long()

    for t_step in tqdm(reversed(range(timesteps)), desc="Generating Summary", total=timesteps, leave=False):
        t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

        # 1. Predict x_0 logits
        predicted_logits = model(sentence_embeddings, current_mask, t, attention_mask)
        predicted_probs = torch.softmax(predicted_logits, dim=-1)
        
        # p_x0_0: Probability token should be KEPT (0)
        # p_x0_1: Probability token should be MASKED (1)
        p_x0_0 = predicted_probs[..., 0]
        p_x0_1 = predicted_probs[..., 1]

        # 2. Get scheduling values
        alpha_bar_t = alphas_cumprod[t_step]
        alpha_bar_t_prev = alphas_cumprod[t_step - 1] if t_step > 0 else torch.tensor(1.0, device=device)
        
        # 3. Calculate Posterior Probability P(x_{t-1} = 1 | x_t = 1, x_0)
        # If x_t is 0, it stays 0 (Absorbing state logic reversed).
        # If x_t is 1, it might flip to 0.
        
        # The probability of STAYING masked (1) given we assume x_0 is 0:
        # ratio = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # This ratio represents "how much noise is left at t-1 vs t"
        
        # Clamp denominator to avoid division by zero at t=T if alpha_bar_T approx 0
        denom = 1 - alpha_bar_t
        if abs(denom) < 1e-6:
             ratio = 1.0 # Fallback, though usually 1-alpha_bar_t is large at t=0
        else:
             ratio = (1 - alpha_bar_t_prev) / denom
        
        # Full posterior for state 1:
        # P(x_{t-1}=1) = P(x_0=1) * 1.0 + P(x_0=0) * ratio
        prob_next_is_1 = p_x0_1 + p_x0_0 * ratio
        
        # Clip probabilities
        prob_next_is_1 = torch.clamp(prob_next_is_1, 0.0, 1.0)

        # 4. Sample
        sample_is_1 = torch.bernoulli(prob_next_is_1).long()
        
        # 5. Apply Update Logic
        # If current_mask was 0, it MUST stay 0 (cannot re-mask revealed tokens).
        # If current_mask was 1, it becomes sample_is_1.
        current_mask = current_mask * sample_is_1

    return current_mask

def reconstruct_text(sentences, mask):
    """Helper to turn a binary mask + list of sentences into a single string."""
    # Note: mask 1 means "masked" (removed) in diffusion terms, but "selected" in summary terms?
    # CHECK: In train_mask.py: q_sample takes summary_mask (x_start).
    # summary_mask: 1 usually means "In Summary", 0 means "Not in Summary".
    # BUT train_mask.py says: "Transition: Data (mostly 0s) -> Pure 1s (Noise)."
    # And: "noisy_mask = x_start | corruption_mask"
    # This implies 1 is the ABSORBING state.
    # If the summary is the sparse signal (1s) and we absorb to 1s, then noise is "everything selected".
    # If standard text masking (MaskGIT), 1 is [MASK] token.
    # Let's assume for Extractive Summarization:
    # 1 = Selected/Masked(Noise)? 
    # Usually: 1 = Sentence Included in Summary.
    # The diffusion process adds MORE 1s until everything is 1.
    # So clean state = Sparse 1s. Noisy state = All 1s.
    
    # Therefore, at the end of generation, we have a mask of 1s and 0s.
    # 1 = Selected Sentence. 0 = Rejected Sentence.
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

    # --- 3. Prepare Diffusion Schedule (COSINE) ---
    # Using the local cosine_beta_schedule function
    betas = cosine_beta_schedule(config.diffusion.timesteps).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # --- 4. Metrics Storage ---
    all_results = []
    
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

            # C. Run Diffusion Sampling (Updated)
            pred_mask_tensor = p_sample_loop(
                model, 
                sentence_embeddings, 
                attention_mask, 
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
                
                all_results.append({
                    'full_source_text': " ".join(curr_sents).replace('\n', ' '), 
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

    # --- 6. Aggregate Results ---
    avg_f1 = total_f1 / total_batches if total_batches > 0 else 0
    avg_prec = total_prec / total_batches if total_batches > 0 else 0
    avg_rec = total_rec / total_batches if total_batches > 0 else 0
    avg_acc = total_acc / total_batches if total_batches > 0 else 0
    
    # --- 7. Write to Text File ---
    txt_output_filename = "evaluation_predictions.txt"
    LOGGER.info(f"Saving text-formatted results to {txt_output_filename}...")
    
    with open(txt_output_filename, "w", encoding="utf-8") as f:
        for item in all_results:
            f.write(f"Source: {item['full_source_text']}\n")
            f.write(f"GT:   {item['ground_truth_mask']}\n")
            f.write(f"Pred: {item['predicted_mask']}\n")
            f.write("-" * 20 + "\n")

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