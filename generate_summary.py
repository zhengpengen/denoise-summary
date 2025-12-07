import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
import nltk
from sentence_transformers import SentenceTransformer

from summarization_model import SummarizationDenoiser
from train_summarizer import linear_beta_schedule
import utils

LOGGER = utils.get_logger(__name__)

@torch.no_grad()
def p_sample_loop(model, sentence_embeddings, attention_mask, alphas, alphas_cumprod, config):
    """
    Implements the reverse diffusion process (sampling) to generate a summary mask.
    """
    device = sentence_embeddings.device
    timesteps = config.diffusion.timesteps
    batch_size, seq_len = attention_mask.shape

    # 1. Start with a random mask (pure noise at T)
    # This is x_T, which is equivalent to a Bernoulli sample with p=0.5
    current_mask = torch.randint(0, 2, (batch_size, seq_len), device=device).long()

    # 2. Iteratively denoise from T-1 down to 0
    for t_step in tqdm(reversed(range(timesteps)), desc="Generating Summary", total=timesteps, leave=False):
        t = torch.full((batch_size,), t_step, device=device, dtype=torch.long)

        # Get the model's prediction for the clean mask (x_0)
        predicted_logits = model(sentence_embeddings, current_mask, t, attention_mask)
        predicted_x0_prob = torch.softmax(predicted_logits, dim=-1) # (B, L, 2)

        # --- D3PM Reverse Step ---
        # Calculate q(x_{t-1} | x_t, x_0)
        # This gives the probability of the mask at the previous step, given the current noisy mask
        # and the model's prediction of the clean mask.

        alpha_t = alphas[t].view(-1, 1)
        alpha_bar_t = alphas_cumprod[t].view(-1, 1)
        alpha_bar_t_prev = alphas_cumprod[t-1] if t_step > 0 else torch.tensor(1.0, device=device)
        alpha_bar_t_prev = alpha_bar_t_prev.view(-1, 1)

        # Probabilities for the two classes (0 and 1) from the model's prediction
        p_x0_is_0 = predicted_x0_prob[..., 0]
        p_x0_is_1 = predicted_x0_prob[..., 1]

        # Current mask values
        x_t_is_0 = (current_mask == 0).float()
        x_t_is_1 = (current_mask == 1).float()

        # Calculate terms for the posterior probability q(x_{t-1}=1 | x_t, x_0)
        # This is derived from the D3PM paper's formulas for binary data.
        # Term for x_0 = 1
        term1 = (alpha_t * x_t_is_1 + (1 - alpha_t) * x_t_is_0) * p_x0_is_1 * alpha_bar_t_prev
        # Term for x_0 = 0
        term2 = ((1 - alpha_t) * x_t_is_1 + alpha_t * x_t_is_0) * p_x0_is_0 * (1 - alpha_bar_t_prev)

        # The probability that the mask at the previous step was 1
        prob_xt_prev_is_one = term1 + term2

        # Normalize to get a valid probability
        # The denominator is q(x_t | x_0)
        denominator = (alpha_bar_t * p_x0_is_1 + (1 - alpha_bar_t) * p_x0_is_0) * x_t_is_1 + \
                      ((1 - alpha_bar_t) * p_x0_is_1 + alpha_bar_t * p_x0_is_0) * x_t_is_0
        
        # Avoid division by zero
        prob_xt_prev_is_one = prob_xt_prev_is_one / (denominator + 1e-8)
        prob_xt_prev_is_one = torch.clamp(prob_xt_prev_is_one, min=0.0, max=1.0) 


        # Sample the mask for the previous step
        current_mask = torch.bernoulli(prob_xt_prev_is_one).long()

    # The final mask at t=0
    return current_mask

@hydra.main(version_base=None, config_path='.', config_name='config')
def generate(config: DictConfig):
    """Main function to generate a summary for a given article."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # --- Load Model from Checkpoint ---
    # Assuming the checkpoint is in the 'outputs' directory from a training run.
    # You might need to adjust this path.
    checkpoint_path = config.eval.checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Please provide a valid path in the config.")
    
    LOGGER.info(f"Loading model from: {checkpoint_path}")
    model = SummarizationDenoiser(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # --- Load Sentence Embedder ---
    sentence_embedder = SentenceTransformer(config.summarization.sentence_embedder_name_or_path, device=device)

    # --- Prepare Article ---
    # Example article. You can replace this with any text you want to summarize.
    article_text = (
        "Researchers have made a significant breakthrough in battery technology. "
        "The new design uses a novel solid-state electrolyte, which is safer and more energy-dense than current liquid electrolytes. "
        "This could dramatically increase the range of electric vehicles and the lifespan of consumer electronics. "
        "The team lead, Dr. Eva Rostova, stated that the technology is still several years away from commercialization. "
        "However, she is optimistic about its potential to revolutionize the energy storage industry. "
        "Manufacturing the new electrolyte at scale remains a significant engineering challenge."
    )
    
    article_sents = nltk.sent_tokenize(article_text)
    num_sents = len(article_sents)
    LOGGER.info(f"Article has {num_sents} sentences.")

    # --- Preprocess for Model ---
    sentence_embeddings = sentence_embedder.encode(article_sents, convert_to_tensor=True, normalize_embeddings=True)
    
    # Pad to the model's expected input size
    padded_embeddings = torch.zeros(1, config.summarization.max_sentences, sentence_embedder.get_sentence_embedding_dimension(), device=device)
    attention_mask = torch.zeros(1, config.summarization.max_sentences, device=device, dtype=torch.long)

    padded_embeddings[0, :num_sents] = sentence_embeddings
    attention_mask[0, :num_sents] = 1

    # --- Diffusion Schedule ---
    betas = linear_beta_schedule(config.diffusion.timesteps, config.noise.beta_start, config.noise.beta_end).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # --- Generate Summary Mask ---
    final_mask = p_sample_loop(model, padded_embeddings, attention_mask, alphas, alphas_cumprod, config)
    final_mask = final_mask.squeeze(0)[:num_sents] # Remove batch dim and padding

    # --- Print Result ---
    LOGGER.info("\n--- Generated Summary ---")
    summary_sentences = [sent for i, sent in enumerate(article_sents) if final_mask[i] == 1]
    if not summary_sentences:
        LOGGER.warning("Model did not select any sentences for the summary.")
    else:
        print(" ".join(summary_sentences))

if __name__ == '__main__':
    generate()