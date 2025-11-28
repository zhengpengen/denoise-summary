import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import nltk
import wandb
from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast, GradScaler # Import for Mixed Precision

# Your project's imports
from dataloader import get_summarization_dataloaders
from summarization_model import SummarizationDenoiser
import utils

# --- 1. Diffusion-related components (Noise Scheduler) ---

def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Returns a linear schedule for beta (noise variance)."""
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start: torch.Tensor, t: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Forward process for discrete data (binary mask).
    """
    alpha_bar_t = alphas_cumprod.gather(-1, t).view(-1, 1) # Shape: (B, 1)
    
    prob_x_t_is_one = alpha_bar_t * x_start + (1 - alpha_bar_t) * (1 - x_start)

    # Sample from a Bernoulli distribution
    noisy_mask = torch.bernoulli(prob_x_t_is_one.float()).long()
    
    return noisy_mask


def train_one_epoch(model, dataloader, optimizer, scaler, alphas_cumprod, device, config, sentence_model):
    """Runs a single training epoch with GPU embedding and Mixed Precision."""
    model.train()
    total_loss = 0.0
    timesteps = config.diffusion.timesteps

    for batch in dataloader:
        optimizer.zero_grad()

        # --- 1. GPU Embedding Generation ---
        # The DataLoader default_collate turns a list of lists of strings into a list of tuples.
        # Structure: List of L tuples, where each tuple is size B.
        transposed_sents = batch['article_sents'] 
        
        # We unzip this back into a list of B lists of L strings
        raw_sents_rows = list(zip(*transposed_sents))
        
        # Flatten into a single list of strings for the encoder: [Batch * Max_Len]
        flat_sents = [sent for doc in raw_sents_rows for sent in doc]

        with torch.no_grad():
            # Encode on GPU
            flat_embeddings = sentence_model.encode(
                flat_sents, 
                convert_to_tensor=True, 
                device=device, 
                show_progress_bar=False
            )
        
        # Reshape back to (B, L, D)
        B = len(raw_sents_rows)
        L_max = len(transposed_sents)
        D = flat_embeddings.shape[1]
        
        sentence_embeddings = flat_embeddings.view(B, L_max, D)
        # -----------------------------------

        summary_mask = batch['summary_mask'].to(device) # This is x_0
        attention_mask = batch['attention_mask'].to(device)
        
        # Mask out embeddings for padding strings ("") so they don't affect attention
        sentence_embeddings = sentence_embeddings * attention_mask.unsqueeze(-1)

        # 2. Sample a random timestep t
        t = torch.randint(0, timesteps, (B,), device=device).long()

        # 3. Create noisy mask x_t
        noisy_mask = q_sample(summary_mask, t, alphas_cumprod)

        # 4. Mixed Precision Forward Pass & Loss
        with autocast(device_type='cuda', dtype=torch.float16):
            predicted_logits = model(sentence_embeddings, noisy_mask, t, attention_mask)

            loss = F.cross_entropy(predicted_logits.view(-1, 2), summary_mask.view(-1), reduction='none')
            loss = loss.view(B, L_max)
            loss = (loss * attention_mask).sum() / attention_mask.sum()

        # 5. Scaled Backpropagation
        scaler.scale(loss).backward()
        
        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.gradient_clip_val)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, alphas_cumprod, device, config, sentence_model):
    """Runs a single validation epoch."""
    model.eval()
    total_loss = 0.0
    timesteps = config.diffusion.timesteps

    with torch.no_grad():
        for batch in dataloader:
            
            # --- GPU Embedding Logic (Same as Train) ---
            transposed_sents = batch['article_sents'] 
            raw_sents_rows = list(zip(*transposed_sents))
            flat_sents = [sent for doc in raw_sents_rows for sent in doc]

            flat_embeddings = sentence_model.encode(
                flat_sents, 
                convert_to_tensor=True, 
                device=device, 
                show_progress_bar=False
            )
            
            B = len(raw_sents_rows)
            L_max = len(transposed_sents)
            D = flat_embeddings.shape[1]
            sentence_embeddings = flat_embeddings.view(B, L_max, D)
            # -------------------------------------------

            summary_mask = batch['summary_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            sentence_embeddings = sentence_embeddings * attention_mask.unsqueeze(-1)
            
            t = torch.randint(0, timesteps, (B,), device=device).long()
            noisy_mask = q_sample(summary_mask, t, alphas_cumprod)

            # Use autocast for validation too
            with autocast(device_type='cuda', dtype=torch.float16):
                predicted_logits = model(sentence_embeddings, noisy_mask, t, attention_mask)

                loss = F.cross_entropy(predicted_logits.view(-1, 2), summary_mask.view(-1), reduction='none')
                loss = loss.view(B, L_max)
                loss = (loss * attention_mask).sum() / attention_mask.sum()

            total_loss += loss.item()

    return total_loss / len(dataloader)


@hydra.main(version_base=None, config_path='.', config_name='config')
def train(config: DictConfig):
    try:
      nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
      nltk.download('punkt_tab')

    """Main training function."""
    LOGGER = utils.get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Initialize WandB ---
    if config.wandb.mode != 'disabled':
        try:
            wandb.login(key='590cce1cdb16ab5451d230d3c1630a3897cda782')
        except (ImportError, NameError):
            LOGGER.info("Not in a Colab environment. Assuming 'wandb login' has been run.")
        
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config_dict,
            mode=config.wandb.mode,
            name=os.path.basename(os.getcwd())
        )

    LOGGER.info(f"Using device: {device}")

    # --- Initialize Sentence Transformer on GPU ---
    LOGGER.info("Loading embedding model to GPU...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
    sentence_model.eval() # Ensure it's in eval mode

    # --- Dataloaders ---
    LOGGER.info("Setting up dataloaders...")
    train_loader, valid_loader = get_summarization_dataloaders(config)

    # --- Model ---
    LOGGER.info("Setting up model...")
    model = SummarizationDenoiser(config).to(device)

    # --- Optimizer & Scaler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    scaler = GradScaler(device='cuda') # Initialize Mixed Precision Scaler

    # --- WandB Watch ---
    if config.wandb.mode != 'disabled':
        wandb.watch(model, log='all', log_freq=config.trainer.log_every_n_steps)

    # --- Diffusion Schedule ---
    timesteps = config.diffusion.timesteps
    betas = linear_beta_schedule(
        timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end
    ).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # --- State for Checkpointing ---
    best_val_loss = float('inf')
    start_epoch = 0
    checkpoint_dir = os.getcwd()
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    LOGGER.info(f"Checkpoints will be saved in: {checkpoint_dir}")

    # --- Resume ---
    if config.trainer.resume_from_checkpoint and os.path.exists(config.trainer.resume_from_checkpoint):
        LOGGER.info(f"Resuming training from checkpoint: {config.trainer.resume_from_checkpoint}")
        checkpoint = torch.load(config.trainer.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        LOGGER.info(f"Resumed from epoch {start_epoch-1}. Best validation loss was {best_val_loss:.4f}.")

    # --- Training Loop ---
    LOGGER.info("Starting training...")
    num_epochs = config.trainer.max_steps // len(train_loader)

    for epoch in range(start_epoch, num_epochs):
        LOGGER.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Pass scaler and sentence_model to train function
        train_loss = train_one_epoch(
            model, tqdm(train_loader, desc=f"Training Epoch {epoch+1}"), optimizer, scaler, alphas_cumprod, device, config, sentence_model
        )
        LOGGER.info(f"Average Training Loss: {train_loss:.4f}")

        val_loss = validate_one_epoch(
            model, tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}"), alphas_cumprod, device, config, sentence_model
        )
        LOGGER.info(f"Average Validation Loss: {val_loss:.4f}")

        if config.wandb.mode != 'disabled':
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            LOGGER.info(f"New best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")

    LOGGER.info("Training finished.")

    if config.wandb.mode != 'disabled':
        wandb.finish()

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    train()