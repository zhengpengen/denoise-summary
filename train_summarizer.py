import os
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import hydra
from omegaconf import DictConfig

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
    Corrupts x_start to x_t using probabilities derived from the noise schedule.

    Args:
        x_start (Tensor): The initial clean data (summary_mask), shape (B, L).
        t (Tensor): The timestep for each sample in the batch, shape (B,).
        alphas_cumprod (Tensor): Cumulative product of alphas from the schedule.

    Returns:
        Tensor: The noisy mask at timestep t.
    """
    # Get the cumulative alpha for the given timesteps
    # This value, alpha_bar_t, represents the probability of a bit remaining unchanged.
    alpha_bar_t = alphas_cumprod.gather(-1, t).view(-1, 1) # Shape: (B, 1)

    # For a binary mask, the transition probabilities are:
    # p(x_t=1 | x_0=1) = alpha_bar_t
    # p(x_t=0 | x_0=1) = 1 - alpha_bar_t
    # p(x_t=1 | x_0=0) = 1 - alpha_bar_t
    # p(x_t=0 | x_0=0) = alpha_bar_t
    # This can be simplified. The probability of x_t being 1 is:
    # p(x_t=1) = p(x_t=1|x_0=1)*p(x_0=1) + p(x_t=1|x_0=0)*p(x_0=0)
    #          = alpha_bar_t * x_start + (1 - alpha_bar_t) * (1 - x_start)
    
    prob_x_t_is_one = alpha_bar_t * x_start + (1 - alpha_bar_t) * (1 - x_start)

    # Sample from a Bernoulli distribution with these probabilities
    # to get the noisy mask x_t.
    noisy_mask = torch.bernoulli(prob_x_t_is_one.float()).long()
    
    return noisy_mask


def train_one_epoch(model, dataloader, optimizer, alphas_cumprod, device, config):
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    timesteps = config.diffusion.timesteps

    for batch in dataloader:
        optimizer.zero_grad()

        sentence_embeddings = batch['sentence_embeddings'].to(device)
        summary_mask = batch['summary_mask'].to(device) # This is x_0
        attention_mask = batch['attention_mask'].to(device)
        
        B, L = summary_mask.shape

        # 1. Sample a random timestep t
        t = torch.randint(0, timesteps, (B,), device=device).long()

        # 2. Create noisy mask x_t
        noisy_mask = q_sample(summary_mask, t, alphas_cumprod)

        # 3. Get model prediction
        predicted_logits = model(sentence_embeddings, noisy_mask, t, attention_mask)

        # 4. Calculate loss
        loss = F.cross_entropy(predicted_logits.view(-1, 2), summary_mask.view(-1), reduction='none')
        loss = loss.view(B, L)
        loss = (loss * attention_mask).sum() / attention_mask.sum()

        # 5. Backpropagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.trainer.gradient_clip_val)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, alphas_cumprod, device, config):
    """Runs a single validation epoch."""
    model.eval()
    total_loss = 0.0
    timesteps = config.diffusion.timesteps

    with torch.no_grad():
        for batch in dataloader:
            sentence_embeddings = batch['sentence_embeddings'].to(device)
            summary_mask = batch['summary_mask'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            B, L = summary_mask.shape
            t = torch.randint(0, timesteps, (B,), device=device).long()
            noisy_mask = q_sample(summary_mask, t, alphas_cumprod)

            predicted_logits = model(sentence_embeddings, noisy_mask, t, attention_mask)

            loss = F.cross_entropy(predicted_logits.view(-1, 2), summary_mask.view(-1), reduction='none')
            loss = loss.view(B, L)
            loss = (loss * attention_mask).sum() / attention_mask.sum()

            total_loss += loss.item()

    return total_loss / len(dataloader)


@hydra.main(version_base=None, config_path='.', config_name='config')
def train(config: DictConfig):
    """Main training function."""
    LOGGER = utils.get_logger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    # --- Dataloaders ---
    LOGGER.info("Setting up dataloaders...")
    train_loader, valid_loader = get_summarization_dataloaders(config)

    # --- Model ---
    LOGGER.info("Setting up model...")
    model = SummarizationDenoiser(config).to(device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)

    # --- Diffusion Schedule ---
    timesteps = config.diffusion.timesteps
    betas = linear_beta_schedule(
        timesteps,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end
    ).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)

    # --- Checkpointing ---
    best_val_loss = float('inf')
    checkpoint_dir = os.getcwd() # Hydra sets the working directory for each run
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    LOGGER.info(f"Checkpoints will be saved in: {checkpoint_dir}")

    # --- Training Loop ---
    LOGGER.info("Starting training...")
    # Using max_steps from config, assuming 1 epoch is one pass through data
    num_epochs = config.trainer.max_steps // len(train_loader)

    for epoch in range(num_epochs):
        LOGGER.info(f"--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Wrap the dataloader with tqdm here for a clean progress bar
        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        train_loss = train_one_epoch(model, train_progress_bar, optimizer, alphas_cumprod, device, config)
        LOGGER.info(f"Average Training Loss: {train_loss:.4f}")

        # Do the same for validation
        valid_progress_bar = tqdm(valid_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
        val_loss = validate_one_epoch(model, valid_progress_bar, alphas_cumprod, device, config)
        LOGGER.info(f"Average Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            LOGGER.info(f"New best model saved to {checkpoint_path} with validation loss: {best_val_loss:.4f}")

    LOGGER.info("Training finished.")

if __name__ == '__main__':
    # This will be executed when you run `python train_summarizer.py`
    # Hydra will manage the configuration.
    train()