# Extractive Summarization with Discrete Diffusion

This project trains and runs a discrete diffusion model for extractive summarization. The model learns to select the most important sentences from an article to form a summary.

## 1. Setup

This project uses Conda for environment management to handle PyTorch and CUDA dependencies smoothly.

First, ensure you have Miniconda or Anaconda installed. Then, create and activate the environment using the provided `environment.yml` file.

```bash
# Create the conda environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate mdlm
```

The first time you run the code, it will automatically download the necessary `punkt` tokenizer from NLTK.

## 2. Configuration

All project settings are managed by Hydra via the `configs/config.yaml` file. Before running, you may want to review or modify key parameters:

- **`data`**: Specifies the dataset to use (e.g., `cnn_dailymail`).
- **`model`**: Defines the architecture of the denoising model (e.g., `dim`, `num_heads`, `num_layers`).
- **`summarization`**: Contains settings specific to this task, like `max_sentences` and the sentence embedding model to use.
- **`trainer`**: Controls the training process, including `max_steps` and `gradient_clip_val`.
- **`optim`**: Sets the optimizer parameters like learning rate (`lr`).
- **`eval.checkpoint_path`**: **Crucial for inference.** This path must point to a trained model checkpoint.

## 3. Training the Model

The training process teaches the model to identify important summary sentences.

### How to Run

To start training, simply run the `train_summarizer.py` script from the project's root directory:

```bash
python train_summarizer.py
```

### What Happens

1.  Hydra loads the configuration from `configs/config.yaml`.
2.  The `SummarizationDataset` prepares the `cnn_dailymail` dataset, creating sentence embeddings and oracle summary masks.
3.  The model trains for the number of epochs derived from `trainer.max_steps`.
4.  After each epoch, the model's performance is evaluated on the validation set.
5.  If the validation loss improves, a checkpoint of the model (`best_model.pt`) is saved into a unique output directory created by Hydra. The path will look like: `outputs/cnn_dailymail/YYYY.MM.DD/HHMMSS/`.

### Overriding Configuration

You can easily override any configuration parameter from the command line for quick experiments:

```bash
# Train with a different learning rate and for fewer steps
python train_summarizer.py optim.lr=1e-5 trainer.max_steps=50000
```

## 4. Generating a Summary (Inference)

Once you have a trained model, you can use it to generate summaries for new articles.

### How to Run

1.  **Update the Checkpoint Path**: Open `configs/config.yaml` and set the `eval.checkpoint_path` to the location of your `best_model.pt` file.

    ```yaml
    # In configs/config.yaml
    eval:
      checkpoint_path: 'outputs/cnn_dailymail/2023.10.27/103000/best_model.pt' # <-- Update this path
    ```

2.  **Run the Generation Script**:

    ```bash
    python generate_summary.py
    ```

### What Happens

1.  The script loads your trained model from the specified checkpoint.
2.  It processes the example article defined inside `generate_summary.py`.
3.  It performs the full reverse diffusion process (sampling) to generate a binary mask indicating which sentences to include in the summary.
4.  The final extractive summary is printed to the console.

You can edit the `article_text` variable inside `generate_summary.py` to summarize your own text.



I do not have experience writing and training diffusion models. i want to train a discrete diffusion denoising model for extractive summarization (where summary is constructed as selecting sentences from the source article), where the clean data is the summary, and the noisy data is the source text, and the sentences not in the extractive summary are the "noise". How would I approach this, both idea wise and code wise? I have went ahead and made some changed to dataloader.