# Extractive Summarization with Discrete Diffusion

This project trains and runs a discrete diffusion model for extractive summarization. The model learns to select the most important sentences from an article to form a summary.

## 1. Setup

This project uses Conda for environment management to handle PyTorch and CUDA dependencies smoothly.

First, ensure you have Miniconda or Anaconda installed. Then, create and activate the environment using the provided `environment.yml` file.

```bash
conda env create -f environment.yml
conda activate denoise
```

The first time you run the code, it will automatically download the necessary `punkt` tokenizer from NLTK.

## 2. Configuration

All project settings are managed by Hydra via the `config.yaml` file. Before running, you may want to review or modify key parameters:

- **`data`**: Specifies the dataset to use (e.g., `cnn_dailymail_extractive`).
- **`model`**: Defines the architecture of the denoising model (e.g., `dim`, `num_heads`, `num_layers`).
- **`summarization`**: Contains settings specific to this task, like `max_sentences` and the sentence embedding model to use.
- **`trainer`**: Controls the training process, including `max_steps` and `gradient_clip_val`.
- **`optim`**: Sets the optimizer parameters like learning rate (`lr`).
- **`eval.checkpoint_path`**: This path must point to a trained model checkpoint.

## 3. Training the Model

The training process teaches the model to identify important summary sentences.

### How to Run

To start training, simply run the `train_summarizer.py` script from the project's root directory:

```bash
python train_summarizer.py
```

We also have `train_mask.py` and `train_focal.py` for masked diffusion and focal loss formulation

### Overriding Configuration

You can easily override any configuration parameter from the command line for quick experiments:

```bash
# Train with a different learning rate and for fewer steps
python train_summarizer.py optim.lr=1e-5 trainer.max_steps=50000
```

## 4. Generating a Summary (Inference)

Once you have a trained model, you can test it with `evaluate_model.py`, or if the model was trained with `train_mask.py`, then you would evaluate it with `eval_mask.py`. Please make sure to set the `eval.checkpoint_path` to the location of your `best_model.pt` file, as well as set the model hyperparameters to be that of the ones used during training. 