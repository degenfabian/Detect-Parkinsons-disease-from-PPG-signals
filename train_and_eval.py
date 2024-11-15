import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from preprocess_data import get_dataloaders, PPGDataset
from model import Heart_GPT_FineTune, NewHead
from metrics import BinaryClassificationMetrics

"""
The values for the Config class and large parts of the main function were taken from: 
https://github.com/harryjdavies/HeartGPT/blob/main/HeartGPT_finetuning.py

Check out the paper by Davies et al. (2024) "Interpretable Pre-Trained Transformers for Heart Time-Series Data" 
available at https://arxiv.org/abs/2407.20775 to find out more about their amazing work!
"""


class Config:
    """
    Configuration class holding all hyperparameters and settings for the model.

    Attributes:
        save_interval (int): Number of epochs between checkpoints
        epochs (int): Total number of training epochs
        prediction_threshold (float): Threshold for binary classification
        early_stopping_threshold (int): Number of epochs without improvement before stopping
        batch_size (int): Number of samples per batch
        num_workers (int): Number of DataLoader workers
        pin_memory (bool): Whether to pin memory in DataLoader
        persistent_workers (bool): Whether to keep DataLoader workers alive
        use_amp (bool): Whether to use automatic mixed precision
        block_size (int): Size of transformer blocks
        n_embd (int): Embedding dimension
        n_head (int): Number of attention heads
        n_layer (int): Number of transformer layers
        dropout (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
        vocab_size (int): Size of the vocabulary
        device (str): Device to use for training ('cuda', 'mps', or 'cpu')
        model_path (str): Path to save/load model weights
        data_path (str): Path to data directory
    """

    save_interval = 10
    epochs = 1000
    prediction_threshold = 0.5
    early_stopping_threshold = 10
    batch_size = 1024
    num_workers = 4
    pin_memory = True
    persistent_workers = True
    use_amp = False
    block_size = 500
    n_embd = 64
    n_head = 8
    n_layer = 8
    dropout = 0.2
    learning_rate = 3e-04
    vocab_size = 102
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model_path = "model_weights/"
    data_path = "data/"


def evaluate(cfg, model, data_loader, metrics):
    """
    Evaluates model performance on a given dataset.

    Args:
        cfg (Config): Configuration object containing model settings
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): DataLoader containing the evaluation data
        metrics (BinaryClassificationMetrics): Metrics tracker object

    Note:
        Updates the metrics object in-place and prints the evaluation results.
    """

    model = model.to(cfg.device)
    model.eval()

    with torch.no_grad():
        for ppg, label in tqdm(data_loader, desc="Evaluating", position=0, leave=True):
            ppg = ppg.to(cfg.device)
            label = label.to(cfg.device)
            output = model(ppg)

            # Apply sigmoid before thresholding for predictions
            prediction = (nn.functional.sigmoid(output) > cfg.prediction_threshold).float()
            metrics.update_metrics(prediction, label)

    print(metrics.__str__())


def save_checkpoint(cfg, model, optimizer, epoch, scaler, checkpoint_name):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        cfg (Config): Configuration object containing model settings and paths
        model (nn.Module): The model to save
        optimizer (Optimizer): Optimizer for updating model parameters
        epoch (int): Current epoch
        scaler (GradScaler): AMP GradScaler for mixed precision training
        checkpoint_name (str): Name of the checkpoint file to save (e.g., "checkpoint_best.pt")
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if cfg.use_amp:
        checkpoint["scaler"] = scaler.state_dict()

    os.makedirs(cfg.model_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(cfg.model_path, checkpoint_name))


def train(
    cfg,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader=None,
    checkpoint=None,
):
    """
    Trains the model using the specified configuration and data.

    Args:
        cfg (Config): Configuration object containing training settings
        model (nn.Module): The model to train
        optimizer (Optimizer): Optimizer for updating model parameters
        criterion (nn.Module): Loss function
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader, optional): DataLoader for validation data
        checkpoint (dict, optional): Checkpoint to resume training from

    Notes:
        - Implements early stopping based on validation accuracy
        - Saves checkpoints periodically and when best performance is achieved
        - Uses mixed precision training when cfg.use_amp is True
    """

    model = model.to(cfg.device)
    scaler = torch.amp.GradScaler()
    best_val_acc = 0
    early_stopping = 0
    start_epoch = 0

    # Resume from checkpoint if provided
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

        if cfg.use_amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
            print("Restored AMP scaler state from checkpoint")

        print(f"Resuming training from epoch {start_epoch}")

    # Main training loop
    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Training", position=0):
        model.train()
        training_loss = 0.0
        train_metrics = BinaryClassificationMetrics(0, 0, 0, 0)

        # Batch training loop
        for ppg, label in tqdm(
            train_loader, desc=f"Epoch {epoch}", position=0, leave=True
        ):
            ppg = ppg.to(cfg.device)
            label = label.to(cfg.device)

            optimizer.zero_grad()
            with torch.autocast(device_type=cfg.device, dtype=torch.float16, enabled=cfg.use_amp):
                output = model(ppg)
                label = label.to(torch.float32)
                loss = criterion(output, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            training_loss += loss.item()
            # Apply sigmoid before thresholding for predictions
            prediction = (nn.functional.sigmoid(output) > cfg.prediction_threshold).float()
            train_metrics.update_metrics(prediction, label)

        avg_loss = training_loss / len(train_loader)

        # Print training metrics
        print(f"Training metrics for epoch {epoch}: \n")
        print(f"Loss: {avg_loss}")
        print(train_metrics.__str__())

        # Validation phase
        if val_loader is not None:
            val_metrics = BinaryClassificationMetrics(0, 0, 0, 0)

            # Print validation metrics
            print("Validation metrics: \n")
            evaluate(cfg, model, val_loader, val_metrics)

            # Save best model
            if val_metrics.accuracy() > best_val_acc:
                best_val_acc = val_metrics.accuracy()
                save_checkpoint(cfg, model, optimizer, epoch, scaler, "checkpoint_best.pt")
                print(f"Best model saved at epoch {epoch}")
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping >= cfg.early_stopping_threshold:
                    print(f"Stopping training at epoch {epoch} due to early stopping")
                    break

        # Save checkpoint every cfg.save_interval epochs
        if epoch % cfg.save_interval == cfg.save_interval - 1:
            save_checkpoint(
                cfg, model, optimizer, epoch, scaler, f"checkpoint_epoch_{epoch}.pt"
            )
            print(f"Checkpoint saved for epoch {epoch}")


def test(cfg, model, test_loader):
    """
    Tests the model using the best checkpoint from training.

    Args:
        cfg (Config): Configuration object containing model settings
        model (nn.Module): The model to test
        test_loader (DataLoader): DataLoader containing test data
    """

    checkpoint = torch.load(os.path.join(cfg.model_path, "checkpoint_best.pt"))
    model.load_state_dict(checkpoint["model"])
    test_metrics = BinaryClassificationMetrics(0, 0, 0, 0)

    # Print test metrics
    print("Testing metrics: \n")
    evaluate(cfg, model, test_loader, test_metrics)


def main():
    """
    Main function that handles the model fine-tuning process:
    1. Loads pretrained model
    2. Freezes all layers except the last block and new head
    3. Trains the model
    4. Saves the fine-tuned model
    5. Tests the performance of fine-tuned model
    """

    cfg = Config()
    model = Heart_GPT_FineTune(cfg).to(cfg.device)

    # Use cuDNN benchmark for faster training if using CUDA
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {cfg.device}")

    model.load_state_dict(
        torch.load(
            os.path.join(cfg.model_path, "PPGPT_500k_iters.pth"),
            map_location=cfg.device,
            weights_only=True,
        )
    )

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    model.lm_head = NewHead(cfg)

    # Replace and unfreeze head
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Unfreeze layer normalization
    for param in model.ln_f.parameters():
        param.requires_grad = True

    last_block = model.blocks[-1]

    # Unfreeze last transformer block
    for param in last_block.parameters():
        param.requires_grad = True

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    train(
        cfg,
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
    )
    torch.save(model.state_dict(), os.path.join(cfg.model_path, "PPGPT_finetuned.pth"))

    test(cfg, model, test_loader)


if __name__ == "__main__":
    main()
