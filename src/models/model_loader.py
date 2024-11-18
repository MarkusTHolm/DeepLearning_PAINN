import torch
from src.models import PaiNN

def save_checkpoint(model, optimizer, epoch, seed, config, filepath="model_checkpoint.pth"):
    """
    Saves the model, optimizer state, and metadata.

    Args:
    - model (torch.nn.Module): Trained model to save.
    - optimizer (torch.optim.Optimizer): Optimizer used during training.
    - epoch (int): Current epoch number.
    - seed (int): Random seed used for reproducibility.
    - config (dict): Configuration or metadata (e.g., batch size, learning rate).
    - filepath (str): Path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'seed': seed,
        'config': config,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved at {filepath}")
    
    
def load_checkpoint(filepath, optimizer=None):
    """
    Loads the model, optimizer state, and metadata.

    Args:
    - filepath (str): Path to the checkpoint file.
    - model (torch.nn.Module): Model instance to load the weights into.
    - optimizer (torch.optim.Optimizer, optional): Optimizer instance to load the state into.
    
    Returns:
    - dict: Metadata including epoch, seed, and config.
    """
    checkpoint = torch.load(filepath)
    
    # Instatiate painn model
    model_config = checkpoint['config']
    model = PaiNN(**model_config)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state (optional)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return metadata
    metadata = {
        'epoch': checkpoint['epoch'],
        'seed': checkpoint['seed'],
        'config': checkpoint['config'],
    }
    print(f"Checkpoint loaded from {filepath}")
    
    return model, metadata