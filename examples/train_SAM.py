"""
Basic example of how to train the PaiNN model to predict the QM9 property
"internal energy at 0K". This property (and the majority of the other QM9
properties) is computed as a sum of atomic contributions.
"""
import sys, os
# Add the root directory of your project to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
from tqdm import trange
import torch.nn.functional as F
from src.data import QM9DataModule
from pytorch_lightning import seed_everything
from src.models import PaiNN, AtomwisePostProcessing
from src.models import model_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import hydra
import pickle
import os
from sam.sam import SAM


import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

@hydra.main(config_path=f'./conf',
            config_name='config.yaml',
            version_base='1.1')     
def main(cfg):
    cfg = cfg.experiment
    seed_everything(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    print(f"Working directory  : {os.getcwd()}")

    dm = QM9DataModule(
        target=cfg.data.target,
        data_dir=cfg.data.data_dir,
        batch_size_train=cfg.data.batch_size_train,
        batch_size_inference=cfg.data.batch_size_inference,
        num_workers=cfg.data.num_workers,
        splits=cfg.data.splits,
        seed=cfg.seed,
        subset_size=cfg.data.subset_size,
    )
    dm.prepare_data()
    dm.setup()
    y_mean, y_std, atom_refs = dm.get_target_stats(
        remove_atom_refs=True, divide_by_atoms=True
    )

    model_config = {'num_message_passing_layers': cfg.model.num_message_passing_layers,
                    'num_features': cfg.model.num_features,
                    'num_outputs': cfg.model.num_outputs, 
                    'num_rbf_features': cfg.model.num_rbf_features,
                    'num_unique_atoms': cfg.model.num_unique_atoms,
                    'cutoff_dist': cfg.model.cutoff_dist,
                    'device': device
                    }
    
    painn = PaiNN(**model_config)
    
    post_processing = AtomwisePostProcessing(
        cfg.model.num_outputs, y_mean, y_std, atom_refs
    )

    painn.to(device)
    post_processing.to(device)

    # Define the base and SAM optimizer
    base_optimizer = torch.optim.AdamW
    optimizer = SAM(
        painn.parameters(),
        base_optimizer,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        rho = cfg.training.rho,
        adaptive=cfg.training.sam_adaptive
    )
    
    # Scheduler for learning rate decay
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=cfg.training.decay_factor,
                                  patience=cfg.training.decay_patience)
    
    unit_conversion = dm.unit_conversion[cfg.data.target]
    
    # File to save losses
    data_file = f"{cfg.data.results_dir}/data.pickle"
    if os.path.exists(data_file):
        # Load existing logs if they exist
        with open(data_file, 'rb') as f:
            logs = pickle.load(f)
    else:
        logs = {'train_loss': [],
                'val_loss': [],
                'train_MAE': [],
                'val_MAE': [],
                'lr': [],
                'epoch': [],
                'test_MAE': [],
                'val_loss_smoothed': []}  # Initialize logs
    

    # Initialize
    pbar = trange(cfg.training.num_epochs)
    early_stop_counter = 0
    best_val_loss = float('inf')
    smoothed_val_loss = None  # This will hold the smoothed value across epochs
    
    try:
        for epoch in pbar:
            
            # Save current
            logs['lr'].append(scheduler.get_last_lr()[0])
            logs['epoch'].append(epoch)
            
            # Training
            painn.train()
            loss_epoch = 0.
            MAE_epoch = 0.
            for batch in dm.train_dataloader():
                batch = batch.to(device)
                
                # Define closure for SAM optimizer
                def closure():
                    atomic_contributions = painn(
                    atoms=batch.z,
                    atom_positions=batch.pos,
                    graph_indexes=batch.batch
                    )
                    preds = post_processing(
                        atoms=batch.z,
                        graph_indexes=batch.batch,
                        atomic_contributions=atomic_contributions,
                    )
                    loss_step = F.mse_loss(preds, batch.y, reduction='sum')
                    loss = loss_step / len(batch.y)
                    loss.backward()
                    return loss
                
                atomic_contributions = painn(
                    atoms=batch.z,
                    atom_positions=batch.pos,
                    graph_indexes=batch.batch
                )
                preds = post_processing(
                    atoms=batch.z,
                    graph_indexes=batch.batch,
                    atomic_contributions=atomic_contributions,
                )
                
                loss_step = F.mse_loss(preds, batch.y, reduction='sum')
                loss = loss_step / len(batch.y)
                loss.backward()
                optimizer.step(closure) # Perform both both SAM steps
                optimizer.zero_grad()
                
                # accumulate loss and MAE
                loss_epoch += loss_step.detach().item()
                MAE_epoch += F.l1_loss(preds, batch.y, reduction='sum').detach().item()
            # calculate average (per epoch) loss and MAE
            loss_epoch /= len(dm.data_train)
            MAE_epoch /= len(dm.data_train)
            logs['train_loss'].append(loss_epoch)
            logs['train_MAE'].append(unit_conversion(MAE_epoch))
                    
            # Validation (on every epoch)
            painn.eval()
            val_loss_epoch = 0.
            val_MAE_epoch = 0.
            with torch.no_grad():
                for val_batch in dm.val_dataloader():
                    val_batch = val_batch.to(device)
                    atomic_contributions = painn(
                        atoms=val_batch.z,
                        atom_positions=val_batch.pos,
                        graph_indexes=val_batch.batch
                    )
                    preds = post_processing(
                        atoms=val_batch.z,
                        graph_indexes=val_batch.batch,
                        atomic_contributions=atomic_contributions,
                    )
                    val_loss_epoch += F.mse_loss(preds, val_batch.y, reduction='sum').detach().item()
                    val_MAE_epoch += F.l1_loss(preds, val_batch.y, reduction='sum').detach().item()
            val_loss_epoch /= len(dm.data_val)
            val_MAE_epoch /= len(dm.data_val)
            
            # Exponential smoothing on the validation loss
            if smoothed_val_loss is None:
                smoothed_val_loss = val_loss_epoch
            else:
                smoothed_val_loss = cfg.training.alpha * val_loss_epoch + (1 - cfg.training.alpha) * smoothed_val_loss
            
            # Save logs
            logs['val_loss'].append(val_loss_epoch)
            logs['val_loss_smoothed'].append(smoothed_val_loss)
            logs['val_MAE'].append(unit_conversion(val_MAE_epoch))
            
            
            # Save logs to file after every epoch
            with open(data_file, 'wb') as f:
                pickle.dump(logs, f)

            # Update learning rate scheduler
            scheduler.step(smoothed_val_loss)

            # Early stopping
            if smoothed_val_loss < best_val_loss:
                best_val_loss = smoothed_val_loss
                early_stop_counter = 0  # Reset counter
            else:
                early_stop_counter += 1

            if early_stop_counter >= cfg.training.early_stopping_patience:
                print("Early stopping triggered. Training stopped.")
                break

            # Progress update
            pbar.set_postfix_str(f'Train loss: {loss_epoch:.3e}, Val loss: {smoothed_val_loss:.3e}, lr: {scheduler.get_last_lr()[0]:.3e}')

    except Exception as e:
        # Save logs in case of an error
        with open(data_file, 'wb') as f:
            pickle.dump(logs, f)
        print(f"An error occurred: {e}. Logs have been saved.")

    # Always run the below code when training is finished (due to break, error or completion)
    finally:
        mae = 0
        painn.eval()
        with torch.no_grad():
            for batch in dm.test_dataloader():
                batch = batch.to(device)

                atomic_contributions = painn(
                    atoms=batch.z,
                    atom_positions=batch.pos,
                    graph_indexes=batch.batch,
                )
                preds = post_processing(
                    atoms=batch.z,
                    graph_indexes=batch.batch,
                    atomic_contributions=atomic_contributions,
                )
                mae += F.l1_loss(preds, batch.y, reduction='sum')
        
        mae /= len(dm.data_test)
        print(f'Test MAE: {unit_conversion(mae):.3f}')
        logs['test_MAE'].append(unit_conversion(mae).detach().item())
        
        # Save logs
        with open(data_file, 'wb') as f:
            pickle.dump(logs, f)
            
        # Save trained model
        model_loader.save_checkpoint(
            painn, optimizer, epoch, cfg.seed, cfg.data.target,  model_config, f"{cfg.data.results_dir}/model_checkpoint.pth"
        )
    
if __name__ == '__main__':
    main()