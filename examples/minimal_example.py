"""
Basic example of how to train the PaiNN model to predict the QM9 property
"internal energy at 0K". This property (and the majority of the other QM9
properties) is computed as a sum of atomic contributions.
"""
import sys, os
import torch
import argparse
from tqdm import trange
import torch.nn.functional as F
from src.data import QM9DataModule
from pytorch_lightning import seed_everything
from src.models import PaiNN, AtomwisePostProcessing

import hydra


@hydra.main(config_path=f'./conf',
            config_name='config.yaml',
            version_base='1.1')     
def main(cfg):
    cfg = cfg.experiment
    seed_everything(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # y_mean, y_std, atom_refs = dm.get_target_stats(
    #     remove_atom_refs=True, divide_by_atoms=True
    # )

    painn = PaiNN(
        num_message_passing_layers=cfg.model.num_message_passing_layers,
        num_features=cfg.model.num_features,
        num_outputs=cfg.model.num_outputs, 
        num_rbf_features=cfg.model.num_rbf_features,
        num_unique_atoms=cfg.model.num_unique_atoms,
        cutoff_dist=cfg.model.cutoff_dist,
        device=device,
    )
    # post_processing = AtomwisePostProcessing(
    #     args.num_outputs, y_mean, y_std, atom_refs
    # )

    painn.to(device)
    # post_processing.to(device)

    # optimizer = torch.optim.AdamW(
    #     painn.parameters(),
    #     lr=cfg.training.lr,
    #     weight_decay=cfg.training.weight_decay,
    # )

    painn.train()
    pbar = trange(cfg.training.num_epochs)
    for epoch in pbar:

        loss_epoch = 0.
        for batch in dm.train_dataloader():
            batch = batch.to(device)

            atomic_contributions = painn(
                atoms=batch.z,
                atom_positions=batch.pos,
                graph_indexes=batch.batch
            )
            sys.exit()
            preds = post_processing(
                atoms=batch.z,
                graph_indexes=batch.batch,
                atomic_contributions=atomic_contributions,
            )
            loss_step = F.mse_loss(preds, batch.y, reduction='sum')

            loss = loss_step / len(batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch += loss_step.detach().item()
        loss_epoch /= len(dm.data_train)
        pbar.set_postfix_str(f'Train loss: {loss_epoch:.3e}')

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
    unit_conversion = dm.unit_conversion[cfg.data.target]
    print(f'Test MAE: {unit_conversion(mae):.3f}')


if __name__ == '__main__':
    main()