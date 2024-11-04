import torch
import torch.nn as nn


class PaiNN(nn.Module):
    """
    Polarizable Atom Interaction Neural Network with PyTorch.
    """
    def __init__(
        self,
        num_message_passing_layers: int = 3,
        num_features: int = 128,
        num_outputs: int = 1,
        num_rbf_features: int = 20,
        num_unique_atoms: int = 100,
        cutoff_dist: float = 5.0,
    ) -> None:
        """
        Args:
            num_message_passing_layers: Number of message passing layers in
                the PaiNN model.
            num_features: Size of the node embeddings (scalar features) and
                vector features.
            num_outputs: Number of model outputs. In most cases 1.
            num_rbf_features: Number of radial basis functions to represent
                distances.
            num_unique_atoms: Number of unique atoms in the data that we want
                to learn embeddings for.
            cutoff_dist: Euclidean distance threshold for determining whether 
                two nodes (atoms) are neighbours.
        """
        super().__init__()
        raise NotImplementedError


    def forward(
        self,
        atoms: torch.LongTensor,
        atom_positions: torch.FloatTensor,
        graph_indexes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of PaiNN. Includes the readout network highlighted in blue
        in Figure 2 in (Schütt et al., 2021) with normal linear layers which is
        used for predicting properties as sums of atomic contributions. The
        post-processing and final sum is perfomed with
        src.models.AtomwisePostProcessing.

        Args:
            atoms: torch.LongTensor of size [num_nodes] with atom type of each
                node in the graph.
            atom_positions: torch.FloatTensor of size [num_nodes, 3] with
                euclidean coordinates of each node / atom.
            graph_indexes: torch.LongTensor of size [num_nodes] with the graph 
                index each node belongs to.

        Returns:
            A torch.FloatTensor of size [num_nodes, num_outputs] with atomic
            contributions to the overall molecular property prediction.
        """
        raise NotImplementedError

class Message():   
    def __init__(self):
        pass

    def foward(self):
        # For computing phi
        self.sj_linear = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_features*3)
        )

        # Compute phi
        phi = self.sj_linear()

        # Compute W
        self.radial_basis_function(),
        self.rbf_linear = nn.Linear(self.num_rbf_features, self.num_features*3)
        W = self.cosine_cutoff()

        # Compute normalised positions
        rj_norm = rj/torch.norm(rj)

        # Multiply before split
        pre_split = phi*W

        # Split values
        split_vj = pre_split[0:self.num_features]  
        split_rj = pre_split[0:self.num_features]
        delta_sim = pre_split[0:self.num_features]

        delta_vim = torch.sum(vj*split_vj + split_rj*rj_norm)
       
        return delta_sim, delta_vim
    
    def update():
        raise NotImplementedError
    
    def gated_equivariant_block():
        raise NotImplementedError
    
    def radial_basis_function():
        raise NotImplementedError
    
    def cosine_cutoff():
        raise NotImplementedError