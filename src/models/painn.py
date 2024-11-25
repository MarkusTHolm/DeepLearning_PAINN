import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import logging
from omegaconf import DictConfig
import hydra
import time
from torch_geometric.nn import radius_graph

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
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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

        self.logger = logging.getLogger(__name__)

        # Initialize inputs
        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist
        self.device = device

        # Initialize embedding
        self.embedding = nn.Embedding(self.num_unique_atoms, self.num_features,
                                      device=self.device)
        # Intialize message layers
        self.message_layers = nn.ModuleList([
            Message(self.num_features, self.num_rbf_features, 
                    self.cutoff_dist, self.device)
            for _ in range(self.num_message_passing_layers)
        ])

        # Initialize update layers
        self.update_layers = nn.ModuleList([
            Update(self.num_features, self.num_rbf_features,
                   self.cutoff_dist, self.device)
            for _ in range(self.num_message_passing_layers)
        ])

        # Initialize final reduction layers
        self.final_reduction = nn.Sequential(
            nn.Linear(self.num_features, self.num_features//2, bias=True),
            nn.SiLU(),
            nn.Linear(self.num_features//2, self.num_outputs, bias=True)
        )

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

        # Make adjecency matrix
        A_row, A_col = self.make_adjecency_matrix_2(atom_positions, graph_indexes)

        start = time.time()

        # Dimensions of nodes and edges
        N_i = len(atoms)
        N_j = len(A_col)

        # Find relative positions
        r_ij = atom_positions[A_col] - atom_positions[A_row]

        # Get Z and embeddings
        Z_i = atoms
        Z_j = atoms[A_col]

        s_i = self.embedding(Z_i)
        s_j = self.embedding(Z_j)

        v_i = torch.zeros((N_i, 3, self.num_features), device=self.device)
        v_j = torch.zeros((N_j, 3, self.num_features), device=self.device)

        # Perform message passing
        for i in range(self.num_message_passing_layers):
            # Message step
            delta_vi, delta_si = self.message_layers[i](v_j, s_j, r_ij, N_i, A_row)
            v_i = v_i + delta_vi
            s_i = s_i + delta_si
            # Update step
            delta_vi, delta_si = self.update_layers[i](v_i, s_i, N_i)
            v_i = v_i + delta_vi
            s_i = s_i + delta_si
        
        # Final reduction block
        atomic_contributions = self.final_reduction(s_i)

        self.logger.debug(f"Time spent in rest: {time.time() - start})")

        return atomic_contributions
    
    def make_adjecency_matrix(self, atom_positions, graph_indexes):
        """ Create adjecency matrix """
        
        start = time.time()

        N_i = len(graph_indexes)
        A_col = []
        A_row = []

        indices = torch.arange(N_i)
        graph_indexes = graph_indexes.cpu()
        atom_positions = atom_positions.cpu()

        for i, gi in enumerate(graph_indexes):
            matching_graph_index = graph_indexes == gi
            matching_graph_index[i] = False   # Remove the current index from list (to avoid loops in graph)
            pos_same_molecule = atom_positions[matching_graph_index]
            dist_mask = torch.norm(pos_same_molecule - atom_positions[i], 
                                   p=2, dim=1) < self.cutoff_dist
            A_row.append(torch.Tensor.tile(torch.IntTensor([i]), int(dist_mask.sum()) ))
            A_col.append(indices[matching_graph_index][dist_mask])

        A_row = torch.cat(A_row)
        A_col = torch.cat(A_col)

        A_row = A_row.to(device=self.device)
        A_col = A_col.to(device=self.device)

        self.logger.debug(f"Time spent in adjecency matrix: {time.time() - start})")

        return A_row, A_col

    def make_adjecency_matrix_2(self, atom_positions, graph_indexes):
        """ Create adjecency matrix """
        
        start = time.time()
        epsilon = 100

        # Offset positions by graph-index times an epsilon to distinguish molecules
        r = atom_positions + (graph_indexes.unsqueeze(-1))*epsilon

        # Subtract all combinations of r
        r1 = r.unsqueeze(1)
        r2 = r.unsqueeze(0)
        diff = r1 - r2

        # Compute distance (Euclidian norm)
        dist = torch.norm(diff, dim=2)

        # Remove diagonals and obtain (full adjacency matrix)
        dist.fill_diagonal_(epsilon)
        A = (dist < self.cutoff_dist)._to_sparse()

        # Get adjecency matrix components from sparse format
        A_indices = A.indices()
        A_row = A_indices[0, :]
        A_col = A_indices[1, :]

        self.logger.debug(f"Time spent in adjecency matrix 2: {time.time() - start})")

        return A_row, A_col

class Message(nn.Module):   
    """ Message class for PaiNN """
    def __init__(self,
                num_features, 
                num_rbf_features,
                cutoff_dist,
                device) -> None:
        super(Message, self).__init__()
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features
        self.cutoff_dist = cutoff_dist
        self.device = device

        # For computing phi
        self.sj_linear = nn.Sequential(
            nn.Linear(self.num_features, self.num_features, bias=True),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_features*3, bias=True)
        )

        self.rbf_linear = nn.Linear(self.num_rbf_features, self.num_features*3)

    def forward(self, v_j, s_j, r_ij, N_i, A_row):

        # Compute distance between atoms 
        r_ij_norm = torch.norm(r_ij, dim=1)

        # Compute phi
        phi = self.sj_linear(s_j)

        # Compute W
        rbf_output = self.radial_basis_functions(r_ij_norm)
        rbf_output = self.rbf_linear(rbf_output)
        W = self.cosine_cutoff(rbf_output, r_ij_norm)

        # Multiply before split
        conv_product = phi*W

        # Split values
        split_vj = conv_product[:,0:self.num_features].unsqueeze(1)  
        split_rj = conv_product[:,self.num_features:self.num_features*2].unsqueeze(1)
        delta_sjm = conv_product[:,self.num_features*2:self.num_features*3]

        # Compute normalised positions
        rj_norm = r_ij/r_ij_norm.unsqueeze(1)
        rj_norm = rj_norm.unsqueeze(2)

        # Compute temporary value for v_im
        tmp_vim = v_j*split_vj + split_rj*rj_norm

        delta_vim = torch.zeros((N_i, 3, self.num_features), device=self.device)
        delta_sim = torch.zeros((N_i, self.num_features), device=self.device)
        
        delta_vim.index_add_(0, A_row, tmp_vim)
        delta_sim.index_add_(0, A_row, delta_sjm)

        return delta_vim, delta_sim
    
    def radial_basis_functions(self, r_ij_norm):
        ns = torch.arange(1, self.num_rbf_features+1, device=self.device).unsqueeze(0)
        r_ij_unsqueeze = r_ij_norm.unsqueeze(1)
        rbf_output = torch.sin(ns*torch.pi/self.cutoff_dist*r_ij_unsqueeze)/r_ij_unsqueeze        
        return rbf_output

    def cosine_cutoff(self, rbf_output, r_ij_norm):
        fc = 0.5*(torch.cos(torch.pi*r_ij_norm/self.cutoff_dist) + 1)
        return fc.unsqueeze(1)*rbf_output
    
class Update(nn.Module):   
    """ Update class for PaiNN """
    def __init__(self,
                num_features, 
                num_rbf_features,
                cutoff_dist,
                device) -> None:
        super(Update, self).__init__()
        self.num_features = num_features
        self.num_rbf_features = num_rbf_features
        self.cutoff_dist = cutoff_dist
        self.device = device

        self.linear_nobias_U = nn.Linear(self.num_features, self.num_features, bias=False)
        self.linear_nobias_V = nn.Linear(self.num_features, self.num_features, bias=False)
        
        self.sj_linear = nn.Sequential(
            nn.Linear(self.num_features*2, self.num_features, bias=True),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_features*3, bias=True)
        )

    def forward(self, v_i, s_i, N_i):

        tmp_U_vi = self.linear_nobias_U(v_i)
        tmp_V_vi = self.linear_nobias_V(v_i)

        s_i_stack = torch.empty((N_i, self.num_features*2), device=self.device)
        s_i_stack[:, 0:self.num_features] = s_i
        s_i_stack[:, self.num_features:self.num_features*2] = torch.norm(tmp_V_vi, dim=1)

        a = self.sj_linear(s_i_stack)

        a_vv = a[:, 0:self.num_features]
        a_sv = a[:, self.num_features:self.num_features*2]
        a_ss = a[:, self.num_features*2:self.num_features*3]

        tmp_scalar_prod = torch.sum(tmp_U_vi*tmp_V_vi, dim=1)

        delta_viu = tmp_U_vi*a_vv.unsqueeze(1)
        delta_siu = tmp_scalar_prod*a_sv + a_ss

        return delta_viu, delta_siu