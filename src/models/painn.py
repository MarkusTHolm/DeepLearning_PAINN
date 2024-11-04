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

        self.cutoff_dist = cutoff_dist
        self.num_message_passing_layers = num_message_passing_layers

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

        N = len(atoms)

        # Make adjecency matrix
        A_row, A_col = self.make_adjecency_matrix(atom_positions, graph_indexes)

        # Find relative positions
        r_ij = atom_positions[A_col] - atom_positions[A_row]

        # Get Z and embeddings
        Z = atoms[A_row]  #TODO: check this
        s_i0 = nn.Embedding()

        v_i = torch.zeros((self.num_features, N, 3)) # TODO: check sizes

        # Perform message passing
        for i in range(self.num_message_passing_layers):
            delta_vi, delta_si = Message(v_j, s_j, r_ij)
            v_j = delta_vi + v_j
            s_j = delta_si + s_j
            delta_vi, delta_si = Update(v_j, s_j)
            v_j = delta_vi + v_j
            s_j = delta_si + s_j
        
        # Final reduction block

        self.final_reduction = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.SiLU(),
            nn.Linear(self.num_features, self.num_features)
        )

        output = self.final_reduction(s_j)

        energy = output.sum()

        print("Hello")
    
    def make_adjecency_matrix(self, atom_positions, graph_indexes):

        N = len(graph_indexes)
        A_col = []
        A_row = []

        indices = torch.arange(N).cuda()

        for i, gi in enumerate(graph_indexes):
            matching_graph_index = graph_indexes == gi
            matching_graph_index[i] = False   # Remove the current index from list (to avoid loops)
            pos_same_molecule = atom_positions[matching_graph_index]
            dist_mask = torch.norm(pos_same_molecule - atom_positions[i], 
                                   p=2, dim=1) < self.cutoff_dist

            A_row.append(torch.Tensor.tile(torch.IntTensor([i]), int(dist_mask.sum()) ).cuda())
            A_col.append(indices[matching_graph_index][dist_mask])

        A_row = torch.cat(A_row)
        A_col = torch.cat(A_col)

        return A_row, A_col


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
    
class Update():
        raise NotImplementedError

    
# def radial_basis_function():
#     raise NotImplementedError

# def cosine_cutoff():
#     raise NotImplementedError

        
    # def gated_equivariant_block():
    #     raise NotImplementedError