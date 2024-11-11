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
        device = None,
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

        self.num_message_passing_layers = num_message_passing_layers
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_rbf_features = num_rbf_features
        self.num_unique_atoms = num_unique_atoms
        self.cutoff_dist = cutoff_dist
        self.device = device

        self.embedding = nn.Embedding(self.num_unique_atoms, self.num_features,
                                      device=self.device)
        
        self.message_layers = nn.ModuleList([
            Message(self.num_features, self.num_rbf_features, 
                    self.cutoff_dist, self.device)
            for _ in range(self.num_message_passing_layers)
        ])

        self.update_layers = nn.ModuleList([
            Update(self.num_features, self.num_rbf_features,
                   self.cutoff_dist, self.device)
            for _ in range(self.num_message_passing_layers)
        ])

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

        N_i = len(atoms)

        # Make adjecency matrix
        A_row, A_col = self.make_adjecency_matrix(atom_positions, graph_indexes)

        N_j = len(A_col)

        # Find relative positions
        r_ij = atom_positions[A_col] - atom_positions[A_row]

        # Get Z and embeddings
        Z_i = atoms
        Z_j = atoms[A_col]

        s_i = self.embedding(Z_i)
        s_j = self.embedding(Z_j)

        v_i = torch.zeros((N_i, self.num_features, 3), device=self.device)
        v_j = torch.zeros((N_j, self.num_features, 3), device=self.device)

        # Perform message passing
        for i in range(self.num_message_passing_layers):
            delta_vi, delta_si = self.message_layers[i](v_j, s_j, r_ij, N_i, A_row) #TODO: Could insert N_i and A_row into message in another way
            v_i += delta_vi
            s_i += delta_si
            delta_vi, delta_si = self.update_layers[i](v_i, s_i, N_i)
            v_i += delta_vi
            s_i += delta_si
        
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

class Message(nn.Module):   
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
        rbf_output = self.radial_basis_functions(r_ij, r_ij_norm)
        rbf_output = self.rbf_linear(rbf_output)
        W = self.cosine_cutoff(rbf_output, r_ij_norm)

        # Multiply before split
        conv_product = phi*W

        # Split values
        split_vj = conv_product[:,0:self.num_features].unsqueeze(2)  
        split_rj = conv_product[:,self.num_features:self.num_features*2].unsqueeze(2)
        delta_sjm = conv_product[:,self.num_features*2:self.num_features*3]

        # Compute normalised positions
        rj_norm = r_ij/r_ij_norm.unsqueeze(1)
        rj_norm = rj_norm.unsqueeze(1)

        # Compute temporary value for v_im
        tmp_vim = v_j*split_vj + split_rj*rj_norm

        delta_vim = torch.zeros((N_i, self.num_features, 3), device=self.device)
        delta_sim = torch.zeros((N_i, self.num_features), device=self.device)
        
        delta_vim.index_add_(0, A_row, tmp_vim)
        delta_sim.index_add_(0, A_row, delta_sjm)

        return delta_vim, delta_sim
    
    def radial_basis_functions(self, r_ij, r_ij_norm):
        Nrbf = 20
        rbf_output = torch.empty((len(r_ij), Nrbf), device=self.device)
        for n in range(1, Nrbf+1):
            rbf_output[:, n-1] = torch.sin(n*torch.pi/self.cutoff_dist*r_ij_norm)/r_ij_norm
        
        return rbf_output

    def cosine_cutoff(self, rbf_output, r_ij_norm):
        fc = 0.5*(torch.cos(torch.pi*r_ij_norm/self.cutoff_dist) + 1)
        return fc.unsqueeze(1)*rbf_output
    
class Update(nn.Module):   
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

        v_i = v_i.permute([0, 2, 1])
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
#class Update():
#        raise NotImplementedError

    

        
    # def gated_equivariant_block():
    #     raise NotImplementedError