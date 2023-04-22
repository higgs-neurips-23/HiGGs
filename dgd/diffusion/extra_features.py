import torch
# from torch_geometric.transforms import largest_connected_components
import os
# os.chdir("../")
# from dgd import utils
import utils


def get_n_nodes(node_mask):
    return torch.sum(node_mask, dim = -1)

class DummyExtraFeatures:
    def __init__(self):
        """ This class does not compute anything, just returns empty tensors."""

    def check_sparse_to_dense(self, noisy_data):
        # print(str(noisy_data['X_t'].layout))
        # quit()
        if str(noisy_data['X_t'].layout) == "torch.sparse_coo":
            self.return_sparse = True

        else:
            self.return_sparse = False

        return noisy_data

    def __call__(self, noisy_data):

        noisy_data = self.check_sparse_to_dense(noisy_data)

        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']
        empty_x = X.new_zeros((*X.shape[:-1], 0)).float()
        empty_e = E.new_zeros((*E.shape[:-1], 0)).float()
        empty_y = y.new_zeros((y.shape[0], 0)).float()



        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraFeatures:
    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.clusterer = Clustering()
        self.diameteriser = Diametering()
        self.com_features = CommunityFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ['eigenvalues', 'all', 'not-clustering']:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def check_sparse_to_dense(self, noisy_data):
        # print(str(noisy_data['X_t'].layout))
        # quit()
        if str(noisy_data['X_t'].layout) == "torch.sparse_coo":
            noisy_data['X_t'] = noisy_data['X_t'].to_dense()
            noisy_data['E_t'] = noisy_data['E_t'].to_dense()

            self.return_sparse = True

        else:
            self.return_sparse = False

        return noisy_data

    def dense_to_sparse(self, noisy_data):
        noisy_data['X_t'] = noisy_data['X_t'].to_sparse()
        noisy_data['E_t'] = noisy_data['E_t'].to_sparse()

        return noisy_data

    def __call__(self, noisy_data, pos=None):
        n = noisy_data['node_mask'].sum(dim=1).unsqueeze(1) / self.max_n_nodes

        noisy_data = self.check_sparse_to_dense(noisy_data)



        x_cycles, y_cycles = self.ncycles(noisy_data)       # (bs, n_cycles)
        # print(x_cycles)
        # print(x_cycles.shape)
        # print(y_cycles.shape)
        # quit()

        if self.features_type == 'cycles':
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)

            if self.return_sparse:
                y = torch.hstack((n, y_cycles)).to_sparse()
                x_cycles = x_cycles.to_sparse()
                extra_edge_attr = extra_edge_attr.to_sparse()
                noisy_data = self.dense_to_sparse(noisy_data)

            else:
                y = torch.hstack((n, y_cycles))

            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=y)

        elif self.features_type == 'communities':
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            # if pos is not None:
            x_coms = self.com_features(noisy_data, pos)

            x_coms = x_coms[:,:x_cycles.shape[1]]
            x_coms = x_coms[:,:, None]
            x = torch.cat((x_cycles, x_coms), dim=-1)
            # else:
            #     x = x_cycles

            if self.return_sparse:
                y = torch.hstack((n, y_cycles)).to_sparse()
                x_cycles = x_cycles.to_sparse()
                extra_edge_attr = extra_edge_attr.to_sparse()
                noisy_data = self.dense_to_sparse(noisy_data)

            else:
                y = torch.hstack((n, y_cycles))

            return utils.PlaceHolder(X=x, E=extra_edge_attr, y=y)


        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues = eigenfeatures   # (bs, 1), (bs, 10)

            if self.return_sparse:
                y = torch.hstack((n, y_cycles, n_components,
                                                batched_eigenvalues)).to_sparse()
                x_cycles = x_cycles.to_sparse()
                extra_edge_attr = extra_edge_attr.to_sparse()
                noisy_data = self.dense_to_sparse(noisy_data)

            else:
                y = torch.hstack((n, y_cycles, n_components,
                                                batched_eigenvalues))

            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=y)

        elif self.features_type == 'not-clustering':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures   # (bs, 1), (bs, 10),
                                                                                                # (bs, n, 1), (bs, n, 2)

            if self.return_sparse:
                y = torch.hstack((n, y_cycles, n_components, batched_eigenvalues)).to_sparse()
                x = torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1).to_sparse()
                extra_edge_attr = extra_edge_attr.to_sparse()

            else:
                # y = torch.hstack((n, y_cycles, n_components, batched_eigenvalues, global_clustering, diameter))
                y = torch.hstack((n, y_cycles, n_components, batched_eigenvalues))
                x = torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1)


            return utils.PlaceHolder(X=x,
                                     E=extra_edge_attr,
                                     y=y)
        elif self.features_type == 'all':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures   # (bs, 1), (bs, 10),
                                                                                                # (bs, n, 1), (bs, n, 2)


            local_clustering, global_clustering = self.clusterer.calc_clustering(noisy_data)
            # Removing clustering for now - very slow
            # diameter = self.diameteriser.calc_diameter(noisy_data)

            if self.return_sparse:
                y = torch.hstack((n, y_cycles, n_components, batched_eigenvalues)).to_sparse()
                x = torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1).to_sparse()
                extra_edge_attr = extra_edge_attr.to_sparse()

                noisy_data = self.dense_to_sparse(noisy_data)

            else:
                # y = torch.hstack((n, y_cycles, n_components, batched_eigenvalues, global_clustering, diameter))
                y = torch.hstack((n, y_cycles, n_components, batched_eigenvalues, global_clustering))
                x = torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec, local_clustering), dim=-1)


            return utils.PlaceHolder(X=x,
                                     E=extra_edge_attr,
                                     y=y)
        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class CommunityFeatures:
    def __call__(self, noisy_data, pos):
        bs = noisy_data['E_t'].shape[0]
        pos_dim = pos.shape
        pos = pos.view(bs, pos_dim[1], pos_dim[1])

        x_coms = pos[:,:,0].to(torch.int)
        return x_coms

class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = ReducedKNodeCycles()

    def __call__(self, noisy_data):
        adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)   # (bs, n_cycles)
        x_cycles = x_cycles.type_as(adj_matrix) * noisy_data['node_mask'].unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, mode):
        """ mode: 'eigenvalues' or 'all' or 'not-clustering' """
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = torch.linalg.eigvalsh(L)        # bs, n
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == 'all' or self.mode == "not-clustering":
            eigvals, eigvectors = torch.linalg.eigh(L)

            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                               node_mask=noisy_data['node_mask'],
                                                                               n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    n_connected_components[n_connected_components < 0] = 1e-10
    # assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    Warning: this function does not exactly return what is desired, the lcc might not be exactly the returned vector.
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    k0 = min(n, 5)
    first_evs = vectors[:, :, :k0]                         # bs, n, k0
    quantized = torch.round(first_evs * 1000) / 1000       # bs, n, k0
    random_mask = (50 * torch.ones((bs, n, k0)).type_as(vectors)) * (~node_mask.unsqueeze(-1))         # bs, n, k0
    min_batched = torch.min(quantized + random_mask, dim=1).values.unsqueeze(1)       # bs, 1, k0
    max_batched = torch.max(quantized - random_mask, dim=1).values.unsqueeze(1)       # bs, 1, k0
    nonzero_mask = quantized.abs() >= 1e-5
    is_min = (quantized == min_batched) * nonzero_mask * node_mask.unsqueeze(2)                      # bs, n, k0
    is_max = (quantized == max_batched) * nonzero_mask * node_mask.unsqueeze(2)                      # bs, n, k0
    is_other = (quantized != min_batched) * (quantized != max_batched) * nonzero_mask * node_mask.unsqueeze(2)

    all_masks = torch.cat((is_min.unsqueeze(-1), is_max.unsqueeze(-1), is_other.unsqueeze(-1)), dim=3)    # bs, n, k0, 3
    all_masks = all_masks.flatten(start_dim=-2)      # bs, n, k0 x 3
    counts = torch.sum(all_masks, dim=1)      # bs, k0 x 3

    argmax_counts = torch.argmax(counts, dim=1)       # bs
    lcc_indicator = all_masks[torch.arange(bs), :, argmax_counts]                   # bs, n
    not_lcc_indicator = ((~lcc_indicator).float() * node_mask).unsqueeze(2)

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class Clustering:
    # TODO: finish implementation of clustering - probably some bugs
    def __init__(self):
        super().__init__()

    def calc_clustering(self, noisy_data, verbose=False):
        adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()
        # adj_matrix = noisy_data['E_t'] # .sum(dim=-1).float()
        if verbose:
            print(f"Adj shape: {adj_matrix.shape}")


        self.adj_matrix = adj_matrix.float()

        self.A2 = self.adj_matrix @ self.adj_matrix
        self.A3 = self.A2 @ self.adj_matrix


        self.numerator   = batch_diagonal(self.A3)

        # self.degrees     = torch.sum(self.A2, axis = -1) - batch_diagonal(self.A2)
        self.degrees = batch_diagonal(self.A2)

        if verbose:
            print(f"Adj in noisy: {noisy_data['E_t']}")
            print(f"Adjacency: {self.adj_matrix}")
            print(f"A2: {self.A2}")
            print(f"Degrees: {self.degrees}")

        self.denominator = self.degrees * (self.degrees - 1)
        self.denominator[self.denominator <= 1] = int(1e16)

        self.clustering = self.numerator.float() / self.denominator.float()

        if verbose:
            print(f"Total graph clustering: {torch.sum(self.clustering, dim=-1).unsqueeze(-1).float()}")
            # print(f"Clustering shape: {self.clustering.shape}")
            # print(f"Node mask shape: {noisy_data['node_mask'].shape}")

        self.clustering = self.clustering.type_as(adj_matrix) * noisy_data['node_mask']

        # (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

        return self.clustering.unsqueeze(-1).float(), torch.mean(self.clustering, dim = -1).unsqueeze(-1) # torch.sum(self.clustering, dim=-1).unsqueeze(-1).float() /

class Diametering:
    # TODO: finish implementation of clustering - probably some bugs
    def __init__(self):
        super().__init__()

    def calc_diameter(self, noisy_data, n_selects = 10, verbose=False):
        device = noisy_data['E_t'].get_device()
        # print(f"Device: {device}")

        if device == -1:
            device = "cpu"

        adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()
        # adj_matrix = noisy_data['E_t'] # .sum(dim=-1).float()
        if verbose:
            print(f"Adj shape: {adj_matrix.shape}")

        node_counts = get_n_nodes(noisy_data['node_mask']).to(device)
        # print(node_counts)
        self.adj_matrix = adj_matrix.float() # .to(torch.long)

        # print(self.adj_matrix)

        max_ecc = torch.zeros(self.adj_matrix.size(0)).to(device)
        bs = self.adj_matrix.size(0)

        self.node_mask = noisy_data['node_mask'].t()


        # total_values = self.adj_matrix.size(0) * self.adj_matrix.size(1)
        # per_adj_zero_values = self.adj_matrix.size(0)

        # print(multiplier)
        # print(accumulator)
        for i in range(n_selects):

            multiplier = torch.eye(self.adj_matrix.size(1)).repeat(self.adj_matrix.size(0), 1, 1).to(
                torch.float).to(device)  # + self.adj_matrix.to(torch.long)
            accumulator = torch.zeros((self.adj_matrix.size(1), self.adj_matrix.size(
                0))).to(device)

            previous = torch.clone(accumulator)
# Facebook Real 22,470 171,002 15 0,360 6.80 × 10−4
            # X is zeros shape bs x n_nodes, with one element per graph set to one
            node_indices = torch.Tensor([torch.randint(int(node_counts[n]), (1,)) for n in range(node_counts.size(0))]).to(torch.int64).to(device)
            X = torch.nn.functional.one_hot(node_indices, num_classes = int(torch.max(node_counts).item())).to(torch.float).unsqueeze(-1).to(device)

            ecc = 0
            eccs = torch.zeros((bs)).to(device)
            # print(accumulator)
            # print(torch.count_nonzero(accumulator).item())
            while 0. in accumulator[self.node_mask]: #torch.count_nonzero(accumulator[self.node_mask]).item() != total_values:

                # print(f"Ecc: {ecc}, missing values: {torch.count_nonzero(accumulator[self.node_mask]).item()}, {torch.count_nonzero(accumulator).item()}")
                ecc += 1

                # if ecc >= 5:
                #     break

                multiplier = multiplier @ self.adj_matrix
                messages_passed = multiplier @ X
                # print(messages_passed.squeeze().t().shape, accumulator.shape, multiplier.shape)
                accumulator = accumulator + messages_passed.squeeze().t()


                # print(self.node_mask.shape, previous.shape)
                try:
                    if torch.count_nonzero(previous[self.node_mask]).item() == torch.count_nonzero(accumulator[self.node_mask]).item():
                        break
                except:
                    break

                previous = torch.clone(accumulator)

                # print(X)
                # print(self.adj_matrix)
                # print(f"Multipliyer @ X: {multiplier @ X}")
                # print(f"Accumulator: {accumulator}")

                for ib in range(bs):
                    if 0. in accumulator[ib, :][self.node_mask[ib, :]]:
                        eccs[ib] = ecc


            for ind_ecc in range(eccs.size(0)):
                if eccs[ind_ecc] > max_ecc[ind_ecc]:
                    max_ecc[ind_ecc] = eccs[ind_ecc]

        # print(f"Found diameter estimates: {max_ecc}")

        # print(f"As proportion of graphs: {max_ecc / node_counts}")

        return (max_ecc / node_counts).unsqueeze(-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()


    def k2_cycle(self):
        c2 = batch_diagonal(self.k2_matrix)

        return c2.unsqueeze(-1).float(), (torch.sum(c2, dim=-1)/4).unsqueeze(-1).float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        #
        # print(f"C3: {c3}\n"
        #       f"C3 unsqueezed: {(c3 / 2).unsqueeze(-1).float()}\n"
        #       f"second term: {(torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()}")

        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)

        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k2x, k2y = self.k2_cycle()
        assert (k2x >= -0.1).all()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k2x, k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k2y, k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy

class ReducedKNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()

    def k2_cycle(self):
        c2 = batch_diagonal(self.k2_matrix)

        return c2.unsqueeze(-1).float(), (torch.sum(c2, dim=-1) / 4).unsqueeze(-1).float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        #
        # print(f"C3: {c3}\n"
        #       f"C3 unsqueezed: {(c3 / 2).unsqueeze(-1).float()}\n"
        #       f"second term: {(torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()}")

        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()


    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k2x, k2y = self.k2_cycle()
        assert (k2x >= -0.1).all()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()


        kcyclesx = torch.cat([k2x, k3x, k4x], dim=-1)
        kcyclesy = torch.cat([k2y, k3y, k4y], dim=-1)
        return kcyclesx, kcyclesy