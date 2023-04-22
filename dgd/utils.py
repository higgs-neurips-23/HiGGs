import os
from copy import deepcopy
from typing import Optional, Union, Dict
import torch_geometric.utils
from omegaconf import OmegaConf, open_dict
import pytorch_lightning as pl
from overrides import overrides
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch
import networkx as nx
import numpy as np
import wandb


def get_intra_matrix(pos):
    """
    Generates a matrix of whether a node is
    in the same graph as another node

    args:
    - pos: torch.Tensor, (N1 + N2) boolean h1 identifiers, i.e. [0,0,0,0,...,1,1,1,1,1]

    returns:
    - keep_matrix: torch.Tensor, (N1+N2 x N1+N2) boolean for whether each pair of nodes are in the same h1 graph
    """
    pos_x = pos.repeat(pos.shape[0], 1)
    pos_y = pos.reshape(1, -1).t()
    pos_y = pos_y.repeat(1, pos_y.shape[0])

    keep_matrix = torch.eq(pos_x, pos_y)

    return keep_matrix

def to_networkx(node_list, adjacency_matrix):
    """
    Convert graphs to networkx graphs
    node_list: the nodes of a batch of nodes (bs x n)
    adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
    """
    graph = nx.Graph()

    for i in range(len(node_list)):
        if node_list[i] == -1:
            continue
        graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

    rows, cols = np.where(adjacency_matrix >= 1)
    edges = zip(rows.tolist(), cols.tolist())
    for edge in edges:
        edge_type = adjacency_matrix[edge[0]][edge[1]]
        graph.add_edge(edge[0], edge[1], color=float(edge_type), weight=3 * edge_type)

    return graph


def create_folders(args):
    try:
        os.makedirs('graphs')
        os.makedirs('chains')
    except OSError:
        pass

    try:
        os.makedirs('graphs/' + args.general.name)
        os.makedirs('chains/' + args.general.name)
    except OSError:
        pass


class EMA(pl.Callback):
    """Implements EMA (exponential moving average) to any kind of model.
    EMA weights will be used during validation and stored separately from original model weights.
    How to use EMA:
        - Sometimes, last EMA checkpoint isn't the best as EMA weights metrics can show long oscillations in time. See
          https://github.com/rwightman/pytorch-image-models/issues/102
        - Batch Norm layers and likely any other type of norm layers doesn't need to be updated at the end. See
          discussions in: https://github.com/rwightman/pytorch-image-models/issues/106#issuecomment-609461088 and
          https://github.com/rwightman/pytorch-image-models/issues/224
        - For object detection, SWA usually works better. See   https://github.com/timgaripov/swa/issues/16
    Implementation detail:
        - See EMA in Pytorch Lightning: https://github.com/PyTorchLightning/pytorch-lightning/issues/10914
        - When multi gpu, we broadcast ema weights and the original weights in order to only hold 1 copy in memory.
          This is specially relevant when storing EMA weights on CPU + pinned memory as pinned memory is a limited
          resource. In addition, we want to avoid duplicated operations in ranks != 0 to reduce jitter and improve
          performance.
    """

    def __init__(self, decay: float = 0.9999, ema_device: Optional[Union[torch.device, str]] = None, pin_memory=True):
        super().__init__()
        self.decay = decay
        self.ema_device: str = f"{ema_device}" if ema_device else None  # perform ema on different device from the model
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False  # Only works if CUDA is available
        self.ema_state_dict: Dict[str, torch.Tensor] = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    @staticmethod
    def get_state_dict(pl_module: pl.LightningModule):
        """Returns state dictionary from pl_module. Override if you want filter some parameters and/or buffers out.
        For example, in pl_module has metrics, you don't want to return their parameters.
        code:
            # Only consider modules that can be seen by optimizers. Lightning modules can have others nn.Module attached
            # like losses, metrics, etc.
            patterns_to_ignore = ("metrics1", "metrics2")
            return dict(filter(lambda i: i[0].startswith(patterns), pl_module.state_dict().items()))
        """
        return pl_module.state_dict()

    @overrides
    def on_train_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule) -> None:
        # Only keep track of EMA weights in rank zero.
        if not self._ema_state_dict_ready and pl_module.global_rank == 0:
            self.ema_state_dict = deepcopy(self.get_state_dict(pl_module))
            if self.ema_device:
                self.ema_state_dict = {k: tensor.to(device=self.ema_device) for k, tensor in
                                       self.ema_state_dict.items()}

            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()}

        self._ema_state_dict_ready = True

    @overrides
    def on_train_batch_start(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, batch, batch_idx, *args,
                             **kwargs) -> None:
        if self.original_state_dict != {}:
            # Replace EMA weights with training weights
            pl_module.load_state_dict(self.original_state_dict, strict=False)

    @rank_zero_only
    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: pl.LightningModule, *args, **kwargs) -> None:
        # Update EMA weights
        with torch.no_grad():
            for key, value in self.get_state_dict(pl_module).items():
                ema_value = self.ema_state_dict[key]
                ema_value.copy_(self.decay * ema_value + (1. - self.decay) * value, non_blocking=True)

        # Setup EMA for sampling in on_train_batch_end
        self.original_state_dict = deepcopy(self.get_state_dict(pl_module))
        ema_state_dict = pl_module.trainer.training_type_plugin.broadcast(self.ema_state_dict, 0)
        self.ema_state_dict = ema_state_dict
        assert self.ema_state_dict.keys() == self.original_state_dict.keys(), \
            f"There are some keys missing in the ema static dictionary broadcasted. " \
            f"They are: {self.original_state_dict.keys() - self.ema_state_dict.keys()}"
        pl_module.load_state_dict(self.ema_state_dict, strict=False)

        if pl_module.global_rank > 0:
            # Remove ema state dict from the memory. In rank 0, it could be in ram pinned memory.
            self.ema_state_dict = {}

    @overrides
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

    @overrides
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._ema_state_dict_ready:
            return  # Skip Lightning sanity validation check if no ema weights has been loaded from a checkpoint.

    @overrides
    def on_save_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict) -> dict:
        return {"ema_state_dict": self.ema_state_dict, "_ema_state_dict_ready": self._ema_state_dict_ready}

    @overrides
    def on_load_checkpoint(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict):
        self._ema_state_dict_ready = callback_state["_ema_state_dict_ready"]
        self.ema_state_dict = callback_state["ema_state_dict"]


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """
    X : node features
    E : edge features
    y : global features`
    norm_values : [norm value X, norm value E, norm value y]
    norm_biases : same order
    node_mask
    """
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def to_dense(x, edge_index, edge_attr, batch):
    X, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def update_config_with_new_keys(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            if key not in cfg.general.keys():
                setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            if key not in cfg.train.keys():
                setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            if key not in cfg.model.keys():
                setattr(cfg.model, key, val)
    return cfg

#Same as above, but replaces all model keys with those from the saved config
def update_config_with_new_keys_harsh(cfg, saved_cfg):
    saved_general = saved_cfg.general
    saved_train = saved_cfg.train
    saved_model = saved_cfg.model

    for key, val in saved_general.items():
        OmegaConf.set_struct(cfg.general, True)
        with open_dict(cfg.general):
            # if key not in cfg.general.keys():
            setattr(cfg.general, key, val)

    OmegaConf.set_struct(cfg.train, True)
    with open_dict(cfg.train):
        for key, val in saved_train.items():
            # if key not in cfg.train.keys():
            setattr(cfg.train, key, val)

    OmegaConf.set_struct(cfg.model, True)
    with open_dict(cfg.model):
        for key, val in saved_model.items():
            # if key not in cfg.model.keys():
            setattr(cfg.model, key, val)
    return cfg

# Logging graph metrics is called during main.py and sample_main.py, and is model-agnostic
def general_graph_metrics(datamodule, return_values = False):

    if type(datamodule) == nx.Graph:
        G = datamodule
    else:
        G = datamodule.G

    density  = nx.density(G)
    try:
        diameter = nx.diameter(G)
    except:
        diameter = -1.
    n_nodes  = G.number_of_nodes()
    n_edges  = G.number_of_edges()
    transitivity = nx.transitivity(G)
    clustering   = nx.average_clustering(G)

    print(f"=" * 50 +
          f'\nN Nodes: {n_nodes}\n'
          f'N Edges: {n_edges}\n'
          f'Density: {density}\n'
          f'Diameter: {diameter}\n'
          f'Transitivity: {transitivity}\n'
          f'Clustering: {clustering}\n' + "=" * 50)

    wandb.log({"N Nodes": n_nodes,
               "N Edges": n_edges,
               "Density": density,
               "Diameter": diameter,
               "Transitivity": transitivity,
               "Clustering": clustering}, commit=True)

    if return_values:
        return {"N Nodes": n_nodes,
               "N Edges": n_edges,
               "Density": density,
               "Diameter": diameter,
               "Transitivity": transitivity,
               "Clustering": clustering}





class PlaceHolder:
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """ Changes the device and dtype of X, E, y. """
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

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

    def mask(self, node_mask, collapse=False):
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if str(self.X.layout) == "torch.sparse_coo":
            self.X = self.X.to_dense()
            self.E = self.E.to_dense()
            return_sparse = True
        else:
            return_sparse = False

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = - 1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = - 1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))

            if return_sparse:
                self.X = self.X.to_sparse()
                self.E = self.E.to_sparse()
        return self

def histogram_from_pred_sample(sample_ref, sample_pred, title):
    # Not currently implemented
    pass
    # ref_array = np.zeros(sample_ref[0].shape[0])
    # pred_array = np.zeros(sample_pred[0].shape[0])
    # for s in range(len(sample_ref)):
    #     ref_array = ref_array + sample_ref[s]
    #     ref_array = ref_array + sample_ref[s]

    # ref_array = np.concatenate(sample_ref).ravel()
    # pred_array = np.concatenate(sample_pred).ravel()
    #
    # fig, ax = plt.subplots(figsize=(5,5))
    # # ax.hist(ref_array, label = "Real", bins=100)
    # # ax.hist(pred_array, label = "Sampled", bins=100)
    #
    # ax.hist([ref_array, pred_array], label=["Real", "Sampled"])
    #
    # ax.set_yscale('log')
    #
    # ax.legend(shadow=True)
    # ax.set_title(title)
    # plt.savefig(f"{title}.png")
    #
    # try:
    #     im = plt.imread(f"{title}.png")
    #     wandb.log({f"Charts/{title}": [wandb.Image(im, caption=f"{title}.png")]})
    # except:
    #     pass
