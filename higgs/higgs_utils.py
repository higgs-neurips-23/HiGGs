"""
util functions for HiGGs, anonymous author, 2023
"""

import numpy as np
import networkx as nx
import wandb
import os
import sys
import pandas as pd

import omegaconf

import torch
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
import pickle

from datetime import datetime

from torch_geometric.io import read_npz
from torch_geometric.utils import to_networkx

pwd = os.getcwd()
dgd_dir = os.path.join(pwd, "dgd")
sys.path.insert(0, dgd_dir)

import utils
from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics, EGOSamplingMetrics, FBSamplingMetrics, GITSamplingMetrics, FullSampleMetrics, ByClassSampleMetrics

class TooManyNodesError(Exception):
    pass

def get_samplers(cfg):
    """
    Loads a model for sampling using a given config.

    params:
    param: cfg: omegaconf.DictConfig config keys

    returns:
    param: cfg: the same config with updated keys from the loaded model
    param: model: the loaded sampling model
    """
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'

    stage = cfg["dataset"]["h"]
    print(f"Model stage: {stage}")
    if stage == 1:
        resume = cfg.general.h1_model
    elif stage == 2:
        resume = cfg.general.h2_model
    if stage == 1.5:
        resume = cfg.general.glue_model

    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume)

    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg = utils.update_config_with_new_keys_harsh(cfg, saved_cfg)
    return cfg, model

def setup_wandb(cfg):
    """
    Uses a config dictionary to initialise wandb to track sampling.
    Requires a wandb account, https://wandb.ai/

    params: cfg: omegaconf.DictConfig config dictionary

    returns:
    param: cfg: same config
    """
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if cfg.dataset.subsample:
        extra_label = "_subsampled"
    else:
        extra_label = ""

    kwargs = {'name': f"Sampling-" + datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), 'project': f'hidiff_{cfg.general.name}{extra_label}', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=False), 'reinit': True, 'mode': cfg.general.wandb, 'entity':'hierarchical-diffusion'}
    wandb.init(**kwargs)
    wandb.save('*.txt')

    wandb.log({"Type":"Sampling"})

    return cfg

def sampling_quality(original_G, cg):
    """
    Uses sampling metrics from DGD to assess the quality of the sampled graph
    compared to a baseline real version

    params:
    param: original_G: networkx.Graph of the original graph
    param: cg: networkx.Graph, the sampled version. cg refers to the largest connected component
    """
    utils.general_graph_metrics(cg)
    sampling_metrics = FullSampleMetrics([original_G])
    sampling_metrics([cg], "Full Sampling Metrics", 0, 0)

    sampling_metrics = ByClassSampleMetrics(original_G, cg)




def load_original(root_path="", name = None, all_graphs = False):
    """
    Loads original version of the graphs in our experiments

    params:
    param: root_path: string, optional, path of data directory for sbm graphs (all others are downloaded in-code), default=""
    param: name: name of dataset being loaded, ie cora, sbm, default=fb_hierarchies
    param: all_graphs: Boolean, default=False. Whether to load all the graphs in the dataset. Only relevant for the sbm dataset.

    returns:
    param: G: networkx.Graph, the loaded original graph
    """

    print(f"Loading original {name} graph with data directory {root_path}")
    if name == "cora":
        cora_url = 'https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz'
        resp = urlopen(cora_url)
        out = read_npz(BytesIO(resp.read()))

        G = to_networkx(out, to_undirected=True)

        node_classes = {n: out.y[i].item() for i,n in enumerate(list(G.nodes()))}

        nx.set_node_attributes(G, node_classes, name = "target")
        G = G.to_undirected()
        CGs = [G.subgraph(c) for c in nx.connected_components(G)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        G = CGs[0]

        G = nx.convert_node_labels_to_integers(G)

        return G

    elif name == "sbm":
        file_path = os.path.join(root_path, "sbm/raw/sbm_200.pt")

        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(
            file_path)

        if not all_graphs:
            random_graph_selection = np.random.randint(len(adjs))

            adj = adjs[random_graph_selection]

            g = nx.from_numpy_array(adj.numpy())

            these_atom_types = np.ones(g.order())

            these_edges, these_types = [], []
            for edge in g.edges():
                start, end = edge[0], edge[1]
                these_edges += [[start, end]]
                these_types += [1.]

            these_nodes = [(ind, {"target": v}) for ind, v in enumerate(these_atom_types)]
            these_edges = [(edge[0], edge[1], {"type": these_types[ind]}) for ind, edge in enumerate(these_edges)]

            G = nx.Graph()
            G.add_nodes_from(these_nodes)
            G.add_edges_from(these_edges)

            return G
        else:
            graphs = []
            for adj in adjs:
                random_graph_selection = np.random.randint(len(adjs))

                adj = adjs[random_graph_selection]

                g = nx.from_numpy_array(adj.numpy())

                these_atom_types = np.ones(g.order())

                these_edges, these_types = [], []
                for edge in g.edges():
                    start, end = edge[0], edge[1]
                    these_edges += [[start, end]]
                    these_types += [1.]

                these_nodes = [(ind, {"target": v}) for ind, v in enumerate(these_atom_types)]
                these_edges = [(edge[0], edge[1], {"type": these_types[ind]}) for ind, edge in enumerate(these_edges)]

                G = nx.Graph()
                G.add_nodes_from(these_nodes)
                G.add_edges_from(these_edges)

                graphs.append(G)
            return graphs

    else:
        resp = urlopen("https://snap.stanford.edu/data/facebook_large.zip")
        myzip = ZipFile(BytesIO(resp.read()))

        edgelist = pd.read_csv(myzip.open("facebook_large/musae_facebook_edges.csv"))
        G = nx.from_pandas_edgelist(df=edgelist, source="id_1", target="id_2")

        class_df = pd.read_csv(myzip.open("facebook_large/musae_facebook_target.csv"))

        unique_types = np.unique(class_df["page_type"])
        types = {target: i for i, target in enumerate(unique_types.tolist())}

        node_classes = {n: types[class_df.at[n, "page_type"]] for n in list(G.nodes())}

        nx.set_node_attributes(G, node_classes, name="target")
        return G