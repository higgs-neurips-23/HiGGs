"""
Sampling functions for HiGGs, for NeurIPS 23, by anonymous authors
"""


from tqdm import tqdm

from higgs_utils import *

import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

# Need DiGress utils - liable to change in later version
pwd = os.getcwd()
dgd_dir = os.path.join(pwd, "dgd")
sys.path.insert(0, dgd_dir)
import utils


def get_h2_graph(cfg):
    """
    Samples an h2 graph using a pre-trained model (HiGGs Stage One)

    params:
    param: cfg: omegadict.DictConfig, config keys for model loading

    returns:
    param: h2_graph: [[h1_graph_categories, h2_adjacency]] as torch.Tensors
    """
    cfg_h2 = cfg.copy()
    cfg_h2["dataset"]["h"] = 2
    cfg_h2["model"]["extra_features"] = 'all'

    # Load h2 model and updated config, then move to GPU if available
    cfg_h2, model_h2 = get_samplers(cfg_h2)
    device = 'cuda' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'
    model_h2 = model_h2.to(device)

    # Sample the graph, and a networkx.Graph version
    h2_graph, h2_networkx = model_h2.sampling_batch_h2(0, 1, 0, 40, 1, figures = True)

    # print(f"h2 X: {h2_graph[0][0]}")
    # print(f"Unique h2: {np.unique(h2_graph[0][1])}")
    # print(f"h2 E: {h2_graph[0][1]}")
    # print(f"Sampled inter-community {h2_networkx[0]}")

    return h2_graph

def get_h1_graphs(cfg, h1_graph_types, h1_batch_size):
    """
    Sample h1 graphs (HiGGs Stage Two)

    params:
    param: cfg: omegadict.DictConfig, config keys for model loading
    param: h1_graph_types: torch.Tensor categories for conditioning h1 graph sampling
    param: h1_batch_size: Size of batches to use during sampling

    returns:
    param: h1_adjs:     torch.Tensor adjacency matrices for each sampled h1 graph
    param: h1_node_types:   torch.Tensor of node classes for each sampled h1 graph
    param: h1_label_mapping:    dictionary mapping to keep track of node ids
    param: node_to_class:   dictionary mapping of node ids and sampled class for that node
    param: n_nodes_counter:     list of graph sizes as each h1 graph is added
    """

    # Use config to load h1 sampling model and send to gpu if available
    cfg_h1 = cfg.copy()
    cfg_h1["dataset"]["h"] = 1
    cfg_h1, model_h1 = get_samplers(cfg_h1)#, model_kwargs_h1)
    device = 'cuda' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'
    model_h1 = model_h1.to(device)

    # Get dimensions of model outputs
    x_out, y_out, e_out = model_h1.Xdim_output, model_h1.ydim_output, model_h1.Edim_output

    # Throws errors currently for out_dim==1
    if x_out == 1:
        x_out += 1
    if y_out == 1:
        y_out += 1
    if e_out == 1:
        e_out += 1

    # Get One-hot representation of graph categories to condition h1 sampling
    temp_batch_size = h1_batch_size
    h1_y = torch.nn.functional.one_hot(h1_graph_types, num_classes = y_out)

    # Need to instantiate some aggregators, and determine batch sizes
    h1_node_types = []
    h1_adjs = []
    n_nodes_counter = []
    h1_label_mapping = []
    if h1_graph_types.shape[0] % temp_batch_size == 0:

        print(f"Data is {h1_graph_types.shape[0]} long, using {h1_graph_types.shape[0] / temp_batch_size} batches (no remainder)")

        n_batches = int(h1_graph_types.shape[0] / temp_batch_size)
        batch_sizes = np.full(n_batches, temp_batch_size)
    else:

        print(f"Data is {h1_graph_types.shape[0]} long, using {int(h1_graph_types.shape[0] / temp_batch_size) + 1} batches")

        n_batches = int(h1_graph_types.shape[0] / temp_batch_size) + 1
        batch_sizes = np.array([temp_batch_size]*(n_batches - 1) + [h1_graph_types.shape[0] % temp_batch_size])

    # Iterate over batches and conditioning classes, sample h1 graphs
    node_agg = 0
    pbar = tqdm(range(n_batches))
    node_to_class = {}
    for n in pbar:
        batch_ind_low, batch_ind_high = temp_batch_size * n, temp_batch_size*(n+1)
        try:
            y_batch = h1_y[batch_ind_low:batch_ind_high, :]
        except:
            y_batch = h1_y[batch_ind_low:,:]
        y_batch = y_batch.to(device)

        if n % 10 == 0:
            figures = True
        else:
            figures = False

        graphs, networkx_graphs = model_h1.sampling_batch_h1(n, batch_sizes[n], 0, 40, y = y_batch, figures=figures)
        for i in range(len(graphs)):
            adj, nodes = graphs[i][1], graphs[i][0]

            h1_node_types.append(nodes.to("cpu"))
            h1_adjs.append(adj.to("cpu"))
            print(torch.sum(adj[-1,:]))
            n_nodes = adj.shape[0]
            n_nodes_counter += [node_agg]
            h1_label_mapping += [{"start_ind":node_agg, "n_nodes":n_nodes}]

            for i, n in enumerate(nodes.tolist()):
                node_to_class[i+node_agg] = n

            node_agg += n_nodes
            h1_label_mapping[-1]["end_ind"] = node_agg

        pbar.set_description(f"N nodes added: {node_agg}")

    model_h1.to("cpu")
    del h1_y
    del model_h1

    return h1_adjs, h1_node_types, h1_label_mapping, node_to_class, n_nodes_counter

def size_classing(type_idx, mean_size, dev_sizes, num_classes_out = 5):
    """
    Returns the binning of graph size given mu, sigma
    Currently mu, sigma are calculated during initial partitioning
    NB: Not used in HiGGs saved models

    type_idx: list(node class for each node)
    """

    num_nodes = len(type_idx)

    position = (num_nodes - mean_size) / dev_sizes
    shift_for_classes = int((num_classes_out - 1)/2)
    try:
        if position < - shift_for_classes:
            graph_type = 0
        elif position > shift_for_classes:
            graph_type = int(num_classes_out)
        else:
            graph_type = int(np.around(position, decimals=0)) + shift_for_classes
    except:
        print(f"\n\ntype idx {type_idx}\n"
              f"Community size {num_nodes}\n"
              f"Classes shift {shift_for_classes}\n"
              f"And position {position}\n"
              f"Mu, sigma {mean_size} {dev_sizes}\n")
        quit()

    return graph_type

def prepare_glue_dataset(cfg, model_glue, h1_adjs, h1_node_types, h2_edge_index, h2_edge_type, glue_batch_size):
    """
    Function to prepare the dataset for HiGGs Stage Three

    params:
    param: cfg: omegaconf.DictConfig
    param: model_glue: PyTorch model used for Stage Three
    param: h1_adjs: torch.Tensor adjacency matrices for sampled h1 graphs
    param: h1_node_types: torch.Tensor(s) of node categories for sampled h1 graphs
    param: h2_edge_index: torch.Tensor of edges in sampled h2 graph
    param: h2_edge_type: torch.Tensor of edge categories in sampled h2 graph (for conditioning)
    param: glue_batch_size: The batch size for edge-sampling

    returns:
    param: datalist: torch geometric dataset
    param: loader: dataloader for that dataset
    param: batch_size: batch size - slightly redundant to pass this back
    param: n_batches: number of batches
    param: remainder: final batch size
    param: pair_ids: list of (h1_id1, h1_id2) tuples to identify the graphs in the dataset
    """

    device = 'cuda' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'

    # Slowest stage, so use a progressbar
    n_edges_h2 = h2_edge_index.shape[1]
    pbar = tqdm(range(n_edges_h2))
    max_nodes = 700
    datalist = []
    pair_ids = []

    x_out, y_out, e_out = model_glue.Xdim_output, model_glue.ydim_output, model_glue.Edim_output
    # Avoid errors thrown when out_dim==1
    if x_out == 1:
        x_out += 1
    if y_out == 1:
        y_out += 1
    if e_out == 1:
        e_out += 1

    # Lots of code also found in dgd/datasets/..., better documented there
    # Prepares datasets of pairs of h1 graphs
    for n in pbar:
        h2_edge = h2_edge_index[:,n]

        if (h2_edge[1], h2_edge[0]) in pair_ids or h2_edge[0] == h2_edge[1]:
            continue

        pair_ids.append((h2_edge[0], h2_edge[1]))

        com1_adj, com2_adj = h1_adjs[h2_edge[0]], h1_adjs[h2_edge[1]]
        com1_nodes, com2_nodes = h1_node_types[h2_edge[0]], h1_node_types[h2_edge[1]]
        com1_nodes, com2_nodes = com1_nodes, com2_nodes

        com1_edgelist, com1_attr = dense_to_sparse(com1_adj)
        com2_edgelist, com2_attr = dense_to_sparse(com2_adj)

        pair_nodes = torch.cat((com1_nodes, com2_nodes))
        N = pair_nodes.shape[0]
        com2_edgelist = com2_edgelist + com1_nodes.shape[0]
        pair_edges = torch.cat((com1_edgelist, com2_edgelist), axis = -1)

        pos = torch.Tensor([True]*com1_nodes.shape[0] + [False]*com2_nodes.shape[0]).to(torch.int)
        pos = torch.Tensor(utils.get_intra_matrix(pos))
        pos = F.pad(input=pos,
                    pad=(0, max_nodes - N, 0, max_nodes - N),
                    value=False)

        x = F.one_hot(pair_nodes, num_classes=x_out)
        if x_out == 2:
            x = x[:,0].reshape(-1,1)

        graph_type = h2_edge_type[n]
        y = F.one_hot(torch.tensor([graph_type]), num_classes=y_out).float()

        edge_attr = torch.cat((com1_attr, com2_attr))
        edge_attr = F.one_hot(edge_attr, num_classes=e_out).to(torch.float)

        data = Data(x = x,
                    edge_index=pair_edges,
                    edge_attr = edge_attr,
                    pos = pos,
                    y = y).to(device)
        datalist.append(data)


    datalist = datalist
    batch_size = glue_batch_size
    remainder = len(datalist) % batch_size
    n_batches = len(datalist) / batch_size if remainder == 0 else int(len(datalist) / batch_size) + 1
    print(f"Batch size of {batch_size}, leaves remainder {remainder} as final batch")



    loader = DataLoader(datalist, batch_size=batch_size, shuffle=False)#.to(device)
    print(f"Pair-pair community dataset is size {sys.getsizeof(loader)} in memory")

    return datalist, loader, batch_size, n_batches, remainder, pair_ids

def get_glue_model(cfg):
    # Load model for HiGGs Stage Three
    cfg_glue = cfg.copy()
    cfg_glue["dataset"]["h"] = 1.5
    cfg_glue, model_glue = get_samplers(cfg_glue)#, model_kwargs_glue)

    return cfg_glue, model_glue

def get_inter_edges(cfg, model_glue, loader, n_batches, batch_size, remainder):
    """
    Sample inter-h1 edges (HiGGs Stage Three) using a loaded model and prepared dataset

    params:
    param: cfg: omegaconf.DictConfig
    param: model_glue: torch model to sample edges
    param: loader: dataloader of un-connected graphs
    param: n_batches: number of batches
    param: batch_size: the size of each batch
    param: remainder: the size of the final batch

    returns:
    param: out_nx: list of networkx.Graphs, which are connected h1 graphs

    """
    # Send model to GPU if available
    device = 'cuda' if torch.cuda.is_available() and cfg.general.gpus > 0 else 'cpu'
    model_glue = model_glue.to(device)
    out_nx = []

    pbar_glue = tqdm(loader)
    for n, batch in enumerate(pbar_glue):
        pbar_glue.set_description(f"Batch {n}")
        # Determine the size of this batch
        if n != n_batches - 1:
            bs = batch_size
        else:
            bs = remainder if remainder != 0 else batch_size
            print(f"Doing final batch with remainder {remainder}")

        # Often too many batches to visualise all
        if n % 50 == 0:
            figures = True
        else:
            figures = False

        # Predict edges
        pair_graphs, pair_nx = model_glue.sampling_batch_glue(n, bs, 0, 40, 0, data=batch, figures=figures)

        # Send predicted graphs to aggregator
        for i in range(bs):
            out_nx.append(pair_nx[i])
        del pair_graphs, pair_nx

    return out_nx


def inter_edge_specific(h1_nx, glue_nx, pair_ids):
    """
    Extract new inter-h1 edges from graphs of connected h1-subgraphs

    params:
    param: h1_nx: list of sampled h1 networkx.Graph(s)
    param: glue_nx: list of connected h1 pairs as networkx.Graph(s)
    param: pair_ids: identifiers for which h1 graphs are in which glue_nx connected pair

    returns:
    param: all_edges: list of added edges
    """

    # Aggregate all inter-h1 edges
    all_edges = []

    pbar_pairs = tqdm(range(len(pair_ids)))
    pbar_pairs.set_description("Extracting inter h1 edges")

    for i in pbar_pairs:
        # Get identifiers
        com1, com2 = pair_ids[i]

        # Get corresponding h1 graphs
        g1, g2 = h1_nx[com1], h1_nx[com2]

        # Get un-connected graphs and compose into single networkx.Graph object
        gX = nx.compose(g1, g2)
        # Get linked pair of same graphs
        gY = glue_nx[i]

        # Relabel nodes in connected pair to match un-connected original h1 graphs
        relabel_mapping = {}
        for i, n in enumerate(gX.nodes()):
            relabel_mapping[i] = n
        gY = nx.relabel_nodes(gY, relabel_mapping)

        # Add edges if they aren't already in one of the h1 graphs
        x_edges = list(gX.edges())
        y_edges = list(gY.edges())
        for e in y_edges:
            if e not in x_edges and (e[1], e[0]) not in x_edges:
                all_edges.append(e)

    return all_edges

def build_from_components(h1_node_types, h1_adjs, out_nx, h1_label_mapping, node_to_class, pair_ids):
    """
    Combines graphs from each previous stage into the final sampled graph

    params:
    param: h1_node_types: torch.Tensor of node categories from Stage Two
    param: h1_adjs: Adjacency matrices for each sampled graph from Stage Two
    param: out_nx: connected networkx.Graph(s) from Stage Three
    param: h1_label_mapping: Dictionary mapping to indicate start and end node ids in each pair
    param: node_to_class: dictionary mapping, node_id:node_category
    param: pair_ids: List of h1 graph identifiers for each edge-sampling pair

    returns:
    param: G: Final HiGGs sampled networkx.Graph
    param: cg: Largest component from the HiGGs sampled networkx.Graph

    """
    # Instantiate empty graph
    G = nx.Graph()

    # Relabel each h1 graph and compose it into the final sampled graph
    h1_relabels = []
    pbar_h1s = tqdm(range(len(h1_node_types)))
    for i in pbar_h1s:
        mapping = h1_label_mapping[i]
        shift = mapping["start_ind"]

        adj = h1_adjs[i]

        g = nx.from_numpy_array(adj.cpu().numpy())
        g = nx.convert_node_labels_to_integers(g, first_label=shift)
        G = nx.compose(G, g)

        h1_relabels.append(g)

    # Get edges from Stage Three (inter-h1 edges)
    all_edges = inter_edge_specific(h1_relabels, out_nx, pair_ids)

    # Add these edges to the final graph
    print(f"Before adding IC edges G is a {G}")
    G.add_edges_from(all_edges)
    print(f"After adding IC edges G is a {G}")

    # Try and add a node category for each sampled node (-1. if none is found due to some bug, this is v rare)
    for n in tqdm(list(G.nodes())):
        try:
            G.nodes[n]["target"] = node_to_class[n]
        except:
            G.nodes[n]["target"] = -1.

    # Relabel G
    G = nx.convert_node_labels_to_integers(G)
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    cg = CGs[0]
    cg = nx.relabel_nodes(cg, mapping=lambda x: int(x))

    print(f"HiGGs sampled {G}, largest component is a {cg}")

    return G, cg