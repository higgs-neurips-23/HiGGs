import os
import os.path as osp
import pathlib
from typing import Any, Sequence
import pickle

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import to_networkx
from torch_geometric.io import read_npz

import networkx as nx
import networkx.algorithms.community as comm
import wandb
import concurrent.futures
import utils
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
from analysis.visualization import TrainDiscreteNodeTypeVisualization, LargeGraphVisualization


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class CORADataset(InMemoryDataset):
    """
    Anonymous author - datamodule for the Cora graph for HiGGs
    Saves data files for h1, h2 and edge-prediction
    """
    raw_url = ('https://github.com/abojchevski/graph2gauss/raw/master/data/cora_ml.npz')

    def __init__(self, stage, root,  h = 1,
                 max_size=100, resolution=20, n_samples=240, n_workers=4,
                 batch_size = 4, regime = 'majority'):
        """
        param: stage: train, val, test identifier
        param: root: root data directory
        param: h: hierarchy for this set of processing
        param: max_size: maximum size of graph to allow (doubled for edge-prediction)
        param: resolution: resolution to use for Louvain partitioning for dataset creation
        param: n_samples: Number of times to run Louvain partitioning
        param: n_workers: Number of threads to use in processing
        param: batch_size: Batch size - used to restrict dataset size
        param: regime: How to label h1 graphs (ie for conditioning during sampling)
        """


        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.h = h
        self.max_size = max_size
        self.resolution = resolution
        self.partitions_h1 = {}
        self.partitions_h2 = []
        self.n_samples = n_samples
        self.n_workers = n_workers
        self.batch_size = batch_size

        if regime == 'majority':
            self.graph_classer = self.majority_classing
            self.graph_class_dim = None
        elif regime == 'size':
            self.graph_classer = self.size_classing
            self.graph_class_dim = 6

        super().__init__(root, None, None, None)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['cora_ml.npz',
                'cora_target.csv',
                'cora_G.pkl',
                'cora_partitions.csv']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        # Different file names for each hierarchy
        if self.h == 1:
            return ['proc_train_h1.pt', 'proc_val_h1.pt', 'proc_test_h1.pt']
        elif self.h == 2:
            return ['proc_train_h2.pt', 'proc_val_h2.pt', 'proc_test_h2.pt']
        elif self.h == 1.5:
            return ['proc_train_X.pt', 'proc_val_X.pt', 'proc_test_X.pt',
                    'proc_train_Y.pt', 'proc_val_Y.pt', 'proc_test_Y.pt']
        else:
            raise NotImplementedError(f"Hierarchy {self.h} is not implemented!")

    def download(self):
        """
        Download CORA files. Taken from PyG CORA class
        """
        path = download_url(self.processed_url, self.raw_dir)

        if files_exist(self.split_paths):
            return

        edges = read_npz(self.raw_paths[0])
        G = to_networkx(edges, to_undirected=True)

        node_classes = {n: edges.y[i].item() for i,n in enumerate(list(G.nodes()))}
        nx.set_node_attributes(G, node_classes, name = "target")
        # Cora is initially a directed graph
        G = G.to_undirected()
        CGs = [G.subgraph(c) for c in nx.connected_components(G)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        G = CGs[0]
        G = nx.convert_node_labels_to_integers(G)

        print(G)

        print(f"\nFinding communities {self.n_samples} times with a resolution of {self.resolution}\n")
        self.partitions_h2 = []
        sample_call_list = range(self.n_samples)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_partition = {executor.submit(self.communities_split, G): i for i in tqdm(sample_call_list)}
            for future in tqdm(concurrent.futures.as_completed(future_to_partition)):
                partition = future.result()
                try:
                    self.partitions_h2.append(partition)
                except:
                    print(f"Failed with partition {partition}")

        with open(self.raw_paths[3], "wb") as handle:
            pickle.dump(self.partitions_h2, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.G = G
        with open(self.raw_paths[2], "wb") as handle:
            pickle.dump(self.G, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.raw_paths[2], "rb") as handle:
            self.G = pickle.load(handle)
            print(self.G)

        fig, ax = plt.subplots(figsize=(12, 12))

        pos = nx.spring_layout(G)
        nx.draw_networkx_edges(G, ax = ax, pos=pos, alpha=0.5)

        plt.tight_layout()
        plt.savefig("real_full_comm_colours.png", dpi=300)

        im = plt.imread("real_full_comm_colours.png")
        wandb.log(
            {"Media/Real_Sampled_Comms": [wandb.Image(im, caption="Coloured by detected communities")]})

        nodes = [n for n in list(G.nodes())]
        labels = [n[1]["target"] for n in list(G.nodes(data=True))]

        target_df = pd.DataFrame({"id":nodes, "page_type":labels})
        target_df.to_csv(self.raw_paths[1])

        dataset = pd.DataFrame(
            {'community_id': [i for i in range(len(self.partitions_h2))]})  # read_csv(self.raw_paths[1])

        print(f"Done building CSV:\n{dataset.head()}")

        n_train = int(0.8 * self.n_samples)
        n_test = int(0.1 * self.n_samples)
        n_val = self.n_samples - (n_train + n_test)

        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])
        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def communities_split(self, G):
        partition = comm.louvain_communities(G, resolution=self.resolution)

        self.community_diagnostics(partition)
        return partition

    def communities_to_edges(self):
        partition_dict = {}
        for run in self.partitions_h2:
            for i, p in enumerate(run):
                subg = self.G.subgraph(p)
                edges = subg.edges()
                edges = [list(e) for e in edges]
                partition_dict[i] = edges
            if self.partitions_h1 == {}:
                max_partition_so_far = 0
            else:
                max_partition_so_far = max(list(self.partitions_h1.keys()))

            for p in partition_dict:
                self.partitions_h1[p + max_partition_so_far] = partition_dict[p]

    def make_meta_graph(self, partition):
        """
        Use community algorithm to produce meta-graph of communities and intra-links.
        Meta-graph edge weights are the number of inter-community links in original graph.
        """

        # Get community partition of form {"community_id":{"node_id1", "node_id2", "node_id3",...}, ...}
        community_to_node = {i: p for i, p in enumerate(partition)}

        # Invert community to node, ie new partition = {"node_id":"community_id", ...}
        partition = {}
        for comm in community_to_node:
            nodes = community_to_node[comm]
            for n in nodes:
                partition[n] = comm

        # self.partition = partition

        # Find unique community ids
        community_unique = set([k for k in community_to_node.keys()])

        # Produce a sub-graph for each community
        subgraphs = []
        for c in community_unique:
            subgraphs.append(nx.subgraph(self.G, community_to_node[c]))

        # Get nested list of edges in original graph
        G_edgelist = [[e1, e2] for (e1, e2) in nx.edges(self.G)]

        # Build nested list of edges, of form [["community_id1", "community_id2"], ["community_id3", "community_id4"], ...]
        community_edgelist = []
        for e in G_edgelist:
            comm1 = partition[e[0]]
            comm2 = partition[e[1]]

            community_edgelist.append((comm1, comm2))

        # Find unique edges that are inter-community
        unique_comm_edges = list(set(community_edgelist))
        out_edges = []
        for e in unique_comm_edges:
            if (e[1], e[0]) not in out_edges and e[0] != e[1]:
                out_edges.append(e)
        unique_comm_edges = out_edges

        # Build metagraph as a weighted networkx graph
        metaG = nx.Graph()
        # metaG.add_weighted_edges_from(full_description)
        metaG.add_edges_from(unique_comm_edges)

        # Set metagraph and community subgraphs as attributes
        # self.subgraphs = {i:g for i, g in enumerate(subgraphs)}
        return metaG


    def get_joined_pairs(self, partition):
        """
        Use community algorithm to produce meta-graph of communities and intra-links.
        Meta-graph edge weights are the number of inter-community links in original graph.
        """

        # Get community partition of form {"community_id":{"node_id1", "node_id2", "node_id3",...}, ...}
        community_to_node = {i: p for i, p in enumerate(partition)}

        # Invert community to node, ie new partition = {"node_id":"community_id", ...}
        partition = {}
        for comm in community_to_node:
            nodes = community_to_node[comm]
            for n in nodes:
                partition[n] = comm

        # self.partition = partition

        # Find unique community ids
        community_unique = set([k for k in community_to_node.keys()])

        # Produce a sub-graph for each community
        subgraphs = []
        for c in community_unique:
            subgraphs.append(nx.subgraph(self.G, community_to_node[c]))

        # Get nested list of edges in original graph
        G_edgelist = [[e1, e2] for (e1, e2) in nx.edges(self.G)]

        # Build nested list of edges, of form [["community_id1", "community_id2"], ["community_id3", "community_id4"], ...]
        community_edgelist = []
        for e in G_edgelist:
            comm1 = partition[e[0]]
            comm2 = partition[e[1]]

            community_edgelist.append((comm1, comm2))

        # Find unique edges that are inter-community
        unique_comm_edges = list(set(community_edgelist))
        out_edges = []
        for e in unique_comm_edges:
            if (e[1], e[0]) not in out_edges and e[0] != e[1]:
                out_edges.append(e)
        unique_comm_edges = out_edges

        return unique_comm_edges

    def community_diagnostics(self, partition):

        # print(f"N communities: {len(partition)}")

        sizes = [len(partition[k]) for k in range(len(partition))]

        wandb.log({"Mean_Community_Size": np.mean(sizes),
                   "Community_Size_Deviation": np.std(sizes),
                   "Num_communities": len(sizes),
                   "Max_Community_Size": np.max(sizes),
                   "Min_Community_Size": np.min(sizes)})

    def process(self):

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        print(target_df.head())

        node_type_df = pd.read_csv(self.raw_paths[1])
        unique_types = np.unique(node_type_df["page_type"])
        types = {target: i for i, target in enumerate(unique_types.tolist())}

        if self.graph_class_dim is None:
            self.graph_class_dim = len(types)

        with open(self.raw_paths[3], "rb") as handle:
            self.partitions_h2 = pickle.load(handle)
        with open(self.raw_paths[2], "rb") as handle:
            self.G = pickle.load(handle)


        # Calculate sizes here - not a very heavy operation but could definitely be optimised
        # TODO: this would make way more sense outside of stage-specific calculation

        self.communities_to_edges()
        graphs_h1 = [nx.from_edgelist(self.partitions_h1[i]) for i in list(self.partitions_h1.keys())]
        h1_sizes = [len(list(g.nodes())) for g in graphs_h1]

        self.mean_size = np.mean(h1_sizes)
        self.dev_sizes = np.std(h1_sizes)
        

        if self.h == 1:

            self.communities_to_edges()

            graphs_h1 = [nx.from_edgelist(self.partitions_h1[i]) for i in list(self.partitions_h1.keys())]

            skip = []
            dataset_size = 250 * self.batch_size
            for i, G in enumerate(graphs_h1):
                if G.number_of_nodes() > self.max_size or i > dataset_size:
                    skip.append(i)

            suppl = tqdm(graphs_h1)

            data_list = []
            all_nodes = []
            node_types = []
            graphs_h1_plotting = []

            for i, G in enumerate(tqdm(suppl)):
                if i in skip:
                    continue

                N = G.number_of_nodes()

                type_idx = []
                for node in list(G.nodes()):
                    node_type = node_type_df.at[node, "page_type"]
                    type_idx.append(types[node_type])

                G = nx.convert_node_labels_to_integers(G)
                graphs_h1_plotting.append(G)
                typedict = {idx:type_idx[idx] for idx in range(len(type_idx))}
                node_types.append(typedict)

                row, col, edge_type = [], [], []
                for edge in list(G.edges()):
                    start, end = edge[0], edge[1]
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [1]

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type, num_classes=2).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                try:
                    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                except:
                    continue

                graph_type = self.graph_classer(type_idx)
                try:
                    y = F.one_hot(torch.tensor([graph_type]), num_classes=self.graph_class_dim).float()
                except:
                    print(f"Graph type {graph_type}, max classes {self.graph_class_dim}")
                    quit()

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                data_list.append(data)

            print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
            torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

            n_total = len(all_nodes)
            type_counts = [all_nodes.count(typ) for typ in list(np.unique(all_nodes))]
            self.node_types = torch.tensor(type_counts) / n_total
            print(f"File node type marginals: {self.node_types}")

            visualization_tools = TrainDiscreteNodeTypeVisualization()

            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_h1')
                visualization_tools.visualize(result_path, graphs_h1_plotting, min(15, len(graphs_h1_plotting)),
                                              node_types = node_types, log = "h1_Real")
                visualization_tools.visualize_grid(result_path, graphs_h1_plotting, min(15, len(graphs_h1_plotting)),
                                              node_types=node_types, log = "h1_Real")

        elif self.h == 1.5:

            self.communities_to_edges()

            X_graphs = []
            Y_graphs = []

            n_skipped_skip = 0
            n_skipped_target_df = 0
            n_not_skipped = 0

            dataset_size = 250 * self.batch_size

            if self.stage == "train":
                n_runs_considered = int(3*self.n_samples / 5)
            else:
                n_runs_considered = int(self.n_samples / 5)
            min_nodes = 2
            for n_part, model_partition in tqdm(enumerate(self.partitions_h2[:n_runs_considered])):
                if n_part > dataset_size:
                    continue
                joined_pairs = self.get_joined_pairs(model_partition)

                for pair in joined_pairs:
                    x1, x2 = pair[0], pair[1]


                    nodes1, nodes2 = model_partition[x1], model_partition[x2]

                    if len(nodes1) > self.max_size or len(nodes2) > self.max_size or len(nodes1) < min_nodes or len(nodes2) < min_nodes:
                        n_skipped_skip += 1
                        continue

                    g1, g2, g3 = nx.subgraph(self.G, nodes1), nx.subgraph(self.G, nodes2), nx.subgraph(self.G, list(nodes1) + list(nodes2))

                    separate_graph = nx.compose(g1, g2)
                    X_graphs.append(separate_graph)
                    Y_graphs.append(g3)

            skip = []
            max_nodes = 0

            included = 0
            for i, G in enumerate(X_graphs):
                if G.number_of_nodes() > 2 * self.max_size or included >= dataset_size:
                    skip.append(i)
                else:
                    if G.number_of_nodes() > max_nodes:
                        max_nodes = G.number_of_nodes()
                    included += 1

            data_list = []
            all_nodes = []
            node_types = []
            edge_types = []
            graphs_plotting = []
            suppl = tqdm(Y_graphs)
            for i, G in enumerate(tqdm(suppl)):
                xG = X_graphs[i]

                if i in skip:
                    if i in skip:
                        n_skipped_skip += 1
                    else:
                        n_skipped_target_df += 1
                    continue
                else:
                    n_not_skipped += 1

                N = G.number_of_nodes()
                type_idx = []
                for node in list(G.nodes()):
                    node_type = node_type_df.at[node, "page_type"]
                    type_idx.append(types[node_type])

                previous_node_ids = list(G.nodes())
                G = nx.convert_node_labels_to_integers(G)

                node_mapping = {previous_node_ids[i]:list(G.nodes())[i] for i in range(len(previous_node_ids))}
                xG = nx.relabel_nodes(xG, mapping=node_mapping)
                graphs_plotting.append(G)
                typedict = {idx:type_idx[idx] for idx in range(len(type_idx))}
                node_types.append(typedict)

                row, col, edge_type = [], [], []
                for edge in list(G.edges()):
                    start, end = edge[0], edge[1]
                    row += [start, end]
                    col += [end, start]
                    edge_type += 2 * [1]

                component_graphs = nx.connected_components(xG)
                components = [list(c) for c in component_graphs]
                pos = []
                for node in list(G.nodes()):
                    c = components[0]
                    if node in c:
                        pos += [1]
                    else:
                        pos += [0]

                pos = torch.tensor(pos, dtype=torch.bool)
                pos = self.get_intra_matrix(pos)
                pos = F.pad(input=pos,
                            pad=(0, max_nodes - N, 0, max_nodes - N),
                            value = False)

                edge_types.append({e:e for e in edge_type})


                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)

                edge_attr = F.one_hot(edge_type, num_classes=2).to(torch.float)


                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                try:
                    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                except:
                    continue

                graph_type = self.graph_classer(type_idx)

                y = F.one_hot(torch.tensor([graph_type]), num_classes=self.graph_class_dim).float()
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i, pos = pos)
                data_list.append(data)

            print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
            torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

            n_total = len(all_nodes)
            type_counts = [all_nodes.count(typ) for typ in list(np.unique(all_nodes))]
            # print(type_counts)
            self.node_types = torch.tensor(type_counts) / n_total
            print(f"File node type marginals: {self.node_types}")

            visualization_tools = TrainDiscreteNodeTypeVisualization()

            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_Y')
                visualization_tools.visualize(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                              node_types = node_types, log = "Y_Real")
                visualization_tools.visualize_grid(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                              node_types=node_types, log = "Community_Pairs_Y_Real", largest_component = False)

            print(f"\n\n N_skipped from skiplist: {n_skipped_skip}\n N_skipped target_df: {n_skipped_target_df}\n N_not_skipped: {n_not_skipped}\n\n")

        elif self.h == 2:

            graphs_h2 = [self.make_meta_graph(partition) for partition in self.partitions_h2]
            densities = [nx.density(g) for g in graphs_h2]
            wandb.log({"Mean_Density": np.mean(densities),
                       "Max_Density": np.max(densities),
                       "Min_Density": np.min(densities),
                       "Dev_Density": np.std(densities),
                       "Mean_edges": np.mean([len(list(g.edges())) for g in graphs_h2]),
                       "Mean_nodes": np.mean([len(list(g.nodes())) for g in graphs_h2]),
                       "Mean_community_size": self.G.order() / np.mean([len(list(g.nodes())) for g in graphs_h2])})

            skip = []
            dataset_size = 250 * self.batch_size
            for i, G in enumerate(graphs_h2):
                if G.number_of_nodes() > self.max_size or i > dataset_size:
                    skip.append(i)

            suppl = tqdm(graphs_h2)

            data_list = []

            node_types = []
            edge_types = []
            graphs_h2_plotting = []

            for i, G in enumerate(tqdm(suppl)):
                if i in skip:
                    continue
                N = G.number_of_nodes()
                this_partition = self.partitions_h2[i]
                type_idx = []
                for node in list(G.nodes()):
                    community_nodes = this_partition[node]
                    these_node_types = []
                    for n in community_nodes:
                        these_node_types.append(node_type_df.at[n, "page_type"])
                    node_type = self.graph_classer(these_node_types)
                    type_idx.append(node_type)

                try:
                    x = F.one_hot(torch.tensor(type_idx), num_classes=self.graph_class_dim).float()
                except:
                    continue

                G = nx.convert_node_labels_to_integers(G)
                graphs_h2_plotting.append(G)
                typedict = {idx: type_idx[idx] for idx in range(len(type_idx))}
                node_types.append(typedict)

                type_idx = []
                row, col, edge_type = [], [], []
                for edge in list(G.edges()):
                    start, end = edge[0], edge[1]
                    row += [start, end]
                    col += [end, start]
                    etype = [1]
                    edge_type += 2 * etype
                    type_idx.append(etype)

                typedict = {idx: type_idx[idx] for idx in range(len(type_idx))}
                edge_types.append(typedict)

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                edge_attr = F.one_hot(edge_type, num_classes=2).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                # Code now requires at least something like a graph class
                y = np.random.randint(2)
                y = F.one_hot(torch.tensor([y]), num_classes=2).float()
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                data_list.append(data)
            print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
            torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

            visualization_tools = TrainDiscreteNodeTypeVisualization()
            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_h2')
                visualization_tools.visualize(result_path, graphs_h2_plotting, min(15, len(graphs_h2_plotting)),
                                              node_types = node_types, log = "h2_real", node_size = 50)
                visualization_tools.visualize_grid(result_path, graphs_h2_plotting, min(15, len(graphs_h2_plotting)),
                                              node_types=node_types, log = "h2_real")
        else:
            raise NotImplementedError(f"Hierarchy {self.h} is not implemented!")

    def get_intra_matrix(self, pos):
        pos_x = pos.repeat(pos.shape[0], 1)
        pos_y = pos.reshape(1, -1).t()
        pos_y = pos_y.repeat(1, pos_y.shape[0])

        keep_matrix = torch.eq(pos_x, pos_y)

        return keep_matrix

    def majority_classing(self, type_idx):
        """
        Returns the majority class of a graph

        type_idx: list(node class for each node)
        """
        these_node_types, counts = np.unique(type_idx, return_counts=True)
        if counts.shape[0] <= 1:
            graph_type = these_node_types[0]
        else:
            most_common_idx = np.argmax(counts)
            graph_type = these_node_types[most_common_idx]

        return graph_type

    def size_classing(self, type_idx, num_classes_out = 5):
        """
        Returns the binning of graph size given mu, sigma
        Currently mu, sigma are calculated during initial partitioning

        type_idx: list(node class for each node)
        """

        num_nodes = len(type_idx)

        position = (num_nodes - self.mean_size) / self.dev_sizes
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
                  f"Mu, sigma {self.mean_size} {self.dev_sizes}\n")
            quit()

        return graph_type

class CORADataModule(AbstractDataModule):
    def __init__(self, cfg):
        print("Entered CORA datamodule __init__")
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.h = cfg.dataset.h
        self.regime = cfg.dataset.regime
        print("Finished CORA datamodule __init__")

    def prepare_data(self) -> None:
        # if self.h != 1.5:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': CORADataset(stage='train',
                                         root = root_path,
                                         h = self.h,
                                         resolution=self.cfg.dataset.resolution,
                                         max_size=self.cfg.dataset.max_size,
                                         n_workers=self.cfg.train.num_workers,
                                         batch_size = self.cfg.train.batch_size,
                                         regime = self.regime),
                    'val': CORADataset(stage='val',
                                       root=root_path,
                                       h = self.h,
                                       max_size=self.cfg.dataset.max_size,
                                       n_workers=self.cfg.train.num_workers,
                                         batch_size = self.cfg.train.batch_size,
                                                regime = self.regime),
                    'test': CORADataset(stage='test',
                                        root=root_path,
                                        h=self.h,
                                        max_size=self.cfg.dataset.max_size,
                                        n_workers=self.cfg.train.num_workers,
                                         batch_size = self.cfg.train.batch_size,
                                                 regime = self.regime)}
        super().prepare_data(datasets)

        try:
            utils.general_graph_metrics(datasets["train"])
        except AttributeError:
            pass


class CORADatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = dataset_config.name
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = datamodule.node_types()  # There are no node types
        self.edge_types = self.datamodule.edge_counts()

        super().complete_infos(self.n_nodes, self.node_types)
