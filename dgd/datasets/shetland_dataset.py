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
from torch_geometric.data import Data, InMemoryDataset

import networkx as nx
import networkx.algorithms.community as comm
import osmnx as ox

ox.config(use_cache=True, log_console=True)

import wandb
import concurrent.futures


import utils
from datasets.abstract_dataset import AbstractDatasetInfos, AbstractDataModule
from analysis.visualization import TrainDiscreteNodeTypeVisualization, LargeGraphVisualization
from community_layout.layout_class import CommunityLayout


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class ShetlandDataset(InMemoryDataset):

    # raw_url2 = 'https://snap.stanford.edu/data/deezer_shetland_nets.zip'
    # processed_url = 'https://snap.stanford.edu/data/deezer_shetland_nets.zip'

    def __init__(self, stage, root, transform=None,
                 pre_transform=None, pre_filter=None, h=1,
                 max_size=100, resolution=20, n_samples=1000, n_workers=8, xy=None, batch_size=4, regime='majority'):
        print("\nStarting Shetland dataset init\n")
        self.bounding_box = [66.887, 62.805, -11.755, -26.323]
        # self.bounding_box = [ 65.7797, 65.5417, -19.7218, -20.7999]

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

        self.regime = regime

        if regime == 'majority':
            self.graph_classer = self.majority_classing
            self.max_ic_edge_class = 2
            self.graph_class_dim = None
        elif regime == 'size':
            self.graph_classer = self.size_classing
            self.graph_class_dim = 6
            self.max_ic_edge_class = 5

        elif regime == 'highway':
            self.graph_classer = self.graph_classing_highway
            self.graph_class_dim = 4
            self.max_ic_edge_class = 2

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['graph_store.pkl',
                'shetland_partitions.csv',
                'shetland_nodes.csv',
                'shetland_edgelist.csv']  # , 'deezer_shetland_nets/deezer_edges.json']

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
        if self.h == 1:
            return ['proc_train_h1.pt', 'proc_val_h1.pt', 'proc_test_h1.pt']
        elif self.h == 2:
            return ['proc_train_h2.pt', 'proc_val_h2.pt', 'proc_test_h2.pt']
        elif self.h == 1.5:
            return ['proc_train_X.pt', 'proc_val_X.pt', 'proc_test_X.pt',
                    'proc_train_Y.pt', 'proc_val_Y.pt', 'proc_test_Y.pt']
        else:
            raise NotImplementedError(f"Hierarchy {self.h} is not implemented!")

    def get_and_save_map(self, keep_attr=False):
        # output_node_path = os.path.join(self.raw_dir, self.raw_file_names[1])
        # output_edge_path = os.path.join(self.raw_dir, self.raw_file_names[2])

        if os.path.isfile(self.raw_paths[3]):
            return

        G = ox.graph_from_bbox(*self.bounding_box, network_type='drive')
        G = nx.convert_node_labels_to_integers(G)

        edge_df = nx.to_pandas_edgelist(G, source="id_1", target="id_2")
        edge_df.to_csv(self.raw_paths[3], index=False)

        node_labels = np.ones(G.order())
        node_df = pd.DataFrame()
        node_df["id"] = list(G.nodes())
        node_df["page_type"] = node_labels
        node_df.to_csv(self.raw_paths[2])

    def download(self):
        """
        Download raw shetland files. Taken from PyG Shetland class
        """

        print(self.raw_dir)
        self.get_and_save_map()
        if files_exist(self.split_paths):
            return

        edgelist = pd.read_csv(self.raw_paths[3])
        G = nx.from_pandas_edgelist(df=edgelist, source="id_1", target="id_2", edge_attr="highway")
        del edgelist
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
                    # print(f"Success with partition {partition}")
                except:
                    print(f"Failed with partition {partition}")

        with open(self.raw_paths[1], "wb") as handle:
            pickle.dump(self.partitions_h2, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.G = G
        with open(self.raw_paths[0], "wb") as handle:
            pickle.dump(self.G, handle, protocol=pickle.HIGHEST_PROTOCOL)

        #
        #
        #
        dataset = pd.DataFrame(
            {'community_id': [i for i in range(len(self.partitions_h2))]})  # read_csv(self.raw_paths[1])
        # # dataset = dataset.sample(frac = 0.1)
        print(f"Done building CSV:\n{dataset.head()}")
        # # n_samples = len(dataset)
        n_train = int(0.8 * self.n_samples)
        n_test = int(0.1 * self.n_samples)
        n_val = self.n_samples - (n_train + n_test)
        #
        # # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])
        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

        # quit()

    def communities_split(self, G):
        partition = comm.louvain_communities(G, resolution=self.resolution)

        self.community_diagnostics(partition)
        return partition

    def communities_to_edges(self):
        partition_dict = {}
        for run in self.partitions_h2:
            for i, p in enumerate(run):
                subg = self.G.subgraph(p)
                edges = subg.edges(data=True)
                # edges = [list(e) for e in edges]
                edges = [[e[0], e[1], e[2]["highway"]] for e in edges]
                partition_dict[i] = edges
            # del partition_dict
            if self.partitions_h1 == {}:
                max_partition_so_far = 0
            else:
                max_partition_so_far = max(list(self.partitions_h1.keys()))

            for p in partition_dict:
                self.partitions_h1[p + max_partition_so_far] = partition_dict[p]
        # print(p, max_partition_so_far, p + max_partition_so_far)

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
            #
            # elif (e[1], e[0]) in all_out_edges:
            #     if e in all_out_edges:
            #         all_out_edges[e] = all_out_edges[e] + 1
            #     else:
            #         all_out_edges[e] = 1

        all_out_edges = {}
        for e in list(community_edgelist):

            if e in out_edges:
                if e in all_out_edges:
                    all_out_edges[e] = all_out_edges[e] + 1
                else:
                    all_out_edges[e] = 1
            elif (e[1], e[0]) in out_edges:
                if (e[1], e[0]) in all_out_edges:
                    all_out_edges[(e[1], e[0])] = all_out_edges[(e[1], e[0])] + 1
                else:
                    all_out_edges[(e[1], e[0])] = 1

        unique_comm_edges = out_edges

        if self.regime == "size":
            weighted_edges = [(e[0], e[1], all_out_edges[e]) for e in unique_comm_edges]
        else:
            weighted_edges = [(e[0], e[1], 1.) for e in unique_comm_edges]

        # Build metagraph as a weighted networkx graph
        metaG = nx.Graph()
        metaG.add_weighted_edges_from(weighted_edges)
        # metaG.add_edges_from(unique_comm_edges)

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
        # print(target_df.head())

        node_type_df = pd.read_csv(self.raw_paths[2])
        unique_types = np.unique(node_type_df["page_type"])
        types = {target: i for i, target in enumerate(unique_types.tolist())}

        if self.graph_class_dim is None:
            self.graph_class_dim = len(types)

        # for f in open(self.raw_paths[0], "r"):
        #     all_edges = json.loads(f)
        # graphs_h2 = [nx.from_edgelist(all_edges[i]) for i in list(all_edges.keys())]
        with open(self.raw_paths[1], "rb") as handle:
            self.partitions_h2 = pickle.load(handle)
        with open(self.raw_paths[0], "rb") as handle:
            self.G = pickle.load(handle)

            layout = CommunityLayout(self.G.copy(), community_compression=0.05,
                                     layout_kwargs={"k": 1, "iterations": 250},
                                     community_kwargs={"resolution": self.resolution})

            fig, ax = plt.subplots(figsize=(12, 12))
            layout.display(ax=ax, complex_alphas=False)
            plt.tight_layout()
            plt.savefig("real_full_comm_colours.png", dpi=300)

            im = plt.imread("real_full_comm_colours.png")
            wandb.log(
                {"Media/Real_Sampled_Comms": [wandb.Image(im, caption="Coloured by detected communities")]})

        # edge_target = pd.read_csv(self.raw_paths[3])
        # print(edge_target)
        # print(edge_target["highway"])
        # print(np.unique(edge_target["highway"], return_counts=True))
        # quit()

        # Calculate sizes here - not a very heavy operation but could definitely be optimised
        # TODO: this would make way more sense outside of stage-specific calculation

        self.communities_to_edges()

        graphs_h1 = []
        for i, key in enumerate(list(self.partitions_h1.keys())):
            edge_bunch = [(e[0], e[1], {"highway":e[2]}) for e in self.partitions_h1[key]]
            graphs_h1.append(nx.from_edgelist(edge_bunch))

        # graphs_h1 = [nx.from_edgelist(self.partitions_h1[i]) for i in list(self.partitions_h1.keys())]
        h1_sizes = [len(list(g.nodes())) for g in graphs_h1]

        self.mean_size = np.mean(h1_sizes)
        self.dev_sizes = np.std(h1_sizes)

        if self.h == 1:

            # self.communities_to_edges()

            # graphs_h1 = [nx.from_edgelist(self.partitions_h1[i]) for i in list(self.partitions_h1.keys())]

            skip = []
            dataset_size = 200 * self.batch_size
            for i, G in enumerate(graphs_h1):
                if G.number_of_nodes() > self.max_size or i > dataset_size:
                    skip.append(i)

            suppl = tqdm(graphs_h1)

            data_list = []

            all_nodes = []
            node_types = []

            node_types = []
            edge_types = []
            graphs_h1_plotting = []

            for i, G in enumerate(tqdm(suppl)):
                if i in skip:  # or i not in target_df.index:
                    continue

                N = G.number_of_nodes()


                type_idx = []
                for node in list(G.nodes()):
                    node_type = node_type_df.at[node, "page_type"]
                    type_idx.append(types[node_type])


                # if self.regime != "highway":
                G = nx.convert_node_labels_to_integers(G)
                # graphs[i] = G
                graphs_h1_plotting.append(G)
                typedict = {idx: type_idx[idx] for idx in range(len(type_idx))}
                # print(typedict)
                node_types.append(typedict)

                row, col, edge_type = [], [], []
                for edge in list(G.edges(data=True)):

                    start, end, edge_attr = edge[0], edge[1], edge[2]  # bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    # print(start, end, edge_attr)
                    row += [start, end]
                    col += [end, start]
                    if self.regime == "highway":
                        edge_type += 2 * [self.clean_edge_type_highway(edge_attr["highway"])]
                    else:
                        edge_type += 2 * [1]  # [bonds[bond.GetBondType()] + 1]





                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)
                # print(edge_type)

                if self.regime == "highway":
                    num_edge_classes = 5
                else:
                    num_edge_classes = 2

                edge_attr = F.one_hot(edge_type, num_classes=num_edge_classes).to(torch.float)
                # print(edge_attr)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                # print(type_idx)
                try:
                    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                except:
                    continue

                graph_type = self.graph_classer(type_idx)
                # print(these_node_types, counts, graph_type)

                # y = torch.Tensor([graph_type]).reshape(1,1)
                try:
                    y = F.one_hot(torch.tensor([graph_type]), num_classes=self.graph_class_dim).float()
                except:
                    print(f"Graph type {graph_type}, max classes {self.graph_class_dim}")
                    quit()


                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            # data_list = data_list[:-len(data_list) % self.batch_size]

            print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
            torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

            n_total = len(all_nodes)
            type_counts = [all_nodes.count(typ) for typ in list(np.unique(all_nodes))]
            # print(type_counts)
            self.node_types = torch.tensor(type_counts) / n_total
            print(f"File node type marginals: {self.node_types}")

            visualization_tools = TrainDiscreteNodeTypeVisualization()
            # print(self.stage, graphs_h1_plotting)
            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_h1')
                visualization_tools.visualize(result_path, graphs_h1_plotting, min(15, len(graphs_h1_plotting)),
                                              node_types=node_types, log="h1_Real")
                visualization_tools.visualize_grid(result_path, graphs_h1_plotting, min(15, len(graphs_h1_plotting)),
                                                   node_types=node_types, log="h1_Real")

        elif self.h == 1.5:

            self.communities_to_edges()

            X_graphs = []
            Y_graphs = []

            n_skipped_skip = 0
            n_skipped_target_df = 0
            n_not_skipped = 0

            dataset_size = 200 * self.batch_size

            if self.stage == "train":
                n_runs_considered = max([10, int(3 * self.n_samples / 5)])
            else:
                n_runs_considered = max([3, int(self.n_samples / 5)])
            min_nodes = 2
            max_ic_edge_class = self.max_ic_edge_class
            ic_edge_counts = []

            for n_part, model_partition in tqdm(enumerate(self.partitions_h2[:n_runs_considered])):
                if n_part > dataset_size:
                    continue
                joined_pairs = self.get_joined_pairs(model_partition)

                for pair in joined_pairs:
                    x1, x2 = pair[0], pair[1]

                    nodes1, nodes2 = model_partition[x1], model_partition[x2]

                    if len(nodes1) > self.max_size or len(nodes2) > self.max_size or len(nodes1) < min_nodes or len(
                            nodes2) < min_nodes:
                        n_skipped_skip += 1
                        continue

                    g1, g2, g3 = nx.subgraph(self.G, nodes1), nx.subgraph(self.G, nodes2), nx.subgraph(self.G, list(
                        nodes1) + list(nodes2))

                    separate_graph = nx.compose(g1, g2)
                    X_graphs.append(separate_graph)
                    Y_graphs.append(g3)

                    ic_edge_counts.append(g3.number_of_edges() - separate_graph.number_of_edges())

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

                if i in skip:  # or i not in target_df.index:
                    if i in skip:
                        n_skipped_skip += 1
                    else:
                        n_skipped_target_df += 1
                    continue
                else:
                    n_not_skipped += 1

                # try:
                #     nodelist = list(G.nodes())
                N = G.number_of_nodes()
                #     min_node = min(nodelist)
                # except:
                #     continue

                type_idx = []
                for node in list(G.nodes()):
                    node_type = node_type_df.at[node, "page_type"]
                    type_idx.append(types[node_type])

                previous_node_ids = list(G.nodes())
                G = nx.convert_node_labels_to_integers(G)

                node_mapping = {previous_node_ids[i]: list(G.nodes())[i] for i in range(len(previous_node_ids))}
                xG = nx.relabel_nodes(xG, mapping=node_mapping)
                # graphs[i] = G
                graphs_plotting.append(G)
                typedict = {idx: type_idx[idx] for idx in range(len(type_idx))}
                # print(typedict)
                node_types.append(typedict)

                # row, col, edge_type = [], [], []
                # for edge in list(G.edges()):
                #     start, end = edge[0], edge[1]
                #     row += [start, end]
                #     col += [end, start]
                #     # if edge in list(xG.edges()):
                #     #     edge_type += 2 * [1]
                #     # else:
                #     edge_type += 2 * [1]


                row, col, edge_type = [], [], []
                for edge in list(G.edges(data=True)):

                    start, end, edge_attr = edge[0], edge[1], edge[2]  # bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    # print(start, end, edge_attr)
                    row += [start, end]
                    col += [end, start]
                    if self.regime == "highway":
                        edge_type += 2 * [self.clean_edge_type_highway(edge_attr["highway"])]
                    else:
                        edge_type += 2 * [1]  # [bonds[bond.GetBondType()] + 1]

                component_graphs = nx.connected_components(xG)
                components = [list(c) for c in component_graphs]
                pos = []
                for node in list(G.nodes()):
                    # for ic, c in enumerate(components):
                    c = components[0]
                    if node in c:
                        pos += [1]
                    else:
                        pos += [0]

                pos = torch.tensor(pos, dtype=torch.bool)

                pos = self.get_intra_matrix(pos)
                # print(pos.shape)
                pos = F.pad(input=pos,
                            pad=(0, max_nodes - N, 0, max_nodes - N),
                            value=False)
                # print(pos.shape)
                # pos = F.one_hot(pos, num_classes = torch.unique(pos).shape[0]).to(torch.float)

                # print(f"Found {inter_counter} inter-community edges, and {len(edge_type)} edges total")

                edge_types.append({e: e for e in edge_type})

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)

                if self.regime == "highway":
                    num_edge_classes = 5
                else:
                    num_edge_classes = 2

                edge_attr = F.one_hot(edge_type, num_classes=num_edge_classes).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                try:
                    x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                except:
                    continue

                # these_node_types, counts = np.unique(type_idx, return_counts=True)
                # if counts.shape[0] <= 1:
                #     graph_type = these_node_types[0]
                # else:
                #     most_common_idx = np.argmax(counts)
                #     graph_type = these_node_types[most_common_idx]

                # TODO: apply this (glue classing by num ic edges) for fb dataset
                if self.regime == "size":
                    num_ic_edges = ic_edge_counts[i]

                    if num_ic_edges > max_ic_edge_class:
                        graph_type = max_ic_edge_class
                    else:
                        graph_type = num_ic_edges
                else:
                    graph_type = 1

                # print(f"Num ic edges: {self.num_ic_edges}"
                #       f"Assigned class: {graph_type}")
                # graph_type = self.gr  aph_classer(type_idx)

                y = F.one_hot(torch.tensor([graph_type]), num_classes=max_ic_edge_class + 1).float()

                # print(x.shape, pos.shape, edge_index.shape, edge_attr.shape)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i, pos=pos)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            # print(len(data_list) % self.batch_size)
            # print(len(data_list))
            # print(self.batch_size)
            # quit()
            # data_list = data_list[:-len(data_list) % self.batch_size]

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
                                              node_types=node_types, log="Y_Real")
                visualization_tools.visualize_grid(result_path, graphs_plotting, min(15, len(graphs_plotting)),
                                                   node_types=node_types, log="Community_Pairs_Y_Real",
                                                   largest_component=False)
            #
            #

            print(
                f"\n\n N_skipped from skiplist: {n_skipped_skip}\n N_skipped target_df: {n_skipped_target_df}\n N_not_skipped: {n_not_skipped}\n\n")
            # quit()


        elif self.h == 2:

            graphs_h2 = [self.make_meta_graph(partition) for partition in self.partitions_h2]

            # graphs_h2 = [entry[0] for entry in meta_g_informations]
            # edge_counts = [entry[1] for entry in meta_g_informations]

            densities = [nx.density(g) for g in graphs_h2]
            wandb.log({"Mean_Density": np.mean(densities),
                       "Max_Density": np.max(densities),
                       "Min_Density": np.min(densities),
                       "Dev_Density": np.std(densities),
                       "Mean_edges": np.mean([len(list(g.edges())) for g in graphs_h2]),
                       "Mean_nodes": np.mean([len(list(g.nodes())) for g in graphs_h2]),
                       "Mean_community_size": self.G.order() / np.mean([len(list(g.nodes())) for g in graphs_h2])})

            skip = []
            dataset_size = 200 * self.batch_size
            for i, G in enumerate(graphs_h2):
                if G.number_of_nodes() > self.max_size or i > dataset_size:
                    skip.append(i)

            suppl = tqdm(graphs_h2)

            data_list = []

            node_types = []
            edge_types = []
            graphs_h2_plotting = []
            max_ic_edge_class = self.max_ic_edge_class

            for i, G in enumerate(tqdm(suppl)):
                if i in skip:  # or i not in target_df.index:
                    continue

                # try:
                #     nodelist = list(G.nodes())
                N = G.number_of_nodes()
                #     min_node = min(nodelist)
                # except:
                #     continue

                this_partition = self.partitions_h2[i]

                # typedict = {}
                type_idx = []
                for node in list(G.nodes()):
                    community_nodes = this_partition[node]

                    if self.regime != "highway":
                        these_node_types = []
                        for n in community_nodes:
                            these_node_types.append(node_type_df.at[n, "page_type"])

                        node_type = self.graph_classer(these_node_types)

                    else:
                        edge_bunch = [(e[0], e[1], {"highway": e[2]}) for e in self.partitions_h1[node]]
                        ig = nx.from_edgelist(edge_bunch)

                        edge_type = []
                        for edge in list(ig.edges(data=True)):

                            edge_attr = edge[2]  # bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            edge_type += 2 * [self.clean_edge_type_highway(edge_attr["highway"])]

                        node_type = self.graph_classing_highway(edge_type) - 1


                    type_idx.append(node_type)
                # try:
                if self.regime == "highway":
                    x = F.one_hot(torch.tensor(type_idx).to(torch.int64), num_classes=4).float()
                else:
                    x = F.one_hot(torch.tensor(type_idx).to(torch.int64), num_classes=self.graph_class_dim).float()
                # except:
                #     continue

                G = nx.convert_node_labels_to_integers(G)
                graphs_h2_plotting.append(G)
                typedict = {idx: type_idx[idx] for idx in range(len(type_idx))}
                node_types.append(typedict)

                type_idx = []
                row, col, edge_type = [], [], []
                for edge in list(G.edges(data=True)):
                    start, end, ecount = edge[0], edge[1], edge[2]['weight']
                    row += [start, end]
                    col += [end, start]

                    if self.regime == "size":
                        etype = [np.min([ecount, max_ic_edge_class - 1])]
                    else:
                        etype = [1.]

                    # etype = [1]

                    edge_type += 2 * etype

                    type_idx.append(etype)

                typedict = {idx: type_idx[idx] for idx in range(len(type_idx))}
                edge_types.append(typedict)

                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type = torch.tensor(edge_type, dtype=torch.long)

                # print(edge_type, self.max_ic_edge_class)
                edge_attr = F.one_hot(edge_type, num_classes=self.max_ic_edge_class).to(torch.float)

                perm = (edge_index[0] * N + edge_index[1]).argsort()
                edge_index = edge_index[:, perm]
                edge_attr = edge_attr[perm]

                # Code now requires at least something like a graph class
                y = 1 # np.random.randint(2)
                y = F.one_hot(torch.tensor([y]), num_classes=2).float()
                # y = torch.zeros((1, 0), dtype=torch.float)

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

            # data_list = data_list[:-len(data_list) % self.batch_size]

            print(f"\nprocessed paths: {self.processed_paths}\nfile idx: {self.file_idx}\n")
            torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

            visualization_tools = TrainDiscreteNodeTypeVisualization()

            if self.stage == "train":
                # Visualize the final molecules
                current_path = os.getcwd()
                result_path = os.path.join(current_path,
                                           f'graphs/real_communities_h2')
                visualization_tools.visualize(result_path, graphs_h2_plotting, min(15, len(graphs_h2_plotting)),
                                              node_types=node_types, log="h2_real", node_size=50)
                visualization_tools.visualize_grid(result_path, graphs_h2_plotting, min(15, len(graphs_h2_plotting)),
                                                   node_types=node_types, log="h2_real")





        else:
            raise NotImplementedError(f"Hierarchy {self.h} is not implemented!")

    def get_intra_matrix(self, pos):

        # keep_matrix = torch.zeros(pos.shape, dtype = torch.bool)

        # E is shape (B * N * N * Ec)
        # node mask is same shape??

        n_nodes = pos.shape[0]
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

    def size_classing(self, type_idx, num_classes_out=5):
        """
        Returns the binning of graph size given mu, sigma
        Currently mu, sigma are calculated during initial partitioning

        type_idx: list(node class for each node)
        """

        num_nodes = len(type_idx)

        position = (num_nodes - self.mean_size) / self.dev_sizes
        shift_for_classes = int((num_classes_out - 1) / 2)
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

    def clean_edge_type_highway(self, type_string):
        types = {"residential":1., "tertiary":2., "secondary":3., "trunk":3, "primary":4.}
        numerical_type = None
        if type_string == "unclassified":
            numerical_type = 1.
            return numerical_type

        for k in types:
            if k in type_string:
                numerical_type = types[k]
                return numerical_type

        if numerical_type is None:
            return 1.

    def graph_classing_highway(self, edges):
        these_edge_types, counts = np.unique(edges, return_counts=True)
        if counts.shape[0] <= 1:
            graph_type = these_edge_types[0]
        else:
            most_common_idx = np.argmax(counts)
            graph_type = these_edge_types[most_common_idx]

        return graph_type







class ShetlandDataModule(AbstractDataModule):
    def __init__(self, cfg):
        print("Entered Shetland datamodule __init__")
        self.datadir = cfg.dataset.datadir
        super().__init__(cfg)
        self.h = cfg.dataset.h
        self.regime = cfg.dataset.regime
        print("Finished Shetland datamodule __init__")

    def prepare_data(self) -> None:
        # if self.h != 1.5:
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': ShetlandDataset(stage='train',
                                             root=root_path,
                                             h=self.h,
                                             resolution=self.cfg.dataset.resolution,
                                             max_size=self.cfg.dataset.max_size,
                                             n_workers=self.cfg.train.num_workers,
                                             batch_size=self.cfg.train.batch_size,
                                             regime=self.regime),
                    'val': ShetlandDataset(stage='val',
                                           root=root_path,
                                           h=self.h,
                                           max_size=self.cfg.dataset.max_size,
                                           n_workers=self.cfg.train.num_workers,
                                           batch_size=self.cfg.train.batch_size,
                                           regime=self.regime),
                    'test': ShetlandDataset(stage='test',
                                            root=root_path,
                                            h=self.h,
                                            max_size=self.cfg.dataset.max_size,
                                            n_workers=self.cfg.train.num_workers,
                                            batch_size=self.cfg.train.batch_size,
                                            regime=self.regime)}
        try:
            utils.general_graph_metrics(datasets["train"])
        except AttributeError:
            pass
        super().prepare_data(datasets)



class ShetlandDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = dataset_config.name
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = datamodule.node_types()  # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)