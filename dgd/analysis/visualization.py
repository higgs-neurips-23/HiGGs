import os

import imageio
import networkx as nx
import numpy as np
import wandb
import matplotlib.pyplot as plt

from community_layout.layout_class import CommunityLayout
from statistics import mode


def LargeGraphVisualization(G, partition):
    # Use community-layout to (quickly) layout and draw a large graph
    layout = CommunityLayout(G, community_algorithm=partition)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = layout.display(ax=ax)
    plt.tight_layout()
    current_path = os.getcwd()
    result_path = os.path.join(current_path,
                               f'graphs/train_communities/all_communities.png')
    plt.savefig(result_path, dpi=300)



class NonMolecularVisualization:
    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()
        print(node_list)
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

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=20, largest_component=True):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        # Set node colors based on the eigenvectors
        w, U = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).toarray())
        vmin, vmax = np.min(U[:, 1]), np.max(U[:, 1])
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        plt.figure()
        nx.draw(graph, pos, font_size=5, node_size=node_size, with_labels=False, node_color=U[:, 1],
                cmap=plt.cm.coolwarm, vmin=vmin, vmax=vmax, edge_color='grey')

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph', trainer=None):
        # TODO: implement the multi-gpu case
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        # convert graphs to networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, fps=5)
        wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
        return


class DiscreteNodeTypeVisualization:
    def to_networkx(self, node_list, adjacency_matrix):
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

    def visualize_non_molecule(self, graph, pos, path, iterations=100,
                               node_size=100, largest_component=True, ax=None, label=None):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        if graph.order() > 500:
            node_size = 5
            figsize = (15, 15)
            iterations = 10

            print(f"Found large {graph}")
        else:
            node_size = 20
            figsize = (6, 6)
        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        nodelist = list(graph.nodes())
        colors = [graph.nodes[node]["color_val"] for node in nodelist]

        if np.unique(colors).shape[0] == 1:
            node_size = 2

        vmin, vmax = np.min(colors), np.max(colors)
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        edgelist = list(graph.edges(data=True))

        if "color" in edgelist[0][2]:
            ecolors = [edge[2]["color"] for edge in edgelist]
        elif "weight" in edgelist[0][2]:
            ecolors = [edge[2]["weight"] for edge in edgelist]
        elif "highway" in edgelist[0][2]:
            ecolors = [clean_edge_type_highway(edge[2]["highway"]) for edge in edgelist]
        else:
            ecolors = [1 for edge in edgelist]

        if "highway" not in edgelist[0][2]:
            if len(set(ecolors)) > 1:
                evmin, evmax = np.min(ecolors), np.max(ecolors)
        else:
            evmin, evmax = 1, 5

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            was_given_ax = False
        else:
            was_given_ax = True

        if label is not None:
            ax.set_title(f"Cond: {label}, Maj: {mode(colors)}")

        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=colors, vmin=vmin, vmax=vmax, ax=ax,
                               edgecolors="black", cmap="viridis")

        if len(set(ecolors)) > 1:
            nx.draw_networkx_edges(graph, pos, node_size=node_size, edge_color=ecolors,
                                   edge_vmin=evmin, edge_vmax=evmax, ax=ax, width=ecolors)
        else:
            nx.draw_networkx_edges(graph, pos, node_size=node_size,
                                   ax=ax)

        if was_given_ax:
            pass
        else:
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph', trainer=None):
        # TODO: implement the multi-gpu case
        # define path to save figures
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except:
            pass
        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            # file_path = os.path.join(path, 'graph_{}.png'.format(i))
            file_path = os.path.join(path, f"{log}_graph_{i}.png")
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            try:
                im = plt.imread(file_path)
                wandb.log({log: [wandb.Image(im, caption=file_path)]})
            except:
                pass

    def visualize_grid(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph', trainer=None,
                       labels=None, largest_component = False):
        # TODO: implement the multi-gpu case
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        if len(graphs) >= 3:
            nrows = 3
            ncols = int(np.around(num_graphs_to_visualize / 3))

            if ncols == 0:
                return
            fig, axes = plt.subplots(figsize=(ncols * 2, nrows * 2), nrows=nrows, ncols=ncols)
            try:
                axes = [ax for sublist in axes for ax in sublist]
            except:
                pass
        elif len(graphs) == 1:
            fig, ax = plt.subplots(figsize=(4,4))
            axes = [ax]
        elif len(graphs) == 2:
            fig, axes = plt.subplots(ncols=2, figsize=(4,4))
        else:
            return

        for ax in axes:
            ax.set_axis_off()

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            try:
                ax = axes[i]
            except:
                continue
            if labels is not None:
                label = labels[i]
            else:
                label = None
            file_path = os.path.join(path, f"{log}_graph_{i}.png")
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            ax = self.visualize_non_molecule(graph=graph, pos=None, path=file_path, ax=ax, label=label, largest_component=largest_component)

        file_path = os.path.join(path, 'graph_grid.png')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close("all")
        im = plt.imread(file_path)
        wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix, trainer=None):
        # convert graphs to networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]

        final_pos = nx.spring_layout(final_graph, seed=0)
        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            try:
                self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            except:
                continue
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, fps=5)
        wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
        return


class TrainDiscreteNodeTypeVisualization:

    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        CALLED DURING DATA PREPARATION
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

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=20, largest_component=True, ax=None):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            pos = nx.spring_layout(graph, iterations=iterations)

        nodelist = list(graph.nodes())
        colors = [graph.nodes[node]["color_val"] for node in nodelist]

        if np.unique(colors).shape[0] == 1:
            node_size = 2

        vmin, vmax = np.min(colors), np.max(colors)
        m = max(np.abs(vmin), vmax)
        vmin, vmax = -m, m

        edgelist = list(graph.edges(data=True))

        if "color" in edgelist[0][2]:
            ecolors = [edge[2]["color"] for edge in edgelist]
        elif "weight" in edgelist[0][2]:
            ecolors = [edge[2]["weight"] for edge in edgelist]
        elif "highway" in edgelist[0][2]:
            ecolors = [clean_edge_type_highway(edge[2]["highway"]) for edge in edgelist]
        else:
            ecolors = [1 for edge in edgelist]

        if len(set(ecolors)) > 1:
            evmin, evmax = np.min(ecolors), np.max(ecolors)

        if ax is None:
            fig, ax = plt.subplots()
            was_given_ax = False
        else:
            was_given_ax = True

        if node_size > 15:
            alpha = 0.25
        else:
            alpha = 0.75

        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=colors, vmin=vmin, vmax=vmax, ax=ax,
                               edgecolors="black", cmap="viridis")

        if len(set(ecolors)) > 1:
            nx.draw_networkx_edges(graph, pos, node_size=node_size, edge_color=ecolors,
                                   edge_vmin=evmin, edge_vmax=evmax, ax=ax, alpha=alpha, width=ecolors)
        else:
            nx.draw_networkx_edges(graph, pos, node_size=node_size,
                                   ax=ax, alpha=alpha)

        if was_given_ax:
            pass
        else:
            plt.tight_layout()
            plt.savefig(path, dpi=300)
            plt.close("all")

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, node_types=None,
                  edge_types=None, log='graph', trainer=None, largest_component=True, node_size=20):
        # TODO: implement the multi-gpu case
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)
        graphs = self.add_attributes(graphs, node_types=node_types, edge_types=edge_types)
        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, f"{log}_graph_{i}.png")
            graph = graphs[i]

            self.visualize_non_molecule(graph=graph, pos=None, path=file_path, largest_component=largest_component,
                                        node_size=node_size)
            im = plt.imread(file_path)
            wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_grid(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph',
                       node_types=None, edge_types=None, trainer=None, largest_component=True):
        # TODO: implement the multi-gpu case
        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        nrows = 3
        ncols = int(np.around(num_graphs_to_visualize / 3, decimals=0))

        fig, axes = plt.subplots(figsize=(ncols * 2, nrows * 2), nrows=nrows, ncols=ncols)

        try:
            axes = [ax for sublist in axes for ax in sublist]
        except:
            pass

        for ax in axes:
            ax.set_axis_off()

        graphs = self.add_attributes(graphs, node_types=node_types, edge_types=edge_types)
        # visualize the final molecules
        for i in range(num_graphs_to_visualize - 1):
            ax = axes[i]
            graph = graphs[i]
            file_path = os.path.join(path, f"{log}_graph_{i}.png")
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path, ax=ax,
                                        largest_component=largest_component)
        file_path = os.path.join(path, 'graph_grid.png')
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close("all")
        im = plt.imread(file_path)
        wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def add_attributes(self, graphs, node_types=None, edge_types=None):

        for i, G in enumerate(graphs):
            nodelist = list(G.nodes())
            edgelist = list(G.edges())

            if node_types is not None:
                typedict = node_types[i]
            else:
                typedict = {i: 0 for i in range(len(nodelist))}
            for n in nodelist:
                G.nodes[n]["color_val"] = typedict[n]

        return graphs

def clean_edge_type_highway(type_string):
    types = {"residential":1., "tertiary":2., "secondary":3., "primary":4.}
    numerical_type = None
    if type_string == "unclassified":
        numerical_type = 1.
        return numerical_type

    for k in types:
        if k in type_string:
            numerical_type = types[k]
            return numerical_type

    if numerical_type is None:
        print(f"Unrecognised type {type_string}")
        return 1.