"""
anonymous authors' implementation of BTER,

Community structure and scale-free collections of Erdos-Rényi graphs
Seshadhri CKolda TPinar A
Physical Review E - Statistical, Nonlinear, and Soft Matter Physics, 2012
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import networkx.algorithms.community  as comm


class BTER:
    def __init__(self, G_original):
        """
        Initialise BTER and store relevant parameters

        G_original: A networkx.Graph to which BTER will be fit
        """

        self.G_original = G_original
        self.all_degrees = list(self.G_original.degree[n] for n in self.G_original.nodes())
        self.d_max = np.max(self.all_degrees)
    def fit(self):
        """
        Use the degree distribution from initialisation to calculate community sizes.
        """
        # Degrees sorted in ascending order
        sorted_degrees = sorted(self.all_degrees)

        self.community_sizes = []
        # Number of communities is not determined before looping
        while len(sorted_degrees) > 0:
            # As in paper each community size is determined by the degree of
            # the node at the top of the stack
            comm_size = sorted_degrees[0]

            if comm_size > len(sorted_degrees):
                self.community_sizes.append(len(sorted_degrees))
                break

            # High degree nodes are likely to have a degree higher
            # than the number of nodes left in the stack
            else:
                sorted_degrees = sorted_degrees[comm_size:]
                self.community_sizes.append(comm_size)

        # These parameters are fixed to match the paper from Seshadri et al.
        self.rho = 0.7 # np.median([nx.density(g) for g in communities])
        self.eta = 1.25


    def sample(self):
        """
        Samples a graph using the parameters set during initialisation
        calls BTER.excesses, BTER.attach_one_degree_vertices, BTER.add_inter_edges, BTER.add_intra_edges
        """
        sorted_degrees = sorted(self.all_degrees)

        # Build nested list of the degrees of the nodes in each community
        community_nodes_degrees = []
        for n_nodes in self.community_sizes:
            community_degrees = []
            for _ in range(n_nodes):
                # Remove node from top of stack and add to this communities degrees
                degree = sorted_degrees.pop(0)
                community_degrees.append(degree)

            community_nodes_degrees.append(community_degrees)

        # Need to add nodes from each community to nx.Graph
        # This makes the next few stages much simpler, avoids re-indexing etc
        node_partitions = {i:[] for i in range(len(community_nodes_degrees))}
        G_sampled = nx.Graph()
        node_counter = 0
        for i, community in enumerate(community_nodes_degrees):
            for d in community:
                G_sampled.add_node(node_counter)
                node_partitions[i].append(node_counter)
                node_counter += 1


        # Calculate rho for each community then use
        # BTER.add_intra_edges to construct ER graphs for each community
        # Many nodes will have left over degrees
        community_nodes_rhos = []
        for i, degrees in enumerate(community_nodes_degrees):
            nodes = node_partitions[i]
            rho = self.scaled_edge_prob(degrees)
            community_nodes_rhos.append([rho for _ in degrees])

            G_sampled, new_node_degrees = self.add_intra_edges(G_sampled, nodes, degrees, rho)
            community_nodes_degrees[i] = new_node_degrees

        # Add inter-community edges based on excess degrees for the nodes in each community
        G_sampled = self.add_inter_edges(G_sampled, community_nodes_degrees, community_nodes_rhos)

        return G_sampled

    def excesses(self, degrees, rhos, size_reference):
        """
        Calculate excesses for the nodes for a given community

        degrees: list of node degrees (left over)
        rhos:    rhos calculated previously for each node
        size_reference: list of the size of the community for each node,
                        ie size_reference[n]: size of the community containing node n
        """

        excesses = [d - rhos[i] * (size_reference[i] - 1) for i, d in enumerate(degrees)]
        for i, e in enumerate(excesses):
            # Catch one degree nodes and zero out where excesses are 0
            if size_reference[i] == 1:
                excesses[i] = 1.
            elif excesses[i] < 0:
                excesses[i] = 1e-30
            # print(excesses[i], degrees[i], rhos[i])

        return excesses

    def attach_deg_one_vertices(self, G_sampled, degrees, rhos, size_reference):
        """
        One-degree vertices are treated differently in bter

        params:
        param: G_sampled: networkx.Graph edges are added to
        param: degrees: array of remaining node degrees to be added
        param: rhos: array of parameter rho for each node in the graph, used to calculate excesses
        param: size_reference: array of community sizes for each node

        returns:
        param: G_sampled: networkx.Graph edges are added to
        param: degrees: array of remaining node degrees to be added
        param: rhos: array of parameter rho for each node in the graph, used to calculate excesses
        param: size_reference: array of community sizes for each node
        """
        num_one_degree = np.sum(np.array(size_reference) <= 1)
        num_manually_set_excesses = int(0.75 * num_one_degree) # AKA p, parameter from paper

        # Gets identifiers for nodes that are treated differently
        manual_reference = np.zeros(len(size_reference))
        manual_reference[:num_manually_set_excesses] = 1

        # manual reference specifies which nodes are ready to attach, as in paper
        node_excesses = self.excesses(degrees, rhos, size_reference)

        # Until all nodes have 0 excess
        while np.sum(degrees[:num_manually_set_excesses] > 0) > 1:
            # Get probabilities for each node n1, n2 in new edge [n1, n2]
            # n1 is a chosen degree-one node
            probabilities_1 = node_excesses[:num_manually_set_excesses] / np.sum(
                node_excesses[:num_manually_set_excesses])
            probabilities_2 = node_excesses[num_manually_set_excesses:] / np.sum(
                node_excesses[num_manually_set_excesses:])

            # ie these are the options - poss_1 is degree-one, poss_2 is some other node
            possibilities = np.arange(degrees.shape[0])
            possibilities_1 = possibilities[:num_manually_set_excesses]
            possibilities_2 = possibilities[num_manually_set_excesses:]

            # Get real probabilities for each
            probabilities_1 = probabilities_1[degrees[:num_manually_set_excesses] > 0]
            probabilities_1 = probabilities_1 / np.sum(probabilities_1)
            possibilities_1 = possibilities_1[degrees[:num_manually_set_excesses] > 0]

            probabilities_2 = probabilities_2[degrees[num_manually_set_excesses:] > 0]
            probabilities_2 = probabilities_2 / np.sum(probabilities_2)
            possibilities_2 = possibilities_2[degrees[num_manually_set_excesses:] > 0]

            # Randomly chose n1, n2 from the options
            selection = [np.random.choice(possibilities_1, 1, p=probabilities_1)[0],
                         np.random.choice(possibilities_2, 1, p=probabilities_2)[0]]

            # Add that as an edge
            G_sampled.add_edge(selection[0], selection[1])

            # Alter degrees accordingly
            degrees[selection[0]] -= 1
            degrees[selection[1]] -= 1

            # Re-calculate excesses
            node_excesses = self.excesses(degrees, rhos, size_reference)

        return G_sampled, degrees, rhos, size_reference


    def add_inter_edges(self, G_sampled, node_degrees, node_rhos):
        """
        Add inter-community edges - the last stage of BTER

        params:
        param: G_sampled: networkx.Graph edges are added to
        param: rhos: array of parameter rho for each node in the graph, used to calculate excesses
        param: node_degrees: array of remaining node degrees to be added

        returns:
        param: G_sampled: networkx.Graph, final with all nodes and edges
        """
        size_reference = []
        for size in self.community_sizes:
            size_reference += [size] * size

        node_degrees  = np.array([num for sublist in node_degrees for num in sublist])
        node_rhos     = np.array([num for sublist in node_rhos for num in sublist])

        G_sampled, degrees, rhos, size_reference = self.attach_deg_one_vertices(G_sampled, node_degrees, node_rhos, size_reference)

        node_excesses = self.excesses(node_degrees, node_rhos, size_reference)

        while np.sum(node_degrees > 0) > 1:
            probabilities = node_excesses / np.sum(node_excesses)
            possibilities = np.arange(node_degrees.shape[0])

            probabilities = probabilities[node_degrees > 0]
            probabilities = probabilities / np.sum(probabilities)
            possibilities = possibilities[node_degrees > 0]
            selection = [np.random.choice(possibilities, 1, p=probabilities)[0],
                         np.random.choice(possibilities, 1, p=probabilities)[0]]

            G_sampled.add_edge(selection[0], selection[1])

            node_degrees[selection[0]] -= 1
            node_degrees[selection[1]] -= 1

            node_excesses = self.excesses(node_degrees, node_rhos, size_reference)

        return G_sampled

    def add_intra_edges(self, G_sampled, nodes, node_degrees, rho):
        """
        Build ER graph for each community

        params:
        param: G_sampled: networkx.Graph
        param: nodes: list of node ids in this community
        param: node_degrees: array of degrees still to be added to each node in the community
        rho: float parameter for each community

        returns:
        param: G_sampled: networkx.Graph with a seperate component for each community
        param: node_degrees: array of node available degrees adjusted for edges within communities
        """
        # Catch single nodes being passed
        if len(node_degrees) <= 1:
            return G_sampled, node_degrees

        else:
            # Iterate over every node pair in the community
            for i1, node_1 in enumerate(nodes):
                for i2, node_2 in enumerate(nodes):
                    # If a random number exceeds a probability, and that node has the option to gain another edge
                    if np.random.random() < rho and node_degrees[i1] > 0 and node_degrees[i2] > 0:
                        # Add edge and subtract 1 from available node degrees
                        G_sampled.add_edge(node_1, node_2)
                        node_degrees[i1] -= 1
                        node_degrees[i2] -= 1
            return G_sampled, node_degrees

    def scaled_edge_prob(self, degrees):
        """
        Rescale the probability of an edge according to the paper's equation

        params:
        param: degrees: array of node degrees

        returns:
        param: adjusted_prob: float, adjusted probability for edge existence
        """
        d_bar = np.min(degrees)
        adjusted_prob = self.rho * (1 - self.eta*(np.log(d_bar + 1) / np.log(self.d_max + 1))**2)

        return adjusted_prob


if __name__ == "__main__":
    G = nx.erdos_renyi_graph(int(1e3), 0.01)
    bter = BTER(G)
    bter.fit(1)
    G_sampled = bter.sample()

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
    nx.draw_networkx_edges(G, pos = nx.spring_layout(G), ax = ax1)
    nx.draw_networkx_edges(G_sampled, pos = nx.spring_layout(G_sampled), ax = ax2)
    plt.show()