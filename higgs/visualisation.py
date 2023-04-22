import networkx as nx
import matplotlib.pyplot as plt
import wandb

def vis_h1_group(h1_adjs,  ax = None):

    G = nx.Graph()

    for i, adj in enumerate(h1_adjs):
        g = nx.from_numpy_array(adj.cpu().numpy())
        g = nx.convert_node_labels_to_integers(g, first_label=G.order() + 1) # h1_label_mapping[i]["start_ind"])

        G = nx.compose(G, g)

    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="sfdp", args='-Gsmoothing')

    if ax is None:
        was_given_ax = False
        fig, ax = plt.subplots(figsize=(7,7))
    else:
        was_given_ax = True

    nx.draw_networkx_edges(G, node_size=2, pos=pos, alpha=0.5, ax=ax)

    if was_given_ax:
        return ax, G

    else:
        ax.set_title(r"Sampled $h_1$ Graphs")
        plt.tight_layout()
        plt.savefig("h1_graphviz.png", dpi=300)

        im = plt.imread("h1_graphviz.png")
        wandb.log({"Full Sampling/h1_Graphviz": [wandb.Image(im, caption="Un-Coloured-SFDP, h1")]})

        return G

def vis_glue_group(glue_adjs,  ax = None):

    G = nx.Graph()

    for i, g in enumerate(glue_adjs):
        # g = nx.from_numpy_array(adj)

        CGs = [g.subgraph(c) for c in nx.connected_components(g)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        g = CGs[0]

        g = nx.convert_node_labels_to_integers(g, first_label=G.order() + 1)

        G = nx.compose(G, g)

    print(f"Finding big glue positions with {G.order()} nodes")
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="sfdp", args='-Gsmoothing')

    if ax is None:
        was_given_ax = False
        fig, ax = plt.subplots(figsize=(7,7))
    else:
        was_given_ax = True

    G.remove_edges_from(nx.selfloop_edges(G))

    nx.draw_networkx_edges(G, node_size=2, pos=pos, alpha=0.5, ax=ax)

    if was_given_ax:
        return ax

    else:
        ax.set_title(r"Sampled Edge Predictions Graphs")
        plt.tight_layout()
        plt.savefig("glue_graphviz.png", dpi=300)

        im = plt.imread("glue_graphviz.png")
        wandb.log({"Full Sampling/Glue_Graphviz": [wandb.Image(im, caption="Un-Coloured-SFDP, glue")]})

        return G




def vis_big_graphs(original_G, G, cg, graph_number = 0):
    pos_real = nx.drawing.nx_agraph.graphviz_layout(original_G, prog="sfdp", args='-Gsmoothing')
    pos_sample = nx.drawing.nx_agraph.graphviz_layout(G, prog="sfdp", args='-Gsmoothing')
    pos_cg = nx.drawing.nx_agraph.graphviz_layout(cg, prog="sfdp", args='-Gsmoothing')

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 6))

    nx.draw_networkx_edges(original_G, node_size=2, pos=pos_real, alpha=0.5, ax=ax1)
    nx.draw_networkx_nodes(original_G, node_size=1, pos=pos_real, ax=ax1,
                           node_color=[node[1]["target"] for node in original_G.nodes(data=True)])

    nx.draw_networkx_edges(G, node_size=2, pos=pos_sample, ax=ax2, alpha=0.5)
    nx.draw_networkx_nodes(G, node_size=2, pos=pos_sample, ax=ax2,
                           node_color=[node[1]["target"] for node in G.nodes(data=True)])

    nx.draw_networkx_edges(cg, node_size=2, pos=pos_cg, alpha=0.5, ax=ax3)
    nx.draw_networkx_nodes(cg, node_size=2, pos=pos_cg, ax=ax3,
                           node_color=[node[1]["target"] for node in cg.nodes(data=True)])

    ax1.set_title("Original")
    ax2.set_title(r"Synthetic, all CCs")
    ax3.set_title("Synthetic, Largest CC")

    plt.tight_layout()
    plt.savefig(f"sample_vs_real_graphviz_{graph_number}.png", dpi=300)

    im = plt.imread(f"sample_vs_real_graphviz_{graph_number}.png")
    wandb.log({f"Full Sampling/Full_Sampled_Raw_Graphviz": [wandb.Image(im, caption=f"Un-Coloured-SFDP_graph_{graph_number}")]})