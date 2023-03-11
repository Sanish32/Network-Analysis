################################################################################
# DATA16001: Network Analysis (2023)
# Homework 2
# Boilerplate code for Exercise 1
# Last Updated: January 30, 2023
################################################################################


import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Useful library that I used
import itertools

def get_strong_weak_links(G,threshold):
    """
    A function that returns the list of weak and strong links in the graph based on the threshold of edge weight
    
    Args:
        G : NetworkX graph object
        threshold (float) : weight threshold for strong ties
    """
    ###############################################################################
    # TODO: your code here 
    weak_links=[]
    strong_links=[]

    #Idea 1 : this contains both (1,2) and (2,1) since the graph is undirected 
    #for u in G:
    #    for v in G:
    #        if (u,v) in G.edges:
                #print((u,v))
                #print(G[u][v]['weight'])

    #            if G[u][v]['weight']>=threshold:    
    #                strong_links.append((u,v))
    #            else:
    #                weak_links.append((u,v))
    
    #strong_links.sort()
    #weak_links.sort()

    #Idea 2 : using dictionary weight and this contains only (1,2) 
    weight=nx.get_edge_attributes(G, 'weight')
    #print(weight)

    for key,value in weight.items():
        if weight[key]>=threshold:
            strong_links.append(key)
        else:
           weak_links.append(key)

    ###############################################################################
    # Hint: get weights using nx.get_edge_attributes 

    #a=[]
    #b=[]
    #for item in weak_links:
    #    if (item[0],item[1]) not in a and (item[1],item[0]) not in a:
    #        a.append(item)

    #for item in strong_links:
    #    if (item[0],item[1]) not in b and (item[1],item[0]) not in b:
    #        b.append(item)

    #print(a)
    #print(b)

    #weak_links=a
    #strong_links=b
    
    return weak_links, strong_links


def draw_network_with_tie_strength(G,weak_links,strong_links):
    """
    Plots the network and links as described in exercise handout
    Args:
        G : NetworkX graph object 
        weak_links : list of tuples (u,v) of weak edges 
        strong_links : list of tuples (u,v) of strong edges

    """
    ############################################################################
    # TODO: Your code here!
    ############################################################################
    # get layout 
    layout=nx.circular_layout(G)

    # draw nodes
    nx.draw_networkx_nodes(G,pos=layout)

    # draw node labels 
    nx.draw_networkx_labels(G, pos=layout)

    # draw strong links
    nx.draw_networkx_edges(G, pos=layout, edgelist=strong_links, edge_color='red', style='solid')

    # draw weak links 
    nx.draw_networkx_edges(G, pos=layout, edgelist=weak_links, edge_color='blue', style='dashed')

    plt.title("Barn Swallow Contact Network")
    plt.show()

def check_stcp(G,weak_links,strong_links):
    """
    Checks the Strong Triadic Closure Property of all nodes in the graph


    Args:
        G : NetworkX grpah object
        weak_links : list of tuples (u,v) of weak edges
        strong_links : list of tuples (u,v) of strong edges

    Returns:
        dict{node:boolean}: a dict of nodes with True or False for satisfying STPC  
    """

    ############################################################################
    # TODO: Your code here!
    ############################################################################
    a=[]
    b=[]
    for item in weak_links:
        if (item[0],item[1]) not in a and (item[1],item[0]) not in a:
            a.append(item)
            a.append((item[1],item[0]))

    for item in strong_links:
        if (item[0],item[1]) not in b and (item[1],item[0]) not in b:
            b.append(item)
            b.append((item[1],item[0]))

    #print(a)
    #print(b)

    weak_links=a
    strong_links=b


    stcp_validity = {}
    nodes=sorted(G.nodes())
    #print(nodes)

    collecting=[]
    for u in nodes:
        for v in nodes:
            if (u,v) in strong_links:
                collecting.append(v)

        #print(collecting)

        #The condition for at least 2 nodes is implemented here
        checking=list(itertools.combinations(collecting,2))
        #print(checking)

        for item in sorted(checking):
            if item in weak_links or item in strong_links:
                if u not in stcp_validity:
                    stcp_validity[u]=True
            else:
                stcp_validity[u]=False

        collecting.clear()
    
    for n in nodes:
        if n not in stcp_validity:
            stcp_validity[n]=True
                
    # Note: G is undirected, so edge (u,v) is same as (v,u).
    # Hint: use itertools.combinations to get all pairs of two edges of nodes 

    return stcp_validity 

if __name__ == '__main__':
    network_file = Path(__file__).parent / "data" / "aves-barn-swallow-contact-network.edges"
    # keep the network file in a subfolder called data
    if not network_file.exists():
        raise FileNotFoundError(f"Cannot find network file at {network_file.resolve()}. Please check path.")

    aves = nx.read_weighted_edgelist(network_file,nodetype=int)

    #Printing number of nodes and edges
    #print(aves)

    # Choose threshold from exercise 
    weak_links, strong_links = get_strong_weak_links(aves,threshold=3)
    #print(weak_links)
    #print(strong_links)

    draw_network_with_tie_strength(aves,weak_links,strong_links)

    stcp_validity = check_stcp(aves,weak_links,strong_links)

    #print(stcp_validity)

    if sum(stcp_validity.values()) == len(stcp_validity):
        print("All nodes satisfy STCP property")
    else:
        stcp_violaters = [n for n,validity in stcp_validity.items() if not validity]
        print(f"Nodes {stcp_violaters} violate Strong Triadic Closure Property")
    
