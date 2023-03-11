################################################################################
# DATA16001: Network Analysis (2023)
# Homework 2
# Boilerplate code for Exercise 2
# Last Updated: January 30, 2023
################################################################################


import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque

# Useful library that I used
from collections import defaultdict
import pandas as pd
import numpy as np

def load_signed_network(path):
    """Reads from file at path and return graph

    Args:
        path : path location of the csv edge list file

    Returns:
        G : NetworkX Directed Graph object
    """
    ############################################################################
    # TODO: Your code here!
    # NOTE: Graph is directed and stored in edgelist format.
    # Each line of the file has 4 numbers. For example
    # Source Target Rating Time
    # 1 2 4      1289241911.72836
    # The time attribute is NOT needed. 
    # The rating attribute is needed and is the weight of the edge (u,v)
    ############################################################################

    wh = pd.read_csv(path)
    wh.rename(columns={'Rating':'weight'},inplace=True)
    #print(wh.head())
    
    # Hint: load csv into pandas first. Then use load dataframe into networkx
    # use edge_attr to choose columns for edge weights
    #G = None

    G = nx.from_pandas_edgelist(wh,source='Source', target='Target', edge_attr ='weight',create_using=nx.DiGraph())
    #print(G)

    #print(nx.is_directed(G))

    assert nx.is_directed(G)
    return G

def get_asymm_edges_diffs(G):
    """Find the asymmetric edges in a directed graph. Also find the absolute difference in rating in asymmetric edges.

    An asymmetric edge is defined as a pair of nodes u, v where w_{u,v} != w_{v,u}.

    Args:
        G : NetworkX Directed Graph 
    
    Returns:
        asymmetric_edges : list of tuples (u,v) where w_{u,v} != w_{v,u}. Only keep one copy (u,v) for an asymmetric edge
        absolute_diffs : dict of counts of absolute difference of rating in asymmetric edges
    """
    ############################################################################
    # TODO: Your code here!
    ############################################################################
    assert nx.is_directed(G)
    asymmetric_edges = []
    # Note: process an asymmetric edge (u,v) ONLY ONCE and store one entry
    # Hint: keep absolute differences in a list and then use collections.Counter to get dict of counts
    absolute_diffs={}

    edges=G.edges()
    #print(edges)

    for edge in edges:
        if (edge[1],edge[0]) in edges and (edge[1],edge[0]) not in asymmetric_edges and edge not in asymmetric_edges:
            if G[edge[0]][edge[1]]['weight']!=G[edge[1]][edge[0]]['weight']:
                asymmetric_edges.append((edge[0],edge[1]))

                if abs(G[edge[0]][edge[1]]['weight']-G[edge[1]][edge[0]]['weight']) not in absolute_diffs:
                    absolute_diffs[abs(G[edge[0]][edge[1]]['weight']-G[edge[1]][edge[0]]['weight'])]=0
                absolute_diffs[abs(G[edge[0]][edge[1]]['weight']-G[edge[1]][edge[0]]['weight'])]+=1
    
    #print(asymmetric_edges)
    #print(absolute_diffs)

    return asymmetric_edges, absolute_diffs

def plot_absolute_diffs(absolute_diffs):
    
    fig,ax = plt.subplots()
    rects = ax.bar(absolute_diffs.keys(),absolute_diffs.values())
    ax.set_xticks(list(absolute_diffs.keys()))
    ax.set_yscale('log')
    ax.set_xlabel("Absolute Difference in ratings")
    ax.set_ylabel("Number of asymmetrical edges")
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    def autolabel(rects,ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects,ax)
    fig.tight_layout()
    #plt.show()
    plt.savefig("absolute_diffs.png", dpi=200)

def convert_undirected(G,asymmetric_edges):
    """Remove both directions of asymmetric edges and convert to undirected

    Args:
        G : NetworkX Directed Graph object
        asymmetric_edges : list of edges (u,v) where w_{u,v} != w_{v,u}

    Returns:
        G_und : NetowrkX Undirected Graph object
    """
    assert nx.is_directed(G)
    #######################################################
    # TODO: Your code here!
    #######################################################
    # Note: remove both (u,v) and (v,u) edges from G 
    # Then convert to undirected graph 
    #G_und = None
    #print(G)
    G_und = G.copy()

    G_und=G_und.to_undirected()
    G_und.remove_edges_from(asymmetric_edges)

    #print(G_und)

    assert not nx.is_directed(G_und)
    return G_und

def get_supernode_subgraphs(G):
    """A function that return that returns the supernodes.
    The supernodes are the connected components of the original graphs only considering positive edges between nodes.


    Args:
        G : NetworkX Undirected Graph

    Returns:
        supernodes : list of subgraphs induced by nodes in the supernodes]
        negative_edges : set of (u,v) for each negative edge in G
    """

    #######################################################
    # TODO: Your code here!
    #######################################################

    # get negative edges in the graph
    negative_edges = set()

    for edge in G.edges():
        if G[edge[0]][edge[1]]["weight"] <0:
            negative_edges.add(edge)


    # Removing negative edges from the copy
    positive_graph = G.copy()
    #print(positive_graph.edges())
    #print(list(negative_edges))

    positive_graph.remove_edges_from(negative_edges)

    # supernodes are the connected components in the positive graph
    #supernodes = None
    supernodes = []

    collecting=[]

    nodes=positive_graph.nodes()
    edges=dict.fromkeys(positive_graph.edges(),0)
    neighbours={}
    for node in nodes:
        if node not in neighbours:
            neighbours[node]=[n for n in nx.neighbors(positive_graph,node)]
    
    #print(edges)
    #print(neighbours)
        
    #print(positive_graph.nodes())
    #print(positive_graph.edges())

    for n1 in nodes:
        for n2 in neighbours[n1]:
            if (n1,n2) in edges:
                if len(collecting)==0:
                    collecting.append([])
                    collecting[0].append(n1)
                    collecting[0].append(n2)

                else:
                    check=0
                    for item in [n for n in collecting]:
                        if n1 in item and n2 not in item:
                            item.append(n2)
                            check+=1
                        if n1 not in item and n2 in item:
                            item.append(n1)
                            check+=1

                    if check==0:
                        collecting.append([])
                        collecting[-1].append(n1)
                        collecting[-1].append(n2)

    #print(collecting)
    
    for item in collecting:
        item.sort()

    #print(collecting)
    collecting.sort(key=len,reverse=True)
    #print(collecting)

    just=[]
    not_allowing_dublicate_list=[]

    for items in collecting:
        for item in items:
            check=0
            if item not in just:
                just.append(item)
                check+=1
            if check!=0:
                if items not in not_allowing_dublicate_list:
                    not_allowing_dublicate_list.append(items)
    
    just.sort()

    for n in G.nodes():
        if n not in just:
            not_allowing_dublicate_list.append([])
            not_allowing_dublicate_list[-1].append(n)

    #print(just)
    #print(not_allowing_dublicate_list)

    for item in not_allowing_dublicate_list:
        supernodes.append(nx.subgraph(G, item))

    #print(supernodes)
    #print(negative_edges)

    # Make sure return subgraphs and not just nodes of connected components
    assert all(isinstance(obj,nx.Graph) for obj in supernodes)
    return supernodes,negative_edges


def get_reduced_graph(supernodes,negative_edges):
    """Method creates a reduced graph from supernodes and negative edges from original graph

    Args:
        supernodes : list of supernodes subgraphs
        negative_edges : set of (u,v) of negative edges

    Returns:
        reduced_graph: graph where nodes correspond to supernodes and edges connecting supernodes
    """
    supernode_labels = ["s"+str(i) for i in range(len(supernodes))]
    #print(supernode_labels)

    mapping = {}
    for i,s in enumerate(supernodes):
        mapping.update(dict.fromkeys(s.nodes,supernode_labels[i]))
    #print(mapping)

    # remap negative edges (u,v) in original graph to (s_u,s_v)
    # s_u, s_v are labels for supernodes that u and v belong to respectively

    # create a set so that duplicate edges are counted only once 
    remapped_negative_edges = set(map(lambda e : tuple(map(mapping.get,e)),negative_edges))
    # verify there aren't any negative edges inside the same supernode
    # for eg, (s0,s0)
    intra_negative_edges = set((u,v) for u,v in remapped_negative_edges if u==v)
    assert len(intra_negative_edges) == 0

    # create reduced graph
    reduced_graph = nx.empty_graph()
    # add supernodes labels
    reduced_graph.add_nodes_from(supernode_labels)
    # add supernode edges 
    reduced_graph.add_edges_from(remapped_negative_edges)

    return reduced_graph 


def bfs_check(G,src):
    """ Function that performs Breadth First Search on the given graph from the specified source.
    Returns None if Odd Length Cycle is found in the graph.
    Otherwise return levels of nodes.

    Args:
        G : NetworkX undirected graph Object
        src : Node to begin BFS

    Returns:
        level: A dict mapping each node to a level corresponding to traversal depth
    """

    visited = dict.fromkeys(G.nodes,False)
    parents = dict.fromkeys(G.nodes,-1)
    level = dict.fromkeys(G.nodes,0)
    visited[src]=True

    #print(visited)
    #print(level)
    #print(parents)
    #print(G.edges())
    
    q = deque()
    q.append((src,parents[src]))

    #######################################################
    # TODO: Your code here!
    #######################################################
    
    while len(q)!=0:
        
        v=q.pop()

        #print(v)
        #print(v[0])
        #print(v[1])

        for s in sorted([n for n in nx.neighbors(G,v[0])]):
            #print(s)
            if not visited[s]:
                visited[s]=True
                level[s]=level[v[0]]+1
                q.append((s,parents[s]))
                parents[s] = v[0]
            
            elif parents[s] != v[0]:
                #print("Cycles")
                if abs(level[s]-level[v[0]]+1)%2==1:
                    print(level)
                    return None

    #print(visited)
    #print(level)

    # Begin your BFS loop here
    # Note: update the level dict during traversal
    # Check for odd length cycles 
    return level

def check_balance(G):
    """Method that checks the balance of the graph G using the supernode algorithm

    Args:
        G : NetworkX Undirected signed graph

    Returns:
        bipartite_mapping: dict of two keys corresponding to disjoin sets of the nodes of G if balanced. Returns None if unbalanced.
    """

    # balance only defined for directed graphs
    assert not nx.is_directed(G)

    # get supernodes of graph
    supernodes,negative_edges = get_supernode_subgraphs(G)

    # verify if supernodes are valid
    for s in supernodes:
        weights = set(w for _,_,w in s.edges(data="weight"))
        if -1 in weights:
            print("Supernode has internal negative edges")
            return None
    
    print(f"There are {len(supernodes)} supernodes and all are valid.")

    # obtain the reduced graph
    reduced_graph = get_reduced_graph(supernodes,negative_edges)

    levels = bfs_check(reduced_graph,"s0")
    #print(levels)

    if levels is None:
        print("Odd cycle found in reduced graph")
        return None
    
    # graph is balanced
    # get bipartitpe partition of the nodes of the original graph
    bipartite_mapping = defaultdict(list)
    for k,v in levels.items():
        # all supernode labels are s1,s2, ..., 
        supernode_id = int(k[1:])
        nodes = list(supernodes[supernode_id].nodes)
        if v % 2 == 0:
            bipartite_mapping["X"].extend(nodes)
        else:
            bipartite_mapping["Y"].extend(nodes)
    
    #print(bipartite_mapping)
    return bipartite_mapping

if __name__ == '__main__':
    # keep the data files in a subfolder called data

    simple_file = Path(__file__).parent / "data" / "simple.edges"

    simple_graph = nx.read_weighted_edgelist(simple_file,nodetype=int)

    #print(simple_graph)
    
    if check_balance(simple_graph) is None:
        print("Simple Graph is Unbalanced\n")
    else:
        raise ValueError("ERROR: Simple Graph should be unbalanced. Recheck code")
    
    # modify the edge (2,4) as positive and check for balance 
    simple_graph_modified = simple_graph.copy()
    simple_graph_modified[2][4]["weight"] = 1

    bipartite_mapping = check_balance(simple_graph_modified)
    if bipartite_mapping is None:
        raise ValueError("ERROR: Modified Simple Graph should be balanced. Recheck code")
    else:
        print("Bipartite mapping of modified simple graph nodes")
        print("Group X : ",bipartite_mapping["X"])
        print("Group Y: ",bipartite_mapping["Y"])
        print()

    bitcoin_file = Path(__file__).parent / "data" / "soc-sign-bitcoinotc.csv"
    if not bitcoin_file.exists():
        raise FileNotFoundError(f"Cannot find network file at {bitcoin_file.resolve()}. Please check path")
    bitcoin_directed = load_signed_network(bitcoin_file)

    asymmetric_edges, absolute_diffs = get_asymm_edges_diffs(bitcoin_directed)
    
    print(f"There are {len(asymmetric_edges)} pairs of nodes that have asymmetric edges")
    plot_absolute_diffs(absolute_diffs)

    # convert directed network to undirected 
    bitcoin_undirected = convert_undirected(bitcoin_directed,asymmetric_edges)

    #print(bitcoin_undirected)
    
    if check_balance(bitcoin_undirected):
        print("Bitcoin Network is balanced")
    else:
        print("Bitcoin Network is not balanced")