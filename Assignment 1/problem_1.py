#%%
################################################################################
# DATA16001: Network Analysis (2023)
# Homework 1
# Boilerplate code for Problem 1
# Last Updated: January 19, 2023
################################################################################

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Useful library that I used
import math
import random

#%%


def generate_G_nm(n:int, m:int,seed:int=42) -> nx.Graph:
    """Return an instance of Erdos-Renyi graph

    Args:
        n (int): number of nodes
        m (int): number of edges

    Returns:
        nx.Graph: NetworkX Graph object
    """
    ############################################################################
    # TODO: Your code here!
    graph=nx.Graph()
    graph=nx.gnm_random_graph(n,m,seed)
    
    #print(graph)
    ############################################################################
    return graph

#%%
def generate_B_nm(n:int, m:int,seed:int=42) -> nx.Graph:
    """Return an instance of the B(n,m) graph

    Args:
        n (int): number of nodes
        m (int): number of edges

    Returns:
        nx.Graph: NetworkX Graph object
    """
    ############################################################################
    # TODO: Your code here!
    k=math.ceil(m/n)
    
    G=nx.Graph()
    G.add_node(1)

    #k+1 nodes means in range, I need to write k+2 for running for loop which includes k+1 nodes
    for i in range(2,k+2):
        G.add_node(i)

    #print(G)


    #k+1 means in range, I need to write k+2 for running for loop which includes k+1 edges
    for i in range(2,k+2):
        G.add_edge(1,i)

    #Since it starts from k+2, my logic was correct
    #n in range means that I need to make my loop work for n times so need to write n+1 in range section
    
    for i in range(k+2,n+1):  
        count=0

        for u in G:
            count+=G.degree[u]
        
        prob=[]

        for u in G:
            prob.append(G.degree[u]/count)

        U=np.random.choice(G.nodes(),size=k,replace=False,p=prob)
            
        for u in U:
            G.add_node(i)
            G.add_edge(i,u)
    
    #print(G.nodes())
    #print(G.edges())


    ##Before adjusting the edges
    #print("node:",G.number_of_nodes())
    #print("edge:",G.number_of_edges())

    #print(G)

    #print(G.number_of_edges)
    ensuring=G.number_of_edges()-m

    #print(G.number_of_edges())
    #print(ensuring)
    #print(m)

    adjusting=0
    while adjusting!=ensuring:
        choicing=random.choice(list(G.edges()))
        print(choicing)
        if G.has_edge(choicing[0],choicing[1]):
            G.remove_edge(choicing[0],choicing[1])
            adjusting+=1
  
    #print("node:",G.number_of_nodes())
    #print("edge:",G.number_of_edges())

    # NOTE: Add edges based on the pseudocode
    ############################################################################
    #assert G.number_of_edges() == m
    #assert G.number_of_nodes() == n
    return G

#%%
def load_real_world_network(name="air_traffic") -> nx.Graph:
    """
    Read from file and return ``name`` graph.

    INPUT:
    - ``name`` -- name of real-world network

    OUTPUT:
    - NetworkX graph object

    """
    ############################################################################
    # TODO: Your code here!
    dict={}
    G=nx.Graph()

    with open(name+".edgelist") as new_file:
        for line in new_file:
            if " " in line:
                parts=line.split(" ")
            elif "\t" in line:
                parts=line.split("\t")
            parts[0]=parts[0].strip("\n")
            parts[1]=parts[1].strip("\n")
            if parts[0] not in dict:
                dict[parts[0]]=[]
                G.add_node(parts[0])
            if parts[1] not in dict:
                dict[parts[1]]=[]
                G.add_node(parts[1])
            dict[parts[0]].append(parts[1])
            dict[parts[1]].append(parts[0])
            G.add_edge(parts[0], parts[1])
            G.add_edge(parts[1],parts[0])
        
    # NOTE: You will have to provide path to data file. 
    # Graph is stored in edgelist format. 
    # Each line of the file has two numbers separated by a space and represents 
    # an edge between the two nodes.
    ############################################################################
    G.name = name
    return G
#%%

def compute_clustering_coefficient(graph:nx.Graph)->float:
    """
    Compute average clustering coefficient of ``graph``.

    INPUT:
    - ``graph`` -- NetworkX graph object

    OUTPUT:
    - average clustering coefficient of ``graph`` (type: float)

    """
    ############################################################################
    # TODO: Your code here!
    cv=0

    for node in graph.nodes():
        neighbours=[n for n in nx.neighbors(graph,node)]
        n_neighbors=len(neighbours)
        n_links=0
        setting=[]


        if n_neighbors>1:
            for node1 in neighbours:
                for node2 in neighbours:
                    if graph.has_edge(node1,node2):
                        n_links+=1
                        if (node1,node2) not in setting:
                            setting.append((node1,node2))
                            setting.append((node2,node1))
                            

            #n_links/=2 #because n_links is calculated twice
        if n_neighbors>1:
            clustering_coefficient=(n_links)/(n_neighbors*(n_neighbors-1))
        elif n_neighbors==0 or n_neighbors==1:
            clustering_coefficient=0
        cv+=clustering_coefficient
        setting.clear()
        
    #print(cv)
    
    C=cv/len(graph.nodes())
    #print(C)

    #print(nx.average_clustering(graph))

    
    # NOTE: You must implement your own method here. 
    # You may ony use the in-built NetworkX method to verify your answer.
    ############################################################################
    return C

#%%
def get_degree_distribution(graph:nx.Graph):
    """
    Return degree values and cumulative count of number of nodes for each degree value.

    INPUT:
    - ``graph`` -- NetworkX graph object

    OUTPUT:
    - ``x_graph`` -- degree values of nodes in ``graph`` in sorted order (type: list)
    - ``y_graph`` -- number of nodes of degree `d` for all degree values `d` (type: list)
    """
    ############################################################################
    # TODO: Your code here!
    x_graph=[]
    y_graph=[]

    for node in graph.nodes():
        degree=nx.degree(graph,node)
        x_graph.append(degree)

    x_graph.sort()

    degrees=[n for n in range(len(graph.nodes()))]

    dict={}

    for degree in degrees:
        dict[degree]=0
    

    for item in x_graph:
        if item in dict:
            dict[item]+=item

    for key,value in dict.items():
        y_graph.append(value)

    #print(x_graph)
    #print(y_graph)
    
    ############################################################################
    return x_graph, y_graph

#%%
def plot_degree_distributions(graph:nx.Graph,G_nm:nx.Graph,B_nm:nx.Graph):
    """
    Draw degree distribution plot of the real world graph and the two model instances
    """
    ############################################################################
    # TODO: Your code here!
    degrees=[n for n in range(len(graph.nodes()))]
    #print(degrees)

    x=get_degree_distribution(graph)
    #print(x[0])
    #print(x[1])

    x0=np.array(degrees)
    x1=np.array(x[1])

    plt.bar(x0,x1)
    plt.title("Real-World Graph Model")  # Add a title to the figure
    plt.xlabel("Degree d")       # Give a label to the x-axis
    plt.ylabel("Number of Nodes of Degree d (Nd)")       # Give a label to the y-axis
    plt.xscale('log')
    plt.yscale('log')
    plt.show()  

    y=get_degree_distribution(G_nm)
    #print(x[0])
    #print(x[1])

    y0=np.array(degrees)
    y1=np.array(y[1])

    plt.bar(y0,y1)
    plt.title("Gnm Graph Model")  # Add a title to the figure
    plt.xlabel("Degree d")       # Give a label to the x-axis
    plt.ylabel("Number of Nodes of Degree d (Nd)")       # Give a label to the y-axis
    plt.xscale('log')
    plt.yscale('log')
    plt.show()  

    z=get_degree_distribution(B_nm)
    #print(x[0])
    #print(x[1])

    z0=np.array(degrees)
    z1=np.array(z[1])

    plt.bar(z0,z1)
    plt.title("Bnm Model")  # Add a title to the figure
    plt.xlabel("Degree d")       # Give a label to the x-axis
    plt.ylabel("Number of Nodes of Degree d (Nd)")       # Give a label to the y-axis
    plt.xscale('log')
    plt.yscale('log')
    plt.show()   

    # NOTE: Use the `get_degree_distribution` function for each graph
    ############################################################################

#%% 
def print_clustering_coefficient(graph:nx.Graph,G_nm:nx.Graph,B_nm:nx.Graph):
    """
    Draw degree distribution plot of the real world graph and the two model instances
    """
    print(f"Clustering coefficient of {graph.name} = {compute_clustering_coefficient(graph):.4f}")
    print(f"Clustering coefficient of G({n},{m}) = {compute_clustering_coefficient(G_nm):.4f}")
    print(f"Clustering coefficient of B({n},{m}) = {compute_clustering_coefficient(B_nm):.4f}")

#%%
def print_num_nodes_edges(graph:nx.Graph):
    """
    Print number of nodes and edges of a given graphs.
    """
    print(f"{graph.name} Graph: Number of Nodes = {graph.number_of_nodes()}, Number of Edges = {graph.number_of_edges()}")

#%%
if __name__ == '__main__':
    # load real-world graph
    air_traffic_graph = load_real_world_network(name="air_traffic")
    n,m = air_traffic_graph.number_of_nodes(),air_traffic_graph.number_of_edges()

    # print number of edges
    print_num_nodes_edges(air_traffic_graph)

    # generate instance of G(n,m)
    air_traffic_G_nm = generate_G_nm(n,m)

    # generate instance of B(n,m)
    air_traffic_B_nm = generate_B_nm(n,m)

    # print average clustering coefficient
    print_clustering_coefficient(air_traffic_graph,air_traffic_G_nm,air_traffic_B_nm)

    # plot degree distribution
    plot_degree_distributions(air_traffic_graph,air_traffic_G_nm,air_traffic_B_nm)

    


    acad_collab_graph = load_real_world_network(name="academic_collaboration")
    n,m = acad_collab_graph.number_of_nodes(),acad_collab_graph.number_of_edges()
    print_num_nodes_edges(acad_collab_graph)
    
    # generate instance of G(n,m)
    acad_collab_G_nm = generate_G_nm(n,m)

    # generate instance of B(n,m)
    acad_collab_B_nm = generate_B_nm(n,m)

    # print average clustering coefficient
    print_clustering_coefficient(acad_collab_graph,acad_collab_G_nm,acad_collab_B_nm)
    
    # plot degree distribution
    plot_degree_distributions(acad_collab_graph,acad_collab_G_nm,acad_collab_B_nm)
    

# %%
