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



#%%
if __name__ == '__main__':
    start=0
    n=500
    cs=[]

    while start!=10.5:
        cs.append(start)
        start+=0.5
    #print(cs)

    ps=[]
    for i in cs:
        p=(1-(i/(n-1)))**(n-1)
        ps.append(p)
    #print(ps)

    npcs=np.array(cs)
    npps=np.array(ps)

    plt.plot(npcs,npps,color="green")
    plt.title("Values of C VS Probability")  # Add a title to the figure
    plt.xlabel("Values Of C")       # Give a label to the x-axis
    plt.ylabel("Values of Probability")       # Give a label to the y-axis
    plt.show()   

    collection=[]
    for item in ps:
        if nx.is_connected(nx.gnp_random_graph(n,item)):
            collection.append(item)
    #print(collection)

    transformation_of_collection=[]
    for item in collection:
        transformation_of_collection.append(ps.index(item))

    #print(transformation_of_collection)

    final=[]

    for i in range(0,len(transformation_of_collection)):
        final.append(cs[i])
    
    #print(final)

    smallest_c=min(final)

    print(f"The graph with smallest value of c for which the graph becomes connected is ",smallest_c,".")
    

# %%
