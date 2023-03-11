#%%
################################################################################
# DATA16001: Network Analysis (2023)
# Bonus Assignment 
################################################################################

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
import math

#%%
if __name__ == '__main__':
    #Load the xlsx file
    excel_data = pd.read_excel('pvldb.xlsx')


    #Read the values of the file in the dataframe
    data = pd.DataFrame(excel_data, columns=[
                    'COAUTHOR', 'REVIEWER', 'REVIEWER DBLP','CO-AUTHORSHIP 3 YEARS'])
    
    #print(len(data))

    #Print the content
    #print("The content of the file is:\n", data)

    REVIEWER=data['REVIEWER']
    COAUTHOR=data['COAUTHOR']
    CO_AUTHORSHIP_3_YEARS=data['CO-AUTHORSHIP 3 YEARS'].values.tolist()
    #print(len(CO_AUTHORSHIP_3_YEARS))

    weighting=[]

    #Build the main graph
    G=nx.Graph()

    for i in range(len(REVIEWER)):
        G.add_node(REVIEWER[i])
        if pd.isna(CO_AUTHORSHIP_3_YEARS[i])==False:
            G.add_node(COAUTHOR[i])

            #print(CO_AUTHORSHIP_3_YEARS[i])
            #print(type(CO_AUTHORSHIP_3_YEARS[i]))

            #print(CO_AUTHORSHIP_3_YEARS[i])
            CO_AUTHORSHIP_3_YEARS[i]=list(CO_AUTHORSHIP_3_YEARS[i].split(","))
            #print(CO_AUTHORSHIP_3_YEARS[0])
            #print(CO_AUTHORSHIP_3_YEARS[0][0])
            #print(type(CO_AUTHORSHIP_3_YEARS[0]))
            
            weight=0
            for item in CO_AUTHORSHIP_3_YEARS[i]:
                #print(item)
                if ')' in item:
                    item=item.replace(")","")
                    if ']' in item:
                        item=item.replace("]","")
                    #print(item)
                    weight+=int(item)
            G.add_edge(REVIEWER[i],COAUTHOR[i],weight=weight)
            weighting.append(weight)
    #print(len(G.nodes()))

    main_nodes=G.nodes()
    main_edges=G.edges()
    print("1.1)")
    #i)The number of nodes and edges
    print("i)")
    print("Number of nodes of main graph is "+str(len(main_nodes)))
    print("Number of edges of main graph is "+str(len(main_edges)))

    #ii)Number of Connected Components
    print("ii)")
    no_connected_components=nx.number_connected_components(G)
    print("Number of Connected Components is "+str(no_connected_components))
    
    #iii)Frequency Histogram
    print("iii)")
    #weighting.sort()
    #print(weighting)

    plt.title("The Distribution Of Edge Weights W")
    plt.xlabel("Weight of Edges--->")
    plt.ylabel("Number of edges--->")
    plt.hist(weighting)
    plt.show()

    #Finding the quantile of 70% and 90%
    print("Values of Quantiles")
    w1=np.quantile(weighting, 0.7)
    print("Value of w1 is "+str(w1))

    w2=np.quantile(weighting, 0.9)
    print("Value of w2 is "+str(w2))

    #Build two subgraphs 
    print("Build two subgraphs")
    G1=nx.Graph()
    G2=nx.Graph()

    for item in main_edges:
        if G[item[0]][item[1]]['weight']>=w1:
            G1.add_edge(item[0],item[1])
        
        if G[item[0]][item[1]]['weight']>=w2:
            G2.add_edge(item[0],item[1])

    edges1=G1.edges()
    edges2=G2.edges()
    #1.2
    print("1.2)")
    #i)The number of edges
    print("i)")
    print("Edges of G1 is "+str(len(edges1)))
    print("Edges of G2 is "+str(len(edges2)))

    #ii)The number of connected components that contain at least one edge.
    print("ii)")

    #Connected components might not make sense but its all good because more edges means there is small number of connected components

    #This is bad because it counts all the nodes that have at least one edge in G1
    #nodes1=G1.nodes()
    #count1=0
    #for n in nodes1:
    #    if len([n for _ in nx.neighbors(G1,n)])>0:
    #        count1+=1
    #print("The number of connected components that contain at least one edge in G1 is "+str(count1))
    
    print("The number of connected components that contain at least one edge in G1 is "+str(nx.number_connected_components(G1)))

    #This is bad because it counts all the nodes that have at least one edge in G2
    #nodes2=G2.nodes()
    #count2=0
    #for n in nodes2:
    #    if len([n for _ in nx.neighbors(G2,n)])>0:
    #        count2+=1
    #print("The number of connected components that contain at least one edge in G2 is "+str(count2))
    
    print("The number of connected components that contain at least one edge in G2 is "+str(nx.number_connected_components(G2)))

    #1.4)
    #print(G1.nodes())

    #print("deepti raghavan" in REVIEWER.values)
    #print("deepti raghavan" in COAUTHOR.values)

    nodes1=G.nodes()

    reviewerx=[]
    non_reviewer=[]

    for n in nodes1:
        if n in REVIEWER.values:
            reviewerx.append(n)
        else:
            non_reviewer.append(n)

    #print(reviewerx)
    #print(non_reviewer)

    e_also_in_edges1=[]
    e_also_in_edges2=[]

    for e in edges1:
        if e in edges2:
            e_also_in_edges2.append(e)
        else:
            e_also_in_edges1.append(e)

    layout=nx.circular_layout(G1)

    # draw nodes
    nx.draw_networkx_nodes(G1,pos=layout,node_color="gray")
    #nx.draw_networkx_nodes(G1,pos=layout,nodelist=non_reviewer,node_color="blue")

    nx.draw_networkx_labels(G1, pos=layout)

    nx.draw_networkx_edges(G1, pos=layout, edgelist=e_also_in_edges1, edge_color='red', width=5)
    nx.draw_networkx_edges(G1, pos=layout, edgelist=e_also_in_edges2, edge_color='black',width=2)

    plt.title("Visualization of G1")
    plt.show()
    
    

#%%