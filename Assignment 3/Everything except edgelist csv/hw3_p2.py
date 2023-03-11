# %%
################################################################################
# DATA16001: Network Analysis (2023)
# Homework 2
# Boilerplate code for Exercise 2
# Last Updated: Feb 6, 2023
################################################################################

# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
from typing import Dict, Any, Final, List
#! NOTE: INSTALL epydemic using `pip install epydemic`
from epydemic import CompartmentedModel, SynchronousDynamics, StochasticDynamics
import copy
import pandas as pd
import seaborn as sns

# Useful library that I used
import math
import random
import csv


class SIR_custom(CompartmentedModel):
    '''The Susceptible-Infected-Removed model.
    Susceptible nodes are infected by infected neighbours, and recover to
    removed.'''

    # Model parameters
    #: Parameter for probability of infection on contact.
    P_INFECT: Final[str] = 'pInfect'
    #: Parameter for probability of removal (recovery).
    P_REMOVE: Final[str] = 'pRecover'
    P_SEEDS: Final[str] = 'pSeeds'  # : Parameter for initial infected nodes
    # Possible dynamics states of a node for SIR dynamics
    #: Compartment for nodes susceptible to infection.
    SUSCEPTIBLE: Final[str] = 'S'
    INFECTED: Final[str] = 'I'            #: Compartment for nodes infected.
    #: Compartment for nodes recovered/removed.
    REMOVED: Final[str] = 'R'

    # Locus containing the edges at which dynamics can occur
    SI: Final[str] = 'SI'                 #: Edge able to transmit infection.

    def __init__(self):
        super().__init__()

    def build(self, params: Dict[str, Any]):
        '''Build the SIR model.

        :param params: the model parameters'''
        super().build(params)

        # collect parameters which are passed to us in a dictionary
        pInfect = params[self.P_INFECT]
        pRemove = params[self.P_REMOVE]

        # create compartments or states
        self.addCompartment(self.SUSCEPTIBLE, 0.0)
        self.addCompartment(self.INFECTED, 0.0)
        self.addCompartment(self.REMOVED, 0.0)

        # create and track the loci where events will occur
        self.trackEdgesBetweenCompartments(
            self.SUSCEPTIBLE, self.INFECTED, name=self.SI)
        self.trackNodesInCompartment(self.INFECTED)

        # define the events that will occur
        self.addEventPerElement(self.SI, pInfect, self.infect)
        self.addEventPerElement(self.INFECTED, pRemove, self.remove)

    def setUp(self, params: Dict[str, Any]):
        # initialise all nodes to an empty compartment
        # (so we can assume all nodes have a compartment attribute)
        g = self.network()
        for n in g.nodes():
            g.nodes[n][self.COMPARTMENT] = None
        # mark edges as unoccupied
        for (_, _, data) in g.edges(data=True):
            data[self.OCCUPIED] = False
        # Go through all nodes
        for node in g.nodes():
            # if node is in seed set mark as INFECTED
            if node in params[self.P_SEEDS]:
                self.changeCompartment(node, SIR_custom.INFECTED)
            # Otherwise mark as SUSCEPTIBLE
            else:
                self.changeCompartment(node, SIR_custom.SUSCEPTIBLE)

    def infect(self, t: float, e: Any):
        '''Perform an infection event. This changes the compartment of
        the susceptible-end node to :attr:`INFECTED`. It also marks the edge
        traversed as occupied.

        :param t: the simulation time
        :param e: the edge transmitting the infection, susceptible-infected'''
        (n, _) = e
        self.changeCompartment(n, self.INFECTED)
        self.markOccupied(e, t)

    def remove(self, t: float, n: Any):
        '''Perform a removal event. This changes the compartment of
        the node to :attr:`REMOVED`.

        :param t: the simulation time (unused)
        :param n: the node'''
        self.changeCompartment(n, self.REMOVED)


def getGraph(file):
    """Load the graph from a file

    Args:
        file (Path): Path to the edgelist file

    Returns:
        networxX.Graph: The graph
    """
    ###############################################################################
    # TODO: your code here 
    ###############################################################################
    G=nx.Graph()

    with open(file) as new_file:
        for line in new_file:
            if " " in line:
                parts=line.split(" ")
            parts[0]=int(parts[0].strip("\n"))
            parts[1]=int(parts[1].strip("\n"))

            G.add_edge(parts[0],parts[1])
    
    #print(G)
    return G


def SIRsimulation(G: nx.Graph, infected_seeds: List[int], p: float, r: float) -> Dict:
    """Runs one simulation of the SIR model on given graph
    and

    Args:
        G (nx.Graph): graph to run simulation 
        infected_seeds (List[int]): Nodes to be initially infected 
        p (float): The probability of infection spreading to neighbors
        r (float): The probability of recovery after infection

    Returns:
        Dict: results of nodes in different states of S,I and R
    """
    param = dict()
    param[SIR_custom.P_INFECT] = p  # infection probability
    #param[SIR_custom.P_REMOVE] = 1  # probability that node gets recovered

    param[SIR_custom.P_REMOVE] = r  # probability that node gets recovered
    # set the nodes to be initial infected seed nodes
    param[SIR_custom.P_SEEDS] = infected_seeds
    # create model
    m = SIR_custom()
    # create experiment
    e = StochasticDynamics(m, G)
    # set the experiment parameters and run the experiments
    rc = e.set(param).run()
    # gather the experiment results
    results = rc["results"]
    return results


def calculate_average_spread(candidate: int, seed_set: List[int], G: nx.Graph, p: float, r: float, tries: int = 5) -> int:
    """Function to compute the infection spread after adding 
    candidate node to existing seed set 

    Args:
        candidate (int): The node to add to seed set
        seed_set (List[int]): The existing set of initially infected nodes
        G (nx.Graph): The graph
        p (float): the probability of infection
        r (float): The probability of recovery
        tries (int): Number of tries to get average spread

    Returns:
        int: The spread from SIR experiment
    """
    ###############################################################################
    # TODO: your code here 
    # hint: use the SIRsimulation to run multiple experiments
    ###############################################################################
    spreaded=0
    #print("seed_set",seed_set)
    #print("candidate",candidate)
    #print("infected_seed",seed_set+[candidate])

    for i in range(tries):
        spreaded+=SIRsimulation(G=G,infected_seeds=seed_set+[candidate],p=p,r=r)['R']

    spreaded/=tries

    return spreaded


def greedy(p: float, G: nx.Graph, candidate_set: List[int], k: int, r: float, verbose: bool = False, tries: int = 5):
    """Greedy algorithm to select seed set of k 
    with maximum spread from candidate set.

    Args:
        p (float): The probability of infection
        G (nx.Graph): The graph
        candidate_set (List[int]): The set of candidates for seed nodes
        k (int): The number of seed nodes required
        r (float): The probability of recovery
        verbose (bool): Whether to print debug statements
        tries (int): Number of tries to get average spread


    Returns:
        (List[int],List[Int]): The list of seeds and the spread when adding that candidate
    """
    assert len(candidate_set) > k
    spreads = []
    seeds = []
    ###############################################################################
    # TODO: your code here 
    # Implement the Greedy algorithm. Use the calculate_spread function
    # keep seeds and record average spread at each iteration
    ###############################################################################
    #print(p)
    #print(G)
    #print(candidate_set)
    #print(k)
    #print(r)

    candidate_set=sorted(candidate_set,reverse=True)
    #print(candidate_set)

    count=0
    while count!=k:
        f=[]
        #print(candidate)
        for item in candidate_set:
                f.append(calculate_average_spread(candidate=item,seed_set=seeds,G=G,p=p,r=r))

        maximum=0
        index=0

        for a,b in enumerate(f):
            if maximum<=b:
                maximum=b
                index=a
        
        #print(candidate_set)
        #print(f)
        #print(index)
        #print(maximum)

        u_star=candidate_set[index]
        #print(candidate_set)
        #print(u_star)

        seeds.append(u_star)
        spreads.append(f[index])
        candidate_set.remove(u_star)
        count+=1
        f.clear()

    #print(seeds)
    #print(spreads)
    
    return seeds, spreads


#%%
if __name__=="__main__":
    G = getGraph("../h3_graph_data.edgelist")
    results = SIRsimulation(G, infected_seeds=[0, 1, 2], p=0.1, r=1)
    print(results)

    prob=[0.1,0.5]

    for p in prob:
        x=[1,2,3,4,5]

        for i in range(3):
            candidate_set=np.random.choice(G.nodes(),size=7,replace=False)
            #print(candidate_set)

            algo=greedy(p=p,G=G,candidate_set=candidate_set,k=5,r=5)
            y=algo[1]

            if i==0:
                plt.plot(x,y,color="red")
            elif i==1:
                plt.plot(x,y,color="blue")
            else:
                plt.plot(x,y,color="green")

            if p==0.1:
                plt.title("P=0.1")  # Add a title to the figure
            else:
                plt.title("P=0.5")  # Add a title to the figure

            with open("results.csv","a") as new_file:
                line1=""
                line2=""
                csv_writer = csv.writer(new_file)
                for item in algo[0]:
                    line1+=str(item)
                    line1+=";"
                new_file.write("seeds;"+line1+"\n")

                for item in algo[1]:
                    line2+=str(item)
                    line2+=";"
                new_file.write("spreads;"+line2+"\n")

        plt.xlabel("Size Of The Seed Set S")       # Give a label to the x-axis
        plt.ylabel("f(S)")       # Give a label to the y-axis
        plt.show()

        y.clear()

#%%
