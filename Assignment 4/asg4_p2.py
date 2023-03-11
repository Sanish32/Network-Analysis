################################################################################
# DATA16001: Network Analysis (2023)
# Homework 4
# Boilerplate code for Exercise 2
# Last Updated: Feb 10, 2023
################################################################################
#%%
import numpy as np
import networkx as nx

#Important libralies that I have used
import matplotlib
import pickle
import os
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

#%%
def load_dataset(dataset_number:int):
    """Load the matrices from the files

    Args:
        dataset_number (int): The dataset number

    Returns:
        A,X,Y: The adjacency, original features and labels for the dataset
    """
    ###############################################################################
    # TODO: your code here 
    #Finding the common path
    file_dir = os.path.dirname(os.path.realpath('__file__'))
    #print(file_dir)

    #Finding path to A,X and Y
    file_nameA = os.path.join(file_dir,'hw4_p2_data','graph_'+str(dataset_number)+'_A.pkl')
    #print(file_nameA)
    file_nameX = os.path.join(file_dir,'hw4_p2_data','graph_'+str(dataset_number)+'_X.pkl')
    #print(file_nameX)
    file_nameY = os.path.join(file_dir,'hw4_p2_data','graph_'+str(dataset_number)+'_Y.pkl')
    #print(file_nameY)

    #Opening and Accessing
    with open(file_nameA, 'rb') as new_file:
        A=pickle.load(new_file)
        #print(type(A))

    with open(file_nameX, 'rb') as new_file:
        X=pickle.load(new_file)
        #print(type(X))

    with open(file_nameY, 'rb') as new_file:
        Y=pickle.load(new_file)
        #print(type(Y))

    #Checking what I got
    #print(A)
    #print(issymmetric(A))
    #print(A[0])
    #print(X)
    #print(Y)

    #Checking the dimension of matrices
    #print(len(A[0]))
    #print(len(X))
    #print(len(X[0]))
    #print(len(Y))

    #Removing self-loop for adjacency matrix
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i==j:
                A[i][j]=0

    #print(type(A))
    #print(A)
    #A=np.array(A)
    #print(A)

    ###############################################################################
    return A,X,Y
#%%
def compute_laplacian_embeddings(A:np.ndarray):
    """Computes the Laplacian eigenvector embeddings of a given adjacency matrix.

    Args:
        A (np.ndarray): The adjacency matrix of a graph

    Returns:
        np.ndarray: The 2nd and 3rd smallest magnitude eigenvectors
    """
    ###############################################################################
    # TODO: your code here 
    # Hint: Use scipy.sparse.linalg.eigsh 
    ###############################################################################
    
    D=[[sum(A[i][:]) if i==j else 0 for i in range(len(A))] for j in range(len(A[0]))]
    #print(D)
    #print(type(D))

    #Converting to use scipy.sparse.linalg.eigsh even though D is not used here
    D=np.array(D)
    #print(type(D))

    #print(D)
    #print(len(D))
    #print(len(D[199]))

    #print(D)

    L=[[(D[i][j]-A[i][j]) for i in range(len(A))] for j in range(len(A[0]))]

    #print(len(L))
    #print(len(L[0]))

    #To check if L is proper or not
    #c1=0
    #c2=0
    #for i in range(len(L)):
    #    for j in range(len(L[0])):
    #        if i==j:
    #            if L[i][j]>0:
    #                c1+=1
    #        else:
    #            if L[i][j]==-1 or L[i][j]==0:
    #                c2+=1
    
    #print(c1)
    #print(c2)

    #print(L)
    #print(type(L))

    #Converting to use scipy.sparse.linalg.eigsh for L
    L=np.array(L)
    #print(issymmetric(L))
    L=L.astype(np.float32)

    #print(type(L))

    #print(issymmetric(A))
    #A=A.astype(np.float32)
    #print(A)

    #eig_values,eig_vectors=eigsh(A,k=3,which="SM")

    #print(len(eig_values))
    #print(eig_values)
    #print(len(eig_vectors))
    #print(eig_vectors[199,:])

    eig_values,eig_vectors=eigsh(L,k=3,which="SM")

    #print(len(eig_values))
    #print(eig_values)
    #print(len(eig_vectors))
    #print(eig_vectors[0,:])

    #print(eig_values)

    eig_vectors=eig_vectors[:,[1,2]]
    #print(eig_vectors[0,:])
    

    #eig_values=eig_values.tolist()
    #print(type(eig_values))
    #print(0 in eig_values)
    #print(eig_values)
    #print(a)

    #Looks like not possible to find 0 because one of the eigenvalues is close to 0 but not exactly 0
    #print(0 in eig_values)
    #for i in range(len(eig_values)):
    #    if eig_values[i]==0:
    #        print(a[i])
    
    #temp=[a[:,i] for i in range(len(eig_values))]
    #print(temp)
    #print(temp[0])
    #print(temp[1])
    #print(temp[2])
    #print(len(temp))
    
    #eig_vectors=[]
    #eig_vectors.append(temp[1])
    #eig_vectors.append(temp[2])
    #print(len(eig_vectors))
    #print(len(eig_vectors[0]))
    
    return eig_vectors
# %%
def plot_scatter(A:np.ndarray,X:np.ndarray,Y:np.ndarray,dataset_number:int) -> None:
    """Make the scatter plots

    Args:
        A (np.ndarray): The adjacency matrix
        X (np.ndarray): The original feature matrix
        Y (np.ndarray): The label vector
        dataset_number (int): the dataset number
    """
    ###############################################################################
    # TODO: your code here 
    # Hint: Use the compute_laplacian_embeddings to get the embeddings
    ###############################################################################
    

    us=compute_laplacian_embeddings(A)

    u2,u3=us[:,0],us[:,1]

    #print(us[199:])
    #print(u2)
    #print(u3)

    u2=np.array(u2)
    u3=np.array(u3)

    #print(X)
    #print(X[:,0])

    X1,X2=X[:,0],X[:,1]
    
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    fig.suptitle('2 Subplots of Network Dataset '+str(dataset_number))

    colors = ['red','green','blue','purple']

    ax1.set_title('Attributes of Nodes X')
    ax1.set_xlabel('First Column of X--->')
    ax1.set_ylabel('Second Column of X--->')
    
    ax1.scatter(X1,X2,c=Y, cmap=matplotlib.colors.ListedColormap(colors))


    ax2.set_title('Eigenvectors of Laplacian matrix')
    ax2.set_xlabel('First Eigenvector--->')
    ax2.set_ylabel('Second Eigenvector--->')
    ax2.scatter(u2,u3,c=Y,cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()

# %%
def plot_layouts(A:np.ndarray,dataset_number:int) -> None:
    """Make the NetworkX layout plots

    Args:
        A (np.ndarray): The adjacency matrix
        dataset_number (int): The dataset number
    """
    ###############################################################################
    # TODO: your code here 
    # Hint: Use the networkx layout functions
    ###############################################################################
    G=nx.Graph()

    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j]!=0:
                G.add_edge(i,j)

    #Spectral Layout
    pos1=nx.spectral_layout(G)
    nx.draw(G,pos=pos1)
    plt.title("Spectral Layout of Network Dataset "+str(dataset_number))
    plt.show()

    #Spring Layout
    pos2=nx.spring_layout(G)
    nx.draw(G,pos=pos2)
    plt.title("Spring Layout of Network Dataset "+str(dataset_number))
    plt.show()

# %%
def fit_predict(features:np.ndarray,labels:np.ndarray):
    """Method that fits a RandomForestClassifier on a subset of data
    and makes predictions

    Args:
        features (np.ndarray): The features to train the model on
        labels (np.ndarray): The labels to train the model with

    Returns:
        float: the accuracy of the model on test data
    """
    ###############################################################################
    # TODO: your code here 
    # Hint: Use the scikit learn library 
    # Note: Split the features and labels into 40% for training and 60% for testing
    # and keep the splits consistent. For example, using a seed 
    ###############################################################################
    
    X_train,X_test,y_train,y_test=train_test_split(features,labels,train_size=0.40,test_size=0.60)
    #print(len(value))
    #print((value[0]))
    #print((value[1]))

    #print(X_train)
    #print(X_test)
    #print(y_train)
    #print(y_test)

    #print(X_train.size)

    if X_train.size==80:
        #X_train=np.array(X_train)
        #X_test=np.array(X_test)
        #y_train=np.array(y_train)
        #y_test=np.array(y_test)

        # fit model using training features and labels
        clf = RandomForestClassifier()
        clf.fit(X_train.reshape(-1,1),y_train)

        # use trained model to predict on test features
        preds = clf.predict(X_test.reshape(-1,1))
        #print(preds)
        #print(y_test)

    if X_train.size==160:
        #X_train=np.array(X_train)
        #X_test=np.array(X_test)
        #y_train=np.array(y_train)
        #y_test=np.array(y_test)
        #print(X_train)

        # fit model using training features and labels
        clf = RandomForestClassifier()
        clf.fit(X_train.reshape(-1,2),y_train)

        # use trained model to predict on test features
        preds = clf.predict(X_test.reshape(-1,2))
        #print(preds)
        #print(y_test)
    
    if X_train.size==320:
        #X_train=np.array(X_train)
        #X_test=np.array(X_test)
        #y_train=np.array(y_train)
        #y_test=np.array(y_test)

        # fit model using training features and labels
        clf = RandomForestClassifier()
        clf.fit(X_train.reshape(-1,4),y_train)

        # use trained model to predict on test features
        preds = clf.predict(X_test.reshape(-1,4))
        #print(preds)
        #print(y_test)

    # Compute accuracy using test labels
    acc = accuracy_score(y_test,preds)
    return acc

#%%
def plot_accs(A:np.ndarray,X:np.ndarray,Y:np.ndarray,dataset_number:int):
    """Make the accuracy bar plots as per assignment

    Args:
        A (np.ndarray): the adjacency matrix
        X (np.ndarray): the original feature matrix
        Y (np.ndarray): the label vector
        dataset_number (int): the dataset number
    """ 
    ###############################################################################
    # TODO: your code here 
    # Note: Gather results for all feature matrices mentioned in assignment 
    ###############################################################################
    
    us=compute_laplacian_embeddings(A)

    u2,u3=us[:,0],us[:,1]
    #print(us)

    #print(us[199:])
    #print(u2)
    #print(u3)

    u2=np.array(u2)
    u3=np.array(u3)

    #print(u2.size)

    #print(u2)
    #print(u3)

    #u2
    ac1=fit_predict(u2,Y)
    #print(ac1)

    #u3
    ac2=fit_predict(u3,Y)
    #print(ac2)

    #[u2,u3]
    #print(us.size)
    ac3=fit_predict(us,Y)
    #print(ac3)

    #X
    #print(X.size)
    ac4=fit_predict(X,Y)
    #print(ac4)

    X2,X3=X[:,0],X[:,1]

    newX=np.column_stack((X2,X3))
    #print(newX.size)

    #X u2 u3
    new1=np.column_stack((newX,u2))
    new2=np.column_stack((new1,u3))

    #print(new1.size)
    #print(new2.size)

    #print(new)
    ac5=fit_predict(new2,Y)
    #print(ac5)

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])  

    xpart=['u2','u3','[u2 u3]','X','[X u2 u3]']
    acc=[ac1,ac2,ac3,ac4,ac5]

    ax.bar(xpart,acc,width = 0.4)
    ax.set_title('Barplot of Network Dataset '+str(dataset_number))
    ax.set_xlabel('Five Random Forest Models--->')
    ax.set_ylabel('Accuracies--->')
    plt.show()

#%%
if __name__ == "__main__":
    for dataset_number in range(1,4):
        # load the matrices
        A,X,Y = load_dataset(dataset_number)
        # make the scatter plots
        plot_scatter(A,X,Y,dataset_number)
        # make the layout plots
        plot_layouts(A,dataset_number)
        # make the accuracy plots
        plot_accs(A,X,Y,dataset_number)
# %%
