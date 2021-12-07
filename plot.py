from bpa import *
import matplotlib.pyplot as plt
import numpy as np

def plot():
    nn = NN()
    nn.train()

    plt.plot(np.array(range(nn.epochs_run)), nn.cost)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")

    titl = "Learning Rate=" + str(LR) + ", Weight Decay = " + str(WD)

    print('')
    plt.title(titl)

    print('')
    print("Weights layer 1:", nn.synapse_0)
    print('')
    print("Weights layer 2:", nn.synapse_1)
    print('')
    print("Activation values hidden layer", nn.activation_hidden)
    
    plt.show()