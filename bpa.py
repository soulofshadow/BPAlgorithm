# library import
import numpy as np

# config  ome given parameters
INPUT_DIM = 8
HIDDEN_DIM = 3
OUTPUT_DIM = 8
EPOCHS = 20000
LR = 0.5  # learning rate
WD = 0.001  # weight decay

class NN:

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, lr=LR, wd=WD, epochs=EPOCHS):
        # init parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.wd = wd
        self.epochs = epochs

        # generate some data for train
        self.dataset = self.generate_samples()
        self.y = self.generate_samples()

        # set for count cost and epochs
        self.cost = []
        self.epochs_run = 0
        self.activation_hidden = np.zeros((3,))

        # init weight using uniform distribution
        self.synapse_0 = np.random.uniform(-.05, .05, (self.hidden_dim, self.input_dim + 1))
        self.synapse_1 = np.random.uniform(-.05, .05, (self.output_dim, self.hidden_dim + 1))
        # temp store the gradients per calculate
        self.gradients_0 = np.zeros(self.synapse_0.shape)
        self.gradients_1 = np.zeros(self.synapse_1.shape)

    # train
    def train(self):
        train = True
        epoch = 0
        while train and epoch < EPOCHS:
            epoch = epoch+1
            print("epoch:", epoch)
            self.fit()
            if epoch % 10 == 0 and self.test():
                train = False
        self.epochs_run = epoch
        self.test(True)

    def test(self, print_=False):
        n_correct = 0
        CONVERGED = False
        if print_:
            print('')

        for i, sample in enumerate(self.dataset):
            predicted_y, a_hidden_layer = self.forward(sample)
            if np.argmax(sample) == np.argmax(predicted_y):
                n_correct += 1
            if print_:
                v = np.zeros(sample.shape)
                v[np.argmax(predicted_y)] = 1
                print(str(i) + ":")
                print("label:")
                print(sample)
                print("predicted:")
                print(v)

        if print_:
            print("Number of correct classes:", n_correct)

        if n_correct == 8:
            CONVERGED = True

        return CONVERGED

    def fit(self):
        # one gradient per weight
        self.gradients_0 = np.zeros(self.synapse_0.shape)
        self.gradients_1 = np.zeros(self.synapse_1.shape)
        cost = 0

        for sample in self.dataset:
            # get rid of the bias parameter
            # sample (i.e.) = [0,1,0,0,0,0,0,0]
            desired_y = sample

            # forward pass
            predicted_y, a_hidden_layer = self.forward(sample)

            # calculate the cost
            cost = cost + 0.5 * (np.linalg.norm(predicted_y - desired_y) ** 2)

            # compute the delta errors and get the gradient of this tims's backward
            self.backward(predicted_y, desired_y, a_hidden_layer,sample)

        print(cost / self.input_dim)
        self.cost.append(cost / self.input_dim)
        self.update_weights()

    def forward(self, x):
        #calculated as the weighted sum of the inputs
        #let the bias be the first row connecting input node

        #forward layer 1
        weights1 = self.synapse_0
        sum1 = np.dot(weights1[:, 1:], x) + weights1[:, 0]
        a_hidden_layer = self.sigmoid(sum1)
        
        #forward layer2
        weights2 = self.synapse_1
        sum2 = np.dot(weights2[:, 1:], a_hidden_layer) + weights2[:, 0]
        predicted_y = self.sigmoid(sum2)

        #update the hidden node
        self.activation_hidden = a_hidden_layer

        return predicted_y, a_hidden_layer

    def backward(self, p_y, d_y, h_l, x):
        # p_y == predicted_y
        # d_y == desired_y
        # h_l == a_hidden_layer

        # compute delta_output_layer
        delta_output_layer = -1 * (d_y - p_y) * self.sigmoid_derivative(p_y)
        # compute delta_hidden_layer
        delta_hidden_layer = np.dot(self.synapse_1.T[1:], delta_output_layer) * self.sigmoid_derivative(h_l)

        #store the partial derivatives
        # synapse 1
        for i in range(len(delta_output_layer)):
            self.gradients_1[i, 0] = self.gradients_1[i, 0] + delta_output_layer[i]
            for j in range(len(h_l)):
                self.gradients_1[i, j + 1] = self.gradients_1[i, j + 1] + h_l[j] * delta_output_layer[i]

        # synapse 0
        for i in range(len(delta_hidden_layer)):
            self.gradients_0[i, 0] = self.gradients_0[i, 0] + delta_hidden_layer[i]
            for j in range(len(x)):
                self.gradients_0[i, j + 1] = self.gradients_0[i, j + 1] + x[j] * delta_hidden_layer[i]

    def update_weights(self):
        # compute cost gradient
        cost_gradient_synapse1 = np.zeros(self.synapse_1.shape)
        cost_gradient_synapse0 = np.zeros(self.synapse_0.shape)

        cost_gradient_synapse1[:, 1:] = (self.gradients_1[:, 1:] + (self.wd * self.synapse_1[:, 1:])) / self.input_dim
        cost_gradient_synapse0[:, 1:] = (self.gradients_0[:, 1:] + (self.wd * self.synapse_0[:, 1:])) / self.input_dim

        cost_gradient_synapse1[:, 0] = self.gradients_1[:, 0] / self.input_dim
        cost_gradient_synapse0[:, 0] = self.gradients_0[:, 0] / self.input_dim

        # update weights with gradient descent
        self.synapse_0 = self.synapse_0 - self.lr * cost_gradient_synapse0
        self.synapse_1 = self.synapse_1 - self.lr * cost_gradient_synapse1
    
    # for data, only 8 learning samples.
    @staticmethod
    def generate_samples():
        a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        b = np.zeros((a.size, a.size))
        b[a, a] = 1
        return b
    
    # activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)