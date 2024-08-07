
import numpy as np
import matplotlib.pyplot as plt

def validation_curve(train_data, test_data):
       '''
    Plot validation curves for training and testing data.

    Parameters:
    - train_data (list or array): Accuracy scores from the training dataset over epochs.
    - test_data (list or array): Accuracy scores from the testing dataset over epochs.
       '''
    
        plt.plot(train_data, label='training score')
        plt.plot(test_data, label='test score')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy');
        plt.show()

class Layer:
    def __init__(self, input_size, output_size, activation_function = 'sigmoid'):
        '''
        Initialize a neural network layer.

        Parameters:
        - input_size (int): Number of input neurons.
        - output_size (int): Number of output neurons.
        - activation_function (str): Activation function to use ('sigmoid' or 'relu').
        '''
        
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size)
        self.biasses = np.random.randn(output_size)
        self.activation_function = activation_function
        self.input_data = None
        self.output_data = None
        self.z = None

    def forward(self, input_data):
        self.input_data = input_data
        z = self.weights @ input_data + self.biasses
        self.z = z
        if self.activation_function == 'sigmoid':
            self.a = self.sigmoid(self.z)
            return self.a
        elif self.activation_function == 'relu':
            return np.maximum(0,self.z)
        else: 
            raise ValueError('Unsupported activation function: Select sigmoid or relu.')

    def backward(self, dC_da):
        if self.activation_function == 'sigmoid':
            dC_dz = dC_da * self.sigmoid_derivative(self.z)
        elif self.activation_function == 'relu':
            dC_dz = dC_da * (self.z > 0)
        else:
            raise ValueError('Unsupported acctivation function: Please select sigmoid or relu.')
        dC_dw = np.outer(dC_dz, self.input_data)
        dC_db = dC_dz
        dC_da_prev = np.dot(self.weights.T, dC_dz)
        return dC_da_prev, dC_db, dC_dw

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def __repr__(self):
        return f"Layer{self.weights.T.shape}"
    
class NeuralNetwork:
    
    def __init__(self, layer_sizes, activation_function = None):
        
        '''
        Initializes the neural network with specified layers and activation functions.

        Parameters:
        - layer_sizes (list of int): Sizes of each layer in the network.
        - activation_function (str or list of str, optional): Activation functions for each layer.
        '''
        
        self.sizes = layer_sizes 
        if activation_function is None:
            activations = ['sigmoid'] * (len(layer_sizes)-1)
        elif isinstance(activation_function, str):
            activation_function = [activation_function] * (len(layer_sizes) - 1)
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)] #
        self.num_layers = len(self.layers)  # Add num_layers attribute

    def feedforward(self, inputs):
        # Iterate through layers, passing inputs forward through each layer
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def __repr__(self):
        return f"NeuralNetwork(sizes={self.sizes})"

    def backpropagate(self, x, y):
        # step 1: forwar pass
        output = self.feedforward(x)

        # need dC/da for FINAL layer
        dC_da = output - y

        nabla_b, nabla_w = [], []
        for layer in reversed(self.layers):
            dC_da, dC_db, dC_dw = layer.backward(dC_da)
            nabla_b.append(dC_db)
            nabla_w.append(dC_dw)
        return nabla_b[::-1], nabla_w[::-1]

    def evaluate(self, inputs, targets):
        correct_predictions = 0
        for input_data, target in zip(inputs, targets):
            output = self.feedforward(input_data)
            prediction = np.argmax(output)  # Get the index of the neuron with the highest output
            if prediction == np.argmax(target):
                correct_predictions += 1
        return correct_predictions
    
    def SGD(self, Xtrain, ytrain, Xtest, ytest, epochs=10, eta=0.1):
        n_train = len(Xtrain)
        n_test = len(Xtest)
        self.train_scores = []
        self.test_scores = []

        for epoch in range(epochs):
            # Shuffle training data
            #indices = np.random.permutation(n_train)
            #Xtrain_shuffled = Xtrain[indices]
            #ytrain_shuffled = ytrain[indices] 

            # Training loop
            for x, y in zip(Xtrain, ytrain):
                nabla_b, nabla_w = self.backpropagate(x, y)

                # Update weights and biases
                for l in range(len(self.layers)):
                    self.layers[l].weights -= eta * nabla_w[l]
                    self.layers[l].biasses -= eta * nabla_b[l]

            # Evaluate performance on test data
            correct_predictions = self.evaluate(Xtest, ytest)
            self.test_scores.append(correct_predictions / n_test)
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Test accuracy = {(correct_predictions / n_test)*100:.2f}%")
            correct_predictions = self.evaluate(Xtrain, ytrain)
            self.train_scores.append(correct_predictions / n_train)

    
    def layer_testing(self, configs, Xtrain, ytrain, Xtest, ytest, epochs=20, eta=0.005, batch_size=32, output_file='performance.csv'):
        performance_data = []
        for config in configs:
            self.__init__(config)
            self.SGD(Xtrain, ytrain, Xtest, ytest, epochs=epochs, eta=eta)
            last_epoch_performance = self.test_scores[-1]
            performance_data.append({
                'layer_config': config,
                'test_accuracy': last_epoch_performance
            })
            print(f"Configuration: {config}, Test Accuracy: {last_epoch_performance * 100:.2f}%")
        
        # Convert performance data to a DataFrame and save to CSV
        df_performance = pd.DataFrame(performance_data)
        df_performance.to_csv(output_file, index=False)
        print(f"Performance data saved to {output_file}")
        return self.config_performance
