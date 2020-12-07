import numpy as np

# This will contain all of our misc functions to be used for our neural networks
class Misc:

    # This will perform the sigmoid operation, or take the sigmoid derivative from the prediction using its differential equation form
    @staticmethod
    def sigmoid(x, deriv=False):
        if deriv:
            return x * (1 - x)

        return 1/(1 + np.exp(-x))

    # This will perform the relu operation or the derivative of the relu function at some point x
    @staticmethod
    def relu(x, deriv=False):
        if deriv:
            return 1 * (x > 0)

        return x * (x > 0)

    # This will get us the mean squared error or the derivative of the mean squared error for our network to optimize
    @staticmethod
    def MSE(predicted, actual, deriv=False):
        if deriv:
            return (1 / len(predicted)) * (predicted - actual)

        return 0.5 * (1 / len(predicted)) * np.sum((predicted - actual)**2)

# This will be the dense layer of out network
class Dense:
    def __init__(self, input_size, output_size, activation):
        self.__input_size = input_size
        self.__output_size = output_size

        # Randomly initialize the weights and biases to fit the input and output size we specify
        self.__weights = np.random.random((output_size, input_size))
        self.__bias = np.random.random((output_size, 1))

        # Initialize the activation function to be used for this layer
        self.__activation = activation

    # This will predict an outcome based on the inputs we specify
    def predict(self, inputs):
        raw_prediction = np.matmul(self.__weights, inputs) + self.__bias # This will create our raw linear prediction
        activated = self.__activation(raw_prediction) # This will apply the activation function to our raw linear prediction

        return activated
    
    # This will train our network given a set of inputs, a label for those inputs and a derivative of the errors with respect to the activation layer
    def train(self, inputs, prediction, loss_deriv, lr):
        error_deriv = loss_deriv * self.__activation(prediction, deriv=True) # This will calculate the derivative of the error function with respect to the activation layer

        weight_updates = np.matmul(error_deriv, inputs.T) # This will calculate the derivative of the error with respect to the weights
        bias_updates = error_deriv # This calculates the derivative of the error with respect to the bias
        hidden_updates = np.matmul(self.__weights.T, error_deriv) # This calculates the error with respect to the hidden layers

        self.__weights -= lr * weight_updates # Adjust our weights by a portion (specified by the learn rate) of our gradients
        self.__bias -= lr * bias_updates # Adjust our bias by a portion (specified by the learning rate) of our gradients

        return hidden_updates

class Model:
    def __init__(self, loss_function, *layers):
        self.__loss_function = loss_function # Save our loss function to use
        self.__layers = layers # Save the layers to be used for the model


    # This will form a prediction using all the layers in our specified network
    def predict(self, inputs):
        x = inputs
        for layer in self.__layers:
            x = layer.predict(x)
        
        return x

    # This will train all of the layers in our network to minimize the loss function we specify given the inputs and a trained label
    def train(self, inputs, label, lr=0.01):
        # Here we will create the predictions and store the predictions for each layer of our network
        predictions = [inputs]
        x = inputs
        for layer in self.__layers:
            x = layer.predict(x)
            predictions.append(x)

        # We will calculate the derivative of the loss function given our predicted output and its correct label
        error_deriv = self.__loss_function(x, label, deriv=True)

        # Here we will reverse our predictions for ease of iteration backwards over the layers in the network
        predictions_train = predictions[::-1]

        # Here we will iterate over each layer in our reversed network, updating each layer and feeding the error of the last layer into the previous layer to train it iteratively
        for i, layer in enumerate(self.__layers[::-1]):
            error_deriv = layer.train(predictions_train[i + 1], predictions_train[i], error_deriv, lr)


if __name__ == '__main__':
    # Here we will create some test data and labels to fit our test network to
    inputs = np.array([[[1],
                        [0]],
                       [[0],
                        [1]],
                       [[0],
                        [0]],
                       [[1],
                        [1]]])
    labels = np.array([0, 0, 0, 1])

    # Create a network with a loss function of MSE with 2 hidden layers with sizes of 4 and 6 with activation relu, followed by an output layer of size 1 with activation sigmoid
    model = Model(
        Misc.MSE,
        Dense(2, 4, activation=Misc.relu),
        Dense(4, 6, activation=Misc.relu),
        Dense(6, 1, activation=Misc.sigmoid)
    )

    loss = [] # Create an array to store our total loss on our predictions on our data for each epoch
    for _ in range(5000): # This will represent our epochs
        loss_sum = 0 # Create a counter for the total loss for each epoch
        for input, label in zip(inputs, labels): # Iterate over the data in our dataset 
            prediction = model.predict(input) # Perform a prediction on our input
            loss_sum += Misc.MSE(prediction, label) # Add the loss for this prediction to our total loss

            model.train(input, label) # Train our network given this input and label
        loss.append(loss_sum) # Store out total loss for this epoch to be viewed later
        
    
    # Now we will classify all the data in our dataset and compare the classification with its actual classification
    for input, label in zip(inputs, labels):
        prediction = model.predict(input).item()
        print(f"Prediction: {prediction} | Actual: {label}")

    # Here we will print the loss for every specified amount of epochs to observe the models training history
    print(f"\nTotal loss for every 100th epoch: {[loss for i, loss in enumerate(loss) if i % 100 == 0]}") 


