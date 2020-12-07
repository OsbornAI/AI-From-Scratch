import numpy as np

class Misc:
    @staticmethod
    def sigmoid(x, deriv=False):
        if deriv:
            return x * (1 - x)

        return 1/(1 + np.exp(-x))

    @staticmethod
    def relu(x, deriv=False):
        if deriv:
            return 1 * (x > 0)

        return x * (x > 0)

    @staticmethod
    def MSE(predicted, actual, deriv=False):
        if deriv:
            return (1 / len(predicted)) * (predicted - actual)

        return 0.5 * (1 / len(predicted)) * np.sum((predicted - actual)**2)

class Dense:
    def __init__(self, input_size, output_size, activation):
        self.__input_size = input_size
        self.__output_size = output_size

        self.__weights = np.random.random((output_size, input_size))
        self.__bias = np.random.random((output_size, 1))
        self.__activation = activation

    def predict(self, inputs):
        applied = np.matmul(self.__weights, inputs) + self.__bias 
        activated = self.__activation(applied)

        return activated
    
    def train(self, inputs, prediction, loss_deriv, lr):
        error_deriv = loss_deriv * self.__activation(prediction, deriv=True)

        weight_updates = np.matmul(error_deriv, inputs.T)
        bias_updates = error_deriv
        hidden_updates = np.matmul(self.__weights.T, error_deriv)

        self.__weights -= lr * weight_updates
        self.__bias -= lr * bias_updates

        return hidden_updates

class Model:
    def __init__(self, loss_function, *layers):
        self.__loss_function = loss_function
        self.__layers = layers

    def predict(self, inputs):
        x = inputs
        for layer in self.__layers:
            x = layer.predict(x)
        
        return x

    def train(self, inputs, actual, lr=0.01):

        predictions = [inputs]
        x = inputs
        for layer in self.__layers:
            x = layer.predict(x)
            predictions.append(x)

        error_deriv = self.__loss_function(x, actual, deriv=True)

        predictions_train = predictions[::-1]

        for i, layer in enumerate(self.__layers[::-1]): # This will iterate over the layers backwards
            error_deriv = layer.train(predictions_train[i + 1], predictions_train[i], error_deriv, lr)


if __name__ == '__main__':
    inputs = np.array([[[1],
                        [0]],
                       [[0],
                        [1]],
                       [[0],
                        [0]],
                       [[1],
                        [1]]])
    labels = np.array([0, 0, 0, 1])

    model = Model(
        Misc.MSE,
        Dense(2, 4, activation=Misc.relu),
        Dense(4, 6, activation=Misc.relu),
        Dense(6, 1, activation=Misc.sigmoid)
    )

    for _ in range(5000):
        for input, label in zip(inputs, labels):
            model.train(input, label)
    
    for input, label in zip(inputs, labels):
        prediction = model.predict(input)
        print(f"Prediction: {prediction} | Actual: {label}")


