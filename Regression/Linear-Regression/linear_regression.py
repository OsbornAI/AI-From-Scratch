import numpy as np
import matplotlib.pyplot as plt

# Linear regression works by comparing the error between the points and the line, then calculating the loss derivative with respect to the parameters of the model, then 
# subtracts a portion of this gradient from the parameter, then does this iteratively until the loss is minimized

# This is the class that will contain all of our code for our linear regression model
class LinearRegression:
    def __init__(self, dims):
        self.dims = dims # This will store the amount of dimensions for our regression problem, or the number of input variables
        
        self.params = np.random.random(dims) # This will create random coefficients for the amount of input variables we have
        self.intercept = np.random.random() # This will act as an intercept for our final line, allowing it to approximate more functions

    # This will make a prediction from our data based on the inputs given
    def predict(self, inputs):
        return np.dot(inputs, self.params) + self.intercept # This will perform our forward operation by multiplying the inputs with their coefficients as well as adding the intercept

    def __step(self, inputs, actual, lr=0.1):
        prediction = self.predict(inputs)
        loss = (prediction - actual)**2
        d_err = prediction - actual

        self.params -= lr * d_err * inputs
        self.intercept -= lr * d_err

        return loss

    def fit(self, train_x, train_y, epochs):
        # First we must standardize our data between -1 and 1 to make gradients stable for gradient descent by reducing the variance of the data

        scale_factor_x = np.max([np.max([abs(item) for item in subarray]) for subarray in train_x])
        scale_factor_y = np.max([abs(y) for y in train_y]) 
        scale_factor = np.max([scale_factor_x, scale_factor_y])

        train_x = np.asarray(train_x) / scale_factor
        train_y = np.asarray(train_y) / scale_factor

        # Main training loop and tracking of loss
        loss_tracker = []
        for _ in range(epochs):
            for x_label, y_label in zip(train_x, train_y):
                loss = self.__step(x_label, y_label)
                loss_tracker.append(loss.item())

        model.intercept *= scale_factor 

        return loss_tracker

if __name__ == '__main__':
    # Test data generation
    x = np.random.randint(-5, 5, 10)
    y = np.random.randint(-5, 5, 10)

    z = [2 * x - 3 * y + 10 for x, y in zip(x, y)] # This is the linear function we are going to appeoximate (z = 3x + 2y + 3)

    # Group the test data and shuffle it
    data_grouped = [([x, y], z) for x, y, z in zip(x, y, z)]
    np.random.shuffle(data_grouped)

    # Extract the data from the shuffled array into it's own seperate numpy arrays
    train_x = np.asarray([tup[0] for tup in data_grouped])
    train_y = np.asarray([tup[1] for tup in data_grouped])

    # Create our linear regression model that has 2 parameters
    model = LinearRegression(2)

    # Declare our number of epochs to train our model on
    epochs = 500

    loss = model.fit(train_x, train_y, epochs)
    
    print(f"Coefficients: {model.params} | Intercept: {model.intercept}")