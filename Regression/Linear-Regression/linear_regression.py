import numpy as np
import matplotlib.pyplot as plt

# Linear regression works by comparing the error between the points and the line, then calculating the loss derivative with respect to the parameters of the model, then 
# subtracts a portion of this gradient from the parameter, then does this iteratively until the loss is minimized

# This is the class that will contain all of our code for our linear regression model
class LinearRegression:
    def __init__(self, dims):
        self.dims = dims # This will store the amount of dimensions for our regression problem, or the number of input variables
        
        self.coefficients = np.random.random(dims) # This will create random coefficients for the amount of input variables we have
        self.intercept = np.random.random() # This will act as an intercept for our final line, allowing it to approximate more functions

    # This will make a prediction from our data based on the inputs given
    def predict(self, inputs):
        return np.dot(inputs, self.coefficients) + self.intercept # This will perform our forward operation by multiplying the inputs with their coefficients as well as adding the intercept

    # This function will perform a step towards the minimum error using stochastic gradient descent
    def __step(self, inputs, actual, lr=0.1):
        prediction = self.predict(inputs) # Make a prediction from our inputs
        loss = (prediction - actual)**2 # Show us the mean squared error to be used as a tracking metric
        d_err = prediction - actual # Return for us the derivative of the error with respect to prediction

        self.coefficients -= lr * d_err * inputs # Update the coefficients by subtracting the gradients of the error with respect to the coefficients multiplied by our learning rate
        self.intercept -= lr * d_err # Update the intercept by subtracting the gradient of the error with respect to the intercept multiplied by our learning rate

        return loss # Return the loss for metric tracking

    def fit(self, train_x, train_y, epochs):
        # First we must standardize our data between -1 and 1 to make gradients stable for gradient descent by reducing the variance of the data

        scale_factor_x = np.max([np.max([abs(item) for item in subarray]) for subarray in train_x]) # This will get the max value from all of the x values
        scale_factor_y = np.max([abs(y) for y in train_y]) # This will get the max value from the y values
        scale_factor = np.max([scale_factor_x, scale_factor_y]) # This will get the max value from the max of the x and the y values

        # Scale all of our training data by our scale factor, which will scale down our line without changing the shape besides the intercept
        # We will fit our data to this line instead of the raw unscaled data, but this line will approximate the values for the raw unscaled data
        train_x = np.asarray(train_x) / scale_factor
        train_y = np.asarray(train_y) / scale_factor

        # Main training loop and tracking of loss
        loss_tracker = []
        for _ in range(epochs):
            for x_label, y_label in zip(train_x, train_y):
                loss = self.__step(x_label, y_label) # Take a step given this data and its training label
                loss_tracker.append(loss.item()) # Append the loss to be observed later

        model.intercept *= scale_factor # We have to multiply out intercept by our scale factor to approximate the raw values as it was scaled down during out standardization

        return loss_tracker # Return the loss tracker to observe the loss throughout the training of the model

if __name__ == '__main__':
    # Test data generation
    x = np.random.randint(-5, 5, 10)
    y = np.random.randint(-5, 5, 10)
    z = [50 * x - 25 * y + 30 for x, y in zip(x, y)] # This is the linear function we are going to appeoximate (z = 3x + 2y + 3)

    # Group the test data and shuffle it
    data_grouped = [([x, y], z) for x, y, z in zip(x, y, z)]
    np.random.shuffle(data_grouped)

    # Extract the data from the shuffled array into it's own seperate numpy arrays
    train_x = np.asarray([tup[0] for tup in data_grouped])
    train_y = np.asarray([tup[1] for tup in data_grouped])

    # Create our linear regression model that has 2 parameters
    model = LinearRegression(2)

    # Declare our number of epochs to train our model on
    epochs = 100000

    # Fit our model to the data
    loss = model.fit(train_x, train_y, epochs)
    
    # Print out our coefficients
    print(f"Coefficients: {model.coefficients} | Intercept: {model.intercept}")