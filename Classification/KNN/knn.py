import pandas as pd
import numpy as np
from collections import Counter
from sklearn import datasets

class KNN:
    def __init__(self, k):
        self.k = k

    def __eDistance(self, arr1, arr2):
        diff = arr1 - arr2 # This will subtract all the elements from one array from the elements in the nother
        distance = np.sqrt(np.dot(diff, diff)) # This will give us the square root of the sum of the squares of these differences aka the distance

        return distance

    # Compare the input to all other values in the dataset using the euclidean distance and make a classify the value based on 
    def classify(self, test_x_sample, train_x, train_y):
        distance_array = [] # This will contain our distances for each comparison in our train set
        # Iterate over all of the training samples and get the distance between the test sample and the sample from that row then append it to that list
        for i in range(len(train_x)):
            distance = self.__eDistance(test_x_sample, train_x[i])
            distance_array.append(distance)

        # Create a new dataframe containing the distances and the labels that go along with them, then sort by the distances
        df = pd.DataFrame()
        df['distance'] = np.asarray(distance_array)
        df['labels'] = train_y
        df = df.sort_values(by='distance')

        # Select the label that occurs the most in the 'k' shortest distances
        classification = Counter(df['labels'][:k]).most_common()[0][0]

        # Return the classification
        return classification

    # This will return an overall accuracy of our model to determine how well it is doing
    def evaluate(self, test_x, test_y, train_x, train_y):
        test_size = len(test_x) # This will be the size of our test dataset
        correct = 0 # This will be the amount that we classified correctly out of our test set
        for i in range(test_size):
            # Perform the classification and compare it to it's label. If they match then add one to the correct count
            classification = self.classify(test_x[i], train_x, train_y) 
            if classification == test_y[i]:
                correct += 1
        
        # This will be how many test examples the model classified correctly out of our total test set
        accuracy = correct / test_size

        # Return our accuracy score
        return accuracy

if __name__ == '__main__':
    iris = datasets.load_iris() # Load in the iris dataset to perform classification on

    # Create a dataframe out of our iris data, then add a column containing the labels, then shuffle the dataframe
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
    df['target'] = iris.target
    df = df.sample(frac=1)

    # Create 2 new numpy arrays which will store our data for each row in our dataframe in one array and the label of that row in another
    data = np.asarray([df.iloc[i, :-1] for i in range(len(df.index))])
    labels = np.asarray(df['target'])

    # Create our comparison dataset and our test dataset
    train_x = data[:50]
    train_y = labels[:50]

    test_x = data[50:]
    test_y = labels[50:]

    # Select our 'k' value, which in this case will be the square root of the number of examples we have in our train set, then build our model with it
    k = int(np.sqrt(len(train_x)))
    model = KNN(k)

    # Perform our evaluation to get the accuracy of our model
    accuracy = model.evaluate(test_x, test_y, train_x, train_y)

    # Make a prediction to test the model, then print this prediction with its label and the overall accuracy of the model
    index = 6
    prediction = model.classify(test_x[index], train_x, train_y)
    print(f"Prediction: {prediction} | Actual: {test_y[index]} | Model accuracy: {accuracy}")