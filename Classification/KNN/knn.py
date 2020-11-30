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

    # Compare the input to all other values in the dataset using the euclidean distance
    def classify(self, test_x_sample, train_x, train_y):
        distance_array = []
        for i in range(len(train_x)):
            distance = self.__eDistance(test_x_sample, train_x[i])
            distance_array.append(distance)

        df = pd.DataFrame()
        df['distance'] = np.asarray(distance_array)
        df['labels'] = train_y

        df = df.sort_values(by='distance')

        classification = Counter(df['labels'][:k]).most_common()[0][0]

        return classification

    def evaluate(self, test_x, test_y, train_x, train_y):
        test_size = len(test_x)
        correct = 0
        for i in range(test_size):
            classification = self.classify(test_x[i], train_x, train_y)
            if classification == test_y[i]:
                correct += 1
        
        accuracy = correct / test_size

        return accuracy

if __name__ == '__main__':
    iris = datasets.load_iris()

    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df = df.sample(frac=1)

    data = np.asarray([df.iloc[i, :-1] for i in range(len(df.index))])
    labels = np.asarray(df['target'])

    train_x = data[:50]
    train_y = labels[:50]

    test_x = data[50:]
    test_y = labels[50:]

    k = int(np.sqrt(len(data)))
    model = KNN(k)

    accuracy = model.evaluate(test_x, test_y, train_x, train_y)

    index = 6
    prediction = model.classify(test_x[index], train_x, train_y)
    print(f"Prediction: {prediction} | Actual: {test_y[index]} | Model accuracy: {accuracy}")