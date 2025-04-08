from re import X
# Liner  Regration from scrach

import pandas as pd
from sklearn.model_selection import train_test_split

# data visulization
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, x_train, y_train):
        num = 0
        den = 0

        x_train = x_train.values.flatten()  # Convert to 1D array
        y_train = y_train.values.flatten()

        for i in range(len(x_train)):
            num = num + ((x_train[i] - x_train.mean()) * (y_train[i] - y_train.mean()))
            den = den + ((x_train[i] - x_train.mean()) ** 2)

            # Slope
        self.m = num / den
        # Intercept
        self.b = y_train.mean() - (self.m * x_train.mean())
        print(f"THE SLOPE IS : {self.m}")
        print(f"THE INTERSEPT  IS : {self.b}")

    def predict(self, x_test):
        return (self.m * x_train + self.b)


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('placement (2).csv')
x = df[['cgpa']]
y = df[['package']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


def main():
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    # data visulization
    plt.scatter(x, y, color='green', label='Data points')
    plt.plot(x_train, lr.predict(x_train), color='red', label='Regression line')

    plt.xlabel('CGPA')
    plt.ylabel('Package')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
