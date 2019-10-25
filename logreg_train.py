import pandas as pd
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def logistic_func(theta, X): 
    "logistic(sigmoid) function"
    return 1.0 / (1 + np.exp(-1 * np.dot(X, theta.T))) 

def log_gradient(theta, X, y): 
    "logistic gradient function"
    first_calc = logistic_func(theta, X) - y.reshape(X.shape[0], -1)
    final_calc = np.dot(first_calc.T, X) 
    return final_calc 

def cost_func(theta, X, y):
    "cost function, J "

    log_func_v = logistic_func(theta, X)
    step1 = y * np.log(log_func_v)
    step2 = (1 - y) * np.log(1 - log_func_v)
    if np.isnan(step2):
        step2 = 0
    final = -step1 - step2
    return np.mean(final)

def grad_desc(X, value, lr=5e-05):
    "gradient descent function"
    theta = np.matrix(np.zeros(X.shape[1])) 
    costs = []
    for i in range(100):
        cost = cost_func(theta, X, value)
        theta = theta - (lr * log_gradient(theta, X, value))
        costs.append(cost)
    plt.plot(costs)
    return theta

def scale_data(df):
    return (df - df.mean()) / df.std()

def logistic_regression(houseName, df):
    value = []
    for house in df['Hogwarts House']:
        if house == houseName:
            value.append(1)
        else:
            value.append(0)
    value  = np.asarray(value)
    df.drop(['Hogwarts House'], axis=1, inplace=True)
    "scale data"
    df = scale_data(df)
    "important ;)"
    X = np.hstack((np.matrix(np.ones(df.shape[0])).T, df))
    theta = grad_desc(X, value)
    return theta


if __name__ == '__main__':
    np.seterr(divide = 'ignore')
    try:
        "Read parameter file"
        df = pd.read_csv(sys.argv[1])
        "Clean data"
        df.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand', 'Astronomy', 'Transfiguration', 'Care of Magical Creatures', 'Potions'], axis=1, inplace=True)
        "Apply Multi-classification with logistic regression: one-vs-all"
        theta_dic = {}
        "Data of Ravenclaw house"
        df = df.dropna()
        theta_dic['Ravenclaw'] = logistic_regression('Ravenclaw', df.copy()).tolist()[0]
        theta_dic['Slytherin'] = logistic_regression('Slytherin', df.copy()).tolist()[0]
        theta_dic['Gryffindor'] = logistic_regression('Gryffindor', df.copy()).tolist()[0]
        theta_dic['Hufflepuff'] = logistic_regression('Hufflepuff', df.copy()).tolist()[0]
        theta = pd.DataFrame.from_dict(theta_dic)
        mean = pd.DataFrame(df.mean().tolist(), columns= ['Mean'])
        std = pd.DataFrame(df.std().tolist(), columns= ['Std'])
        mean = mean.join(std)
        theta = theta.join(mean)
        theta.to_csv('weights.csv', index = None, header=True)
    except:
        print("Usage: python3 logreg_train.py resources/dataset_train.csv")
        exit (-1)
