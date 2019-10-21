import pandas as pd
import numpy as np
import math
import sys
import matplotlib.pyplot as plt

def logistic_func(theta, X): 
    "logistic(sigmoid) function"
    return 1.0 / (1 + np.exp(-np.dot(X, theta.T))) 

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

def grad_desc(X, value, lr=.01, converge_change=1):
    "gradient descent function"
    theta = np.matrix(np.zeros(X.shape[1])) 
    cost = cost_func(theta, X, value)
    change_cost = 1
    num_iter = 1
    costs = []
    costs.append(cost)
    for i in range(10000):
        old_cost = cost
        theta = theta - (lr * log_gradient(theta, X, value))
        cost = cost_func(theta, X, value)
        change_cost = old_cost - cost 
        num_iter += 1
        costs.append(cost)
    return theta, num_iter

def pred_values(theta, X):
    "function to predict labels"
    pred_prob = logistic_func(theta, X) 
    pred_value = np.where(pred_prob >= .5, 1, 0) 
    return np.squeeze(pred_value)

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
    df.drop(['Hogwarts House'], axis=1, inplace=Truse)
    "scale data"
    df = scale_data(df)
    "important ;)"
    X = np.hstack((np.matrix(np.ones(df.shape[0])).T, df))
    theta, num_iter = grad_desc(X, value)
    return theta


if __name__ == '__main__':
    np.seterr(divide = 'ignore')
    try:
        "Read parameter file"
        df = pd.read_csv(sys.argv[1])
        "Clean data"
        df.drop(['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand'], axis=1, inplace=True)
        print(df)
        "Apply Multi-classification with logistic regression: one-vs-all"
        theta_dic = {}
        "Data of Ravenclaw house"
        df = df.dropna(inplace=True)
        # theta_dic['Ravenclaw'] = logistic_regression('Ravenclaw', df)
        # theta_dic['Slytherin'] = logistic_regression('Slytherin', df)
        # theta_dic['Gryffindor'] = logistic_regression('Gryffindor', df)
        # theta_dic['Hufflepuff'] = logistic_regression('Hufflepuff', df)
    except:
        print("Usage: python3 logreg_train.py resources/dataset_train.csv")
        exit (-1)