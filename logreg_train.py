import pandas as pd
import numpy as np
import math
import sys

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
    #print(v.dot(np.log(1 - log_func_v)))
    step1 = y * np.log(log_func_v)
    step2 = (1 - y) * np.log(1 - log_func_v)
    if np.isnan(step2):
        step2 = 0
    final = -step1 - step2
    return np.mean(final)

def grad_desc(df, value, lr=.01, converge_change=.001):
    "gradient descent function"
    X = df.values
    theta = np.matrix(np.zeros(X.shape[1]))
    cost = cost_func(theta, X, value)
    change_cost = 1
    num_iter = 1
    while (change_cost > converge_change):
        print(cost) 
        old_cost = cost
        theta = theta - (lr * log_gradient(theta, X, value))
        cost = cost_func(theta, X, value)
        change_cost = old_cost - cost 
        num_iter += 1
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
    for x in df['Hogwarts House']:
        if x == houseName:
            value.append(1)
        else:
            value.append(0)
    value  = np.asarray(value)
    del df['Hogwarts House']
    "scale data"
    df = scale_data(df)
    theta, num_iter = grad_desc(df, value)
    print(num_iter)
    #print(theta, num_iter)
    return theta


if __name__ == '__main__':
    np.seterr(divide = 'ignore')
    try:
        "Read parameter file"
        df = pd.read_csv(sys.argv[1])
        "Clean data"
        del df['Index']
        del df['First Name']
        del df['Last Name']
        del df['Birthday']
        del df['Best Hand']
        "Apply Multi-classification with logistic regression: one-vs-all"
        theta_dic = {}
        "Data of Ravenclaw house"
        df = df.dropna()
        theta_dic['Ravenclaw'] = logistic_regression('Ravenclaw', df)
    except:
        print("Usage: python3 logreg_train.py resources/dataset_train.csv")
        exit (-1)