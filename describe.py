import pandas as pd
import numpy as np
import math
import csv

def get_column_data(vector):
    c, s, max_column, min_column = 0, 0, vector[0], vector[0]
    for x in vector:
        s += x
        if x != 0:
            c += 1
        if max_column < x:
            max_column = x
        if min_column > x:
            min_column = x
    return c, s, max_column, min_column

def get_std(vector, mean, count):
    std = 0
    for v in vector:
        if v != 0:
            std += (v - mean) ** 2
    std /= (count - 1)
    return (std ** 0.5)

def get_quartile(sorted_array, q):
    if q - int(q) == 0.25:
        return (sorted_array[math.ceil(q)] + 3 * sorted_array[math.floor(q)]) / 4
    if q - int(q) == 0.5:
        return (sorted_array[math.ceil(q)] + sorted_array[math.floor(q)]) / 2
    return (sorted_array[math.ceil(q)] * 3 + sorted_array[math.floor(q)]) / 4


def print_features(features):
    print(pd.DataFrame.from_dict(features))

def empty_cell(data):
    for line in data:
        for i,element in enumerate(line):
            if len(element) == 0:
                line[i] = 0
            try:
                line[i] = float(element)
            except:
                continue

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    data = []
    with open('resources/dataset_train.csv', 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter = '\t')
        for row in datareader:
            l = row[0].split(',')
            data.append(l)
    empty_cell(data)
    data = pd.DataFrame(data[1:], columns = data[0])
    #print(data)
    #data = pd.read_csv("resources/dataset_train.csv")
    # convert dataframe to matrix
    features = {}
    features_name = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    for i in range(13):
        feature = {}
        feature["Count"], sum_column, max_column, min_column = get_column_data(data[features_name[i]])
        feature["Mean"] = np.float32(sum_column / feature["Count"])
        feature["Std"] = get_std(data[features_name[i]], feature["Mean"], feature["Count"])
        feature["Min"] = min_column
        sorted_array = np.sort(data[features_name[i]])
        q = (feature["Count"] - 1) * 0.25
        if int(q) == q:
            feature["25%"] = sorted_array[int(q)]
        else:
            feature["25%"] = get_quartile(sorted_array, q)
        q = (feature["Count"] - 1) * 0.5
        if int(q) == q:
            print("je suis rrentre")
            feature["50%"] = sorted_array[int(q)]
        else:
            feature["50%"] = get_quartile(sorted_array, q)
        q = (feature["Count"] -  1) * 0.75
        if int(q) == q:
            feature["75%"] = sorted_array[int(q)]
        else:
            feature["75%"] = get_quartile(sorted_array, q)
        feature["Max"] = max_column
        features[features_name[i]] = feature
    print_features(features)
