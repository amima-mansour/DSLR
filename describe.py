import pandas as pd
import numpy as np
import math

def min(vector):
    x = vector[0]
    for v in vector:
        if x > v:
            x = v
    return x

def max(vector):
    x = vector[0]
    for v in vector:
        if x < v:
            x = v
    return x

def len(vector):
    c = 0
    for x in vector:
        if x is not None:
            c += 1
    return (c)

def print_features(features):
    print(pd.DataFrame.from_dict(features))

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    data = pd.read_csv("resources/dataset_train.csv")
    # convert dataframe to matrix
    data = np.array(data)
    features = {}
    features_name = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    for i in range(13):
        feature = {}
        feature["Count"] = len(data[:,i + 6])
        feature["Mean"] = np.float32(np.nansum(data[:, i + 6] / feature["Count"]))
        feature["Std"] = np.float32(math.sqrt((np.nansum(data[:,i + 6] - feature["Mean"]) ** 2) / feature["Count"]))
        feature["Min"] = np.float32(min(data[:,i + 6]))
        sorted_array = np.sort(data[:,i + 6])
        q = (feature["Count"] + 3) / 4
        if int(q) == q:
            feature["25%"] = sorted_array[q - 1]
        else:
            feature["25%"] = (sorted_array[int(q - 1)] + 3 * sorted_array[int(q)]) / 4
        q = (feature["Count"] + 1) / 2
        if int(q) == q:
            feature["50%"] = sorted_array[q - 1]
        else:
            feature["50%"] = (sorted_array[int(q) - 1] + sorted_array[int(q)]) / 2
        q =(3 * feature["Count"] + 1) / 4
        if int(q) == q:
            feature["75%"] = sorted_array[q - 1]
        else:
            feature["75%"] = (sorted_array[int(q) -1] * 3 + sorted_array[int(q)]) / 4
        feature["Max"] = np.float32(max(data[:,i + 6]))
        features[features_name[i]] = feature
    print_features(features)
