import csv
import pandas as pd
import numpy as np 

def empty_cell(data):
    for line in data:
        for i,element in enumerate(line):
            if len(element) == 0:
                line[i] = np.nan
            try:
                line[i] = float(element)
            except:
                continue

def read_file(csvfile):
    data = []
    datareader = csv.reader(csvfile, delimiter = '\t')
    for row in datareader:
        l = row[0].split(',')
        data.append(l)
    empty_cell(data)
    data = pd.DataFrame(data[1:], columns = data[0])
    return data