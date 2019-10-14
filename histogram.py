import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tools

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    try:
        with open('resources/dataset_train.csv', 'r') as csvfile:
            df = tools.read_file(csvfile)   
    except:
        print("Usage: python3 describe resources/dataset_train.csv")
        exit (-1)
    first = df[df['Hogwarts House'] == 'Ravenclaw']
    second = df[df['Hogwarts House'] == 'Slytherin']
    tree = df[df['Hogwarts House'] == 'Gryffindor']
    four = df[df['Hogwarts House'] == 'Hufflepuff']
    plt.figure()
    #plt.legend(['Male', 'Female'])
    plt.title('Male and Female indicator by gender')
    df.plot.hist(alpha=0.5)
    #first.hist(column='Arithmancy')
    #second.hist(column='Arithmancy')
    plt.show()