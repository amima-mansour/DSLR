import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tools

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    df = pd.read_csv('resources/dataset_train.csv')
    first = df[df['Hogwarts House'] == 'Ravenclaw']
    second = df[df['Hogwarts House'] == 'Slytherin']
    three = df[df['Hogwarts House'] == 'Gryffindor']
    four = df[df['Hogwarts House'] == 'Hufflepuff']
    plt.figure()
    features_name = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    plt.xlabel('Notes')
    plt.ylabel('Number of students')
    plt.title('Distribution of Students Notes for Each House')
    #plt.xticks(gas.Year[::3])
    for i in range(1):
        Ravenclaw = first[features_name[i]]
        Slytherin = second[features_name[i]]
        Gryffindor = three[features_name[i]]
        Hufflepuff = four[features_name[i]]
        #plt.hist([Ravenclaw,Slytherin,Gryffindor,Hufflepuff], color=['green', 'red', 'yellow', 'orange'], label = ['Ravenclaw','Slytherin','Gryffindor','Hufflepuff'])
        plt.hist(Ravenclaw, color='red', label = ['Ravenclaw'])
        plt.hist(Slytherin, color='orange', label = ['Slytherin'])
        plt.legend()
    plt.savefig('Hstogram.png', dpi=300)
        #course.plot.hist(alpha=0.5)
    #first.hist(column='Arithmancy')
    #second.hist(column='Arithmancy')
    plt.show()