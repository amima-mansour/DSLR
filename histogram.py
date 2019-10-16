import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import seaborn as sns

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Read data from file 'filename.csv'
    df = pd.read_csv('resources/dataset_train.csv')
    first = df[df['Hogwarts House'] == 'Ravenclaw']
    second = df[df['Hogwarts House'] == 'Slytherin']
    three = df[df['Hogwarts House'] == 'Gryffindor']
    four = df[df['Hogwarts House'] == 'Hufflepuff']
    sns.set(color_codes=True)
    features_name = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    fig = plt.figure(figsize=(15,15))
    fig.suptitle('Distribution of Students Grades for Each House', fontsize=20)
    #plt.xticks(gas.Year[::3])
    plt.xlabel('Grades')
    plt.ylabel('Frequency of students')
    for i in range(13):
        plt.subplot(5,3,i + 2)
        Ravenclaw = first[features_name[i]]
        Slytherin = second[features_name[i]]
        Gryffindor = three[features_name[i]]
        Hufflepuff = four[features_name[i]]
        plt.hist(Ravenclaw, color='#32a8a4', label = 'Ravenclaw')
        plt.hist(Slytherin, color='#a36146', label = 'Slytherin')
        plt.hist(Gryffindor, color='#4b9447', label = 'Gryffindor')
        plt.hist(Hufflepuff, color='#c44f4f', label = 'Hufflepuff')
        plt.title(features_name[i])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.legend(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'],loc=2,fontsize=16,shadow=True,bbox_to_anchor=(0.05, 0.9))
    plt.savefig('Histogram.png', dpi=300)
    plt.show()