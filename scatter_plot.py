import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Read data from file 'filename.csv'
    df = pd.read_csv('resources/dataset_train.csv')
    first = df.loc[df['Hogwarts House'] == 'Ravenclaw']
    second = df.loc[df['Hogwarts House'] == 'Slytherin']
    three = df.loc[df['Hogwarts House'] == 'Gryffindor']
    four = df.loc[df['Hogwarts House'] == 'Hufflepuff']
    features_name = ["Arithmancy","Astronomy","Herbology","Defense Against the Dark Arts","Divination","Muggle Studies","Ancient Runes","History of Magic","Transfiguration","Potions","Care of Magical Creatures","Charms","Flying"]
    
    plt.xlabel('Grades')
    plt.ylabel('Frequency of students')
    for i in range(12):
        fig = plt.figure(figsize=(15,15))
        sfig.suptitle('Distribution of Students Grades for Each House', fontsize=20)
        plt.subplot(5,3,i + 2)
        Ravenclaw = first[features_name[i]]
        Slytherin = second[features_name[i]]
        Gryffindor = three[features_name[i]]
        Hufflepuff = four[features_name[i]]
        for j in range(i + 1, 13):
            plt.hist(Ravenclaw, color='#00006d', label = 'Ravenclaw')
            plt.hist(Slytherin, color='#00613e', label = 'Slytherin')
            plt.hist(Gryffindor, color='#ae0001', label = 'Gryffindor')
            plt.hist(Hufflepuff, color='#f0c75e', label = 'Hufflepuff')
        plt.title(features_name[i])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.legend(['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'],loc=2,fontsize=16,shadow=True,bbox_to_anchor=(0.05, 0.9))
    plt.savefig('Histogram.png', dpi=300)
    plt.show()