# This script was used to assign a random click count for a documenet based on the relevancy score assigned manually
# This was done to add an extra features for the training 

import argparse
import random
from backup import make_backup


def assign_click_count(in_file):
    make_backup(in_file, "training_backup.dat")
    
    training_file = open(in_file, "r")
    lines = training_file.readlines()
    training_file.close()

    new_file = open(in_file, "w")
    for line in lines:
        # If rows beginns with '#' => information row, not actuall training data
        if line[0] == '#':
            new_file.write(line)
            continue

        # Format of a row containing a document:    relevance qid:id 1:v1 2:v2 3:v3 4:v4 5:v5 6:v6 # doc_url
        # 0 -> relevance      1 -> qid      2 -> 1:v1 ....      7 -> 6:v6      8 -> #      9 -> doc_url
        line = line.split(' ')

        click_count_value = get_random_value_based_on_relevance(line[0])
        line.append("7:" + str(click_count_value))

        insert_new_feature_value(line, "7:" + str(click_count_value), 8)

        new_file.write(' '.join(line))

    new_file.close()

def insert_new_feature_value(list, var, i):
    # Reassign all elements at and beyond the insertion point to be the new
    # value, and all but the last value previously there (so size is unchanged)
    list[i:] = var, *list[i:-1]

def get_random_value_based_on_relevance(relevance):
    if relevance == '1':
        return random.randint(0, 9)
    elif relevance == '2':
        return random.randint(10, 19)
    elif relevance == '3':
        return random.randint(20, 29)
    elif relevance == '4':
        return random.randint(30, 39)
    elif relevance == '5':
        return random.randint(40, 49)
    elif relevance == '6':
        return random.randint(50, 59)
    elif relevance == '7':
        return random.randint(60, 69)
    elif relevance == '8':
        return random.randint(70, 79)
    elif relevance == '9':
        return random.randint(80, 89)
    elif relevance == '10':
        return random.randint(90, 99)
    elif relevance == '11':
        return random.randint(200, 300)
    elif relevance == '12':
        return random.randint(300, 500)
    elif relevance == '13':
        return random.randint(500, 800)
    elif relevance == '14':
        return random.randint(800, 1200)
    elif relevance == '15':
        return random.randint(1200, 1700)
    elif relevance == '16':
        return random.randint(1700, 17000)
    elif relevance == '17':
        return random.randint(17000, 180000)
    elif relevance == '18':
        return random.randint(180000, 1900000)
    elif relevance == '19':
        return random.randint(1900000, 20000000)
    elif relevance == '20':
        return random.randint(20000000, 10000000000)
    else: return random.randint(0, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--TrainingFile', help="name of the file with training data", default = "training.dat")

    args = parser.parse_args()
    
    assign_click_count(args.TrainingFile)
