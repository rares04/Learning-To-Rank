# This script was used to check if same document has different values of click count
# If this is the case then only chose the max value click count for that document
# Line that does the replace job is currently commented out, currently the script only prints if there are documents with different click count values

import argparse
from backup import make_backup


def get_docs_with_multiple_click_counts(in_file):
    make_backup(in_file, "training_backup.dat")
    
    training_file = open(in_file, "r")
    lines = training_file.readlines()
    training_file.close()

    docs = {}
    for line in lines:
        if line[0] != '#':
            line = line.split(' ')

            click_count = line[8]
            doc_url = line[10]

            if doc_url not in docs.keys():
                docs[doc_url] = [click_count]

            elif doc_url in docs.keys():
                if click_count not in docs[doc_url]:
                    docs[doc_url].append(click_count)

    number_docs_very_different_click_count = 0
    number_docs_different_click_count = 0
    for doc_id, click_count_list in docs.items():
        if len(click_count_list) > 1:
            number_docs_different_click_count += 1

            exact_value = [int(x.split(':')[1]) for x in click_count_list]
            min_click_count = min(exact_value)
            max_click_count = max(exact_value)

            if(max_click_count - min_click_count > 20000000):
                number_docs_very_different_click_count += 1
                print(doc_id, click_count_list)
                
    print("Total documents in the dataset: " + str(len(docs)))
    print("Number of documents which have multiple click_count values " + str(number_docs_different_click_count))
    print("Number of documents which have very different click_count values: " + str(number_docs_very_different_click_count))


    return in_file, lines, docs


def assign_only_one_click_count_per_doc(in_file, lines, docs):
    new_file = open(in_file, "w")

    for line in lines:
        if line[0] == '#':
            new_file.write(line)
            continue
        
        line = line.split(' ')
        doc_url = line[10]
        if doc_url in docs.keys():
            click_count_values = [int(x.split(':')[1]) for x in docs[doc_url]]
            max_click_count = max(click_count_values)
            line[8] = '7:'+ str(max_click_count)

        new_file.write(' '.join(line))

    new_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--TrainingFile', help="name of the file with training data", default = "training.dat")

    args = parser.parse_args()

    in_file, lines, docs = get_docs_with_multiple_click_counts(args.TrainingFile)
    # assign_only_one_click_count_per_doc(in_file, lines, docs)
