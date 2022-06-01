# This script is used to prepare the data for training

import numpy as np

def extract_data():
    file = open("training.txt", "r")
    data = file.readlines()
    file.close()

    query_ids = []
    doc_features = []
    doc_scores = []
    for doc in data:
        # If line start with '#' skip it
        if doc[0] == '#':
            continue

        doc = doc.split()

        # Get query_id for the current document
        qid = doc[1].split(':')[1]
        query_ids.append(int(qid))

        # Get features for the current document
        features = doc[2:-2]
        features = [float(feature.split(':')[1]) for feature in features]
        doc_features.append(features)

        # Get the score for the current document
        score = doc[0]
        doc_scores.append(int(score))

    query_ids = np.array(query_ids)
    doc_features = np.array(doc_features)
    doc_scores = np.array(doc_scores)

    return query_ids, doc_features, doc_scores

# print("query_ids\n", query_ids.shape, query_ids)
# print("doc_features\n", doc_features.shape, doc_features)
# print("doc_scores\n", doc_scores.shape, doc_scores)