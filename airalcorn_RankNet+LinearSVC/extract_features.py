# dataset.dat contains the training data in the format used for SVMRank
# This script will extract the data from dataset.dat and create for each query a file with the features, ranks and relevant documents
# This data format is used by airalcorns RankNet architecture


import numpy as np


RERANK = 20
with open("RERANK.int", "w") as f:
    f.write(str(RERANK))

file = open("dataset.dat", "r")
lines = file.readlines()
file.close()

results_features = []
results_targets = []
results_ranks = []    
add_data = False
rank = 0
for line in lines:
    if line[0] == '#':
        if add_data:
            np.save("data/{0}_X.npy".format(q_id), np.array(results_features))
            np.save("data/{0}_y.npy".format(q_id), np.array(results_targets))
            np.save("data/{0}_rank.npy".format(q_id), np.array(results_ranks))

        q_id = line.split(' ')[1][1:]

        results_features = []
        results_targets = []
        results_ranks = []    
        add_data = False
        rank = 0

        continue
    
    doc = line.split(' ')
    features = doc[2:9]
    feature_array = []
    for feature in features:
        feature_array.append(feature.split(":")[1])

    feature_array = np.array(feature_array, dtype = "float32")
    results_features.append(feature_array)

    doc_id = doc[-1]
    print(doc_id)

    # Marked each doc with a relevance score between 1-20. Mark as relevant only those with a score higher than 17
    relevance = int(doc[0])
    if relevance > 17:  
        print(rank, doc_id)
        results_ranks.append(rank + 1)
        results_targets.append(1)
        add_data = True
    else:
        results_targets.append(0)

    rank += 1