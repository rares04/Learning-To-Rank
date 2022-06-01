from scipy.stats import rankdata
from sklearn.svm import LinearSVC
import numpy as np
from prepare_data import get_data

rank_files, suffix_len, RERANK, Xs, ys, ranks, total_queries = get_data()

X = np.concatenate(Xs, 0)
y = np.concatenate(ys)
print(len(ranks))
train_per = 0.8
train_cutoff = int(train_per * len(ranks)) * RERANK
train_X = X[:train_cutoff]
train_y = y[:train_cutoff]
test_X = X[train_cutoff:]
test_y = y[train_cutoff:]
model = LinearSVC(dual=False)
model.fit(train_X, train_y)

import json

weights = model.coef_
solr_model = {
  "store" : "myfeature_store",
  "name" : "my_linearsvc_model",
  "class" : "org.apache.solr.ltr.model.LinearModel",
  "features" : [
    { "name" : "originalScore" },
    { "name" : "titleLength" },
    { "name" : "contentLength" },
    { "name" : "titleScore" },
    { "name" : "contentScore" },
    { "name" : "freshness" },
    { "name" : "clickCount"}
  ],
  "params" : {
    "weights" : {
      "originalScore" : weights[0][0],
      "titleLength" : weights[0][1],
      "contentLength" : weights[0][2],
      "titleScore" : weights[0][3],
      "contentScore" : weights[0][4],
      "freshness" : weights[0][5],
      "clickCount" : weights[0][6]
    }
  }
}

with open("my_linearsvm_model.json", "w") as out:
    json.dump(solr_model, out, indent = 4)

preds = model._predict_proba_lr(test_X)

n_test = int(len(test_y) / RERANK)
new_ranks = []
for i in range(n_test):
    start = i * RERANK
    end = start + RERANK
    scores = preds[start:end, 1]
    score_ranks = rankdata(-scores)
    old_rank = np.argmax(test_y[start:end])
    new_rank = score_ranks[old_rank]
    new_ranks.append(new_rank)

new_ranks = np.array(new_ranks)
print("Total Queries: {0}".format(n_test))
print("Top 1: {0}".format((new_ranks == 1).sum() / n_test))
print("Top 3: {0}".format((new_ranks <= 3).sum() / n_test))
print("Top 5: {0}".format((new_ranks <= 5).sum() / n_test))
print("Top 10: {0}".format((new_ranks <= 10).sum() / n_test))