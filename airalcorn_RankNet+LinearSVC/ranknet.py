# This script: 
#  - prepares the data to train for RankNet
#  - trains RankNet model
#  - prints a comparison between RankNet and BM25 rankings


from prepare_data import get_data
from scipy.stats import rankdata
import numpy as np
import json

rank_files, suffix_len, RERANK, Xs, ys, ranks, total_queries = get_data()

Xs = []
for rank_file in rank_files:
    X = np.load(rank_file[:-suffix_len] + "_X.npy")
    if X.shape[0] != RERANK:
        print(rank_file[:-suffix_len])
        continue

    rank = np.load(rank_file)[0]
    pos_example = X[rank - 1]
    for (i, neg_example) in enumerate(X):
        if i == rank - 1:
            continue
        Xs.append(np.concatenate((pos_example, neg_example)))

X = np.stack(Xs)
dim = int(X.shape[1] / 2)

train_per = 0.8
train_cutoff = int(train_per * len(ranks)) * (RERANK - 1)

train_X = X[:train_cutoff]
test_X = X[train_cutoff:]



from keras import backend
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Add, Dense, Input, Lambda
from keras.models import Model
from tensorflow.keras.utils import plot_model

y = np.ones((train_X.shape[0], 1))

INPUT_DIM = 7
h_1_dim = 64
h_2_dim = h_1_dim // 2
h_3_dim = h_2_dim // 2

# Model.
h_1 = Dense(h_1_dim, activation = "relu")
h_2 = Dense(h_2_dim, activation = "relu")
h_3 = Dense(h_3_dim, activation = "relu")
s = Dense(1)

# Relevant document score.
rel_doc = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_rel = h_1(rel_doc)
h_2_rel = h_2(h_1_rel)
h_3_rel = h_3(h_2_rel)
rel_score = s(h_3_rel)

# Irrelevant document score.
irr_doc = Input(shape = (INPUT_DIM, ), dtype = "float32")
h_1_irr = h_1(irr_doc)
h_2_irr = h_2(h_1_irr)
h_3_irr = h_3(h_2_irr)
irr_score = s(h_3_irr)

# Subtract scores.
negated_irr_score = Lambda(lambda x: -1 * x, output_shape = (1, ))(irr_score)
diff = Add()([rel_score, negated_irr_score])

# Pass difference through sigmoid function.
prob = Activation("sigmoid")(diff)

# Build model.
model = Model(inputs = [rel_doc, irr_doc], outputs = prob)
model.compile(optimizer = "adagrad", loss = "binary_crossentropy")

plot_model(model, show_shapes=False)

NUM_EPOCHS = 50
BATCH_SIZE = 4
checkpointer = ModelCheckpoint(filepath = "valid_params.h5", verbose = 1, save_best_only = True)
history = model.fit([train_X[:, :dim], train_X[:, dim:]], y,
                     epochs = NUM_EPOCHS, batch_size = BATCH_SIZE, validation_split = 0.2,
                     callbacks = [checkpointer], verbose = 2)

import matplotlib.pyplot as plt

# function for plotting loss
def plot_metrics(train_metric, val_metric=None, metric_name=None, title=None, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(train_metric,color='blue',label=metric_name)
    if val_metric is not None: plt.plot(val_metric,color='green',label='val_' + metric_name)
    plt.legend(loc="upper right")
    plt.show()

# plot loss history
plot_metrics(history.history['loss'], history.history['val_loss'], "Loss", "Loss", ylim=1700000.0)


model.load_weights("valid_params.h5")
get_score = backend.function([rel_doc], [rel_score])
n_test = int(test_X.shape[0] / (RERANK - 1))
new_ranks = []
for i in range(n_test):
    start = i * (RERANK - 1)
    end = start + (RERANK - 1)
    pos_score = get_score([test_X[start, :dim].reshape(1, dim)])[0]
    neg_scores = get_score([test_X[start:end, dim:]])[0]

    scores = np.concatenate((pos_score, neg_scores))
    score_ranks = rankdata(-scores)
    new_rank = score_ranks[0]
    new_ranks.append(new_rank)

new_ranks = np.array(new_ranks)
print("Total Queries: {0}".format(n_test))
print("Top 1: {0}".format((new_ranks == 1).sum() / n_test))
print("Top 3: {0}".format((new_ranks <= 3).sum() / n_test))
print("Top 5: {0}".format((new_ranks <= 5).sum() / n_test))
print("Top 10: {0}".format((new_ranks <= 10).sum() / n_test))

# Compare to BM25.
old_ranks = ranks[-n_test:]
print("Total Queries: {0}".format(n_test))
print("Top 1: {0}".format((old_ranks == 1).sum() / n_test))
print("Top 3: {0}".format((old_ranks <= 3).sum() / n_test))
print("Top 5: {0}".format((old_ranks <= 5).sum() / n_test))
print("Top 10: {0}".format((old_ranks <= 10).sum() / n_test))


weights = model.get_weights()
solr_model = {"store" : "myfeature_store",
              "name" : "my_airalcorn_ranknet_model",
              "class" : "org.apache.solr.ltr.model.NeuralNetworkModel",
              "features" : [
                { "name" : "originalScore" },
                { "name" : "titleLength" },
                { "name" : "contentLength" },
                { "name" : "titleScore" },
                { "name" : "contentScore" },
                { "name" : "freshness" },
                { "name" : "clickCount" }
              ],
              "params": {}}
layers = []
layers.append({"matrix": weights[0].T.tolist(),
               "bias": weights[1].tolist(),
               "activation": "relu"})
layers.append({"matrix": weights[2].T.tolist(),
               "bias": weights[3].tolist(),
               "activation": "relu"})
layers.append({"matrix": weights[4].T.tolist(),
              "bias": weights[5].tolist(),
              "activation": "relu"})
layers.append({"matrix": weights[6].T.tolist(),
              "bias": weights[7].tolist(),
              "activation": "identity"})
solr_model["params"]["layers"] = layers

with open("my_ranknet_model.json", "w") as out:
    json.dump(solr_model, out, indent = 4)