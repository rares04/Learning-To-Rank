# This script is an implementation of RankNet, called factorised RankNet
# Microsoft published a paper Learning to Rank with Nonsmooth Cost Functions which introduced a speedup version of RankNet, called factorised RankNet
# This script uses generated data and not the training set built for the UBB Search Engine

import tensorflow as tf
from tensorflow.keras import layers, activations, losses, Model, Input
from tensorflow.nn import leaky_relu
import numpy as np
from itertools import combinations
from tensorflow.keras.utils import plot_model
import time
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def plot_metrics(train_metric, val_metric=None, metric_name=None, title=None, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(train_metric,color='blue',label=metric_name)
    if val_metric is not None: plt.plot(val_metric,color='green',label='val_' + metric_name)
    plt.legend(loc="upper right")
    plt.show()
    
# generate data
def generate_data(nb_query=25, mean_doc_per_query=10):
    query = np.repeat(np.arange(nb_query)+1, np.ceil(np.abs(np.random.normal(mean_doc_per_query, size=nb_query, scale=0.15*mean_doc_per_query))+2).astype(np.int))
    doc_features = np.random.random((len(query), 10))
    doc_scores = np.random.randint(5, size=len(query)).astype(np.float32)

    # put data into pairs
    pair_id = []
    pair_query_id = []
    for q in np.unique(query):
        query_idx = np.where(query == q)[0]
        for pair_idx in combinations(query_idx, 2):
            pair_query_id.append(q)        
            pair_id.append(pair_idx)

    pair_id = np.array(pair_id)
    pair_query_id = np.array(pair_query_id)

    pair_id_train, pair_id_test, pair_query_id_train, pair_query_id_test = train_test_split(pair_id, pair_query_id, test_size=0.2, stratify=pair_query_id)
    
    return query, doc_features, doc_scores, pair_id, pair_id_train, pair_id_test, pair_query_id, pair_query_id_train, pair_query_id_test

def get_data(query_id, pair_id, pair_query_id):
    if type(query_id) is not np.ndarray:
        query_id = np.array([query_id]).ravel()
    _ind = np.hstack([np.where(query==i) for i in query_id]).ravel()

    q_unique, q_index, q_cnt  = np.unique(query, return_index=True, return_counts=True)
    doc_cnt = q_cnt[np.searchsorted(q_unique, query_id)].sum()
    x = doc_features[_ind]
    score = doc_scores[_ind]
    
    mask = np.zeros((doc_cnt, doc_cnt), dtype=np.float32)
    _, new_q_index = np.unique(query[_ind], return_index=True)
    _pair_id = np.vstack([pair_id[np.where(pair_query_id==i)] - q_index[q_unique==i] + new_q_index[query_id==i] for i in query_id])
    mask[_pair_id[:,0], _pair_id[:,1]] = 1

    return tf.convert_to_tensor(x, dtype=tf.float32), tf.convert_to_tensor(score, dtype=tf.float32), tf.convert_to_tensor(mask, dtype=tf.float32), tf.convert_to_tensor(doc_cnt, dtype=tf.float32)


class FactorisedRankNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = [layers.Dense(16, activation=leaky_relu), layers.Dense(8, activation=leaky_relu)]
        self.o = layers.Dense(1, activation='linear')
    
    def call(self, inputs):
        xi = inputs
        densei = self.dense[0](xi)
        for dense in self.dense[1:]:
            densei = dense(densei)
        oi = self.o(densei)
        return oi
    
    def build_graph(self):
        x = tf.keras.Input(shape=(10))
        return tf.keras.Model(inputs=x, outputs=self.call(x))

tf.keras.utils.plot_model(FactorisedRankNet().build_graph(), show_shapes=False)


def apply_gradient_factorised(optimizer, model, x, score, mask, doc_cnt):
    with tf.GradientTape() as tape:
        oi = model(x)
    
    S_ij = tf.maximum(tf.minimum(tf.subtract(tf.expand_dims(score,1), score),1.),-1.)
    P_ij = tf.multiply(mask, tf.multiply(0.5, tf.add(1., S_ij)))
    P_ij_pred = tf.multiply(mask,tf.nn.sigmoid(tf.subtract(oi, tf.transpose(oi))))
    lambda_ij = tf.add(tf.negative(P_ij), P_ij_pred)
    lambda_i = tf.reduce_sum(lambda_ij,1) - tf.reduce_sum(lambda_ij,0)
    
    doi_dwk = tape.jacobian(oi, model.trainable_weights)
    
    # 1. reshape lambda_i to match the rank of the corresponding doi_dwk
    # 2. multiple reshaped lambda_i with the corresponding doi_dwk
    # 3. compute the sum across first 2 dimensions
    gradients = list(map(lambda k: 
                         tf.reduce_sum(tf.multiply(tf.reshape(lambda_i,  tf.concat([tf.shape(lambda_i),tf.ones(tf.rank(k) - 1, dtype=tf.int32)], axis=-1)), k), [0,1]),
                         doi_dwk))
    
    # model could still be trained without calculating the loss below
    valid_pair_cnt = tf.reduce_sum(mask)
    loss_value = tf.reduce_sum(tf.keras.losses.binary_crossentropy(P_ij, P_ij_pred))
    loss_value = tf.multiply(loss_value, doc_cnt)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    
    return oi, loss_value


    # this function will do update per query
def train_data_for_one_epoch_factorised(optimizer, model, batch_size=1, apply_gradient=apply_gradient_factorised):
    losses = []
    
    pb_i = Progbar(np.ceil(len(np.unique(query)) // batch_size), stateful_metrics=['loss'])
    _query = np.unique(query)
    np.random.shuffle(_query)
    for step, query_id in enumerate(_query):
        x, score, mask, doc_cnt= get_data(query_id, pair_id_train, pair_query_id_train)
        y_pred, loss_value = apply_gradient(optimizer, model, x, score, mask, doc_cnt)
        losses.append(loss_value)

        pb_i.add(1)
    return losses


def compute_val_loss_factorised(model):
    losses = []

    x, score, mask, doc_cnt = get_data(np.unique(query), pair_id_test, pair_query_id_test)
    oi = model(x)

    S_ij = tf.maximum(tf.minimum(tf.subtract(tf.expand_dims(score,1), score),1.),-1.)
    P_ij = tf.multiply(mask, tf.multiply(0.5, tf.add(1., S_ij)))
    P_ij_pred = tf.multiply(mask,tf.nn.sigmoid(tf.subtract(oi, tf.transpose(oi))))
    valid_pair_cnt = tf.reduce_sum(mask)
    loss_value = tf.reduce_sum(tf.keras.losses.binary_crossentropy(P_ij, P_ij_pred))
    loss_value = tf.divide(tf.multiply(loss_value, doc_cnt), valid_pair_cnt)

    losses.append(loss_value)
    return losses

nb_query = 100
mean_doc_per_query = 50
query, doc_features, doc_scores, pair_id, pair_id_train, pair_id_test, pair_query_id, pair_query_id_train, pair_query_id_test = generate_data(nb_query, mean_doc_per_query)

# init optimizer
optimizer = tf.keras.optimizers.Adam()

# start training
fac_ranknet = FactorisedRankNet()

epochs = 1000
loss_train_history = []
loss_val_history = []

apply_gradient_graph = tf.function(apply_gradient_factorised, experimental_relax_shapes=True)

for epoch in range(epochs):
    print('Epoch %d/%d'%(epoch+1, epochs))
    losses_train = train_data_for_one_epoch_factorised(optimizer, fac_ranknet, apply_gradient=apply_gradient_graph)
    loss_train_history.append(np.sum(losses_train)/pair_id_train.shape[0])
    loss_val_history.append(np.mean(compute_val_loss_factorised(fac_ranknet)))
    print('Train loss: %.4f  Validation Loss: %.4f' % (float(loss_train_history[-1]), float(loss_val_history[-1])))

plot_metrics(loss_train_history, loss_val_history, 'loss', 'loss_debug', ylim=1.0)