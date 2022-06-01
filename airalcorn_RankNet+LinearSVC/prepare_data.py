# This script prepares the data to be used for LinearSVC and RankNet

import glob
import numpy as np


def get_data():
    rank_files = glob.glob("data/*_rank.npy")
    suffix_len = len("_rank.npy")

    RERANK = int(open("RERANK.int").read())

    ranks = []
    casenumbers = []
    Xs = []
    ys = []
    for rank_file in rank_files:
        X = np.load(rank_file[:-suffix_len] + "_X.npy")
        casenumbers.append(rank_file[:suffix_len])
        if X.shape[0] != RERANK:
            print(rank_file[:-suffix_len])
            continue
        
        rank = np.load(rank_file)[0]
        print(rank)
        ranks.append(rank)
        y = np.load(rank_file[:-suffix_len] + "_y.npy")
        Xs.append(X)
        ys.append(y)

    ranks = np.array(ranks)
    total_queries = len(ranks)

    return rank_files, suffix_len, RERANK, Xs, ys, ranks, total_queries