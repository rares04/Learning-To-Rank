# This script will print statistics regarding Solr default BM25 Rankings
# It will print how many queries have first relevant document as top1 result, as top3 result, as top5 result and as top10 results

from prepare_data import get_data


rank_files, suffix_len, RERANK, Xs, ys, ranks, total_queries = get_data()

print("Total Queries: {0}".format(total_queries))
print("Top 1: {0}".format((ranks == 1).sum() / total_queries))
print("Top 3: {0}".format((ranks <= 3).sum() / total_queries))
print("Top 5: {0}".format((ranks <= 5).sum() / total_queries))
print("Top 10: {0}".format((ranks <= 10).sum() / total_queries))