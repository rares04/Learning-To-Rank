# This script was used to update the randomly assigned click counts for documents into Solr
# Solr provides the option to update documents and their properties without reindexing

import requests


def get_click_count_for_documents(in_file):
    training_file = open(in_file, "r")
    lines = training_file.readlines()
    training_file.close()

    docs = {}
    for line in lines:
        if line[0] != '#':
            line = line.split(' ')

            click_count = line[8]
            doc_url = line[10].strip()
            docs[doc_url] = [click_count]

    # print(docs)
    for doc_url, click_count in docs.items():
        if len(click_count) > 1:
            raise Exception(
                "Multiple click_count values per doc not allowed  -  at " + doc_url, click_count)
        else:
            docs[doc_url] = int(click_count[0].split(':')[1])
    return docs


url = 'http://localhost:8983/solr/nutch/update?commit=true'

docs = get_click_count_for_documents("20-rows-queries/training.dat")

for doc_url, click_count in docs.items():
    data = '[{"id": "' + doc_url + '", "clickCount_i": { "set": ' + str(click_count) + '}}]'

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=data, headers=headers)

    res = response.json()

    if res['responseHeader']['status'] != 0:
        print(res)