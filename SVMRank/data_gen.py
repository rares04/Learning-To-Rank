# Script used to generate training data
# For a list of queries it will call the Solr API and retrieve the features for each document
# The order of rankings is the Solr default one, using BM25

import argparse, sys
import urllib.request
import json

def backup(outFile):
    # File pointer is currently at the end of the file, because it was open in 'a+' mode
    # Put it at the beginning
    outFile.seek(0)

    with open("backup.dat", "w+") as backupFile:
        data = outFile.read()
        backupFile.write(data)

    # Place file pointer back at the end
    outFile.seek(0, 2)

def get_last_queryId(outFile):
    # File pointer is currently at the end of the file, because it was open in 'a+' mode
    # Put it at the beginning
    outFile.seek(0)

    last_queryId = 0
    for line in outFile:
        if line[0] == '#':
            last_queryId = line.split(' ')[1][1:]  # Get QueryId
    
    return int(last_queryId)

def main(queries, outFile, host, port, N):
    # Do a backup before adding new query results
    # backup(outFile)

    # Get id of last query from the already existing ones
    last_queryId = get_last_queryId(outFile)
    print(last_queryId)
    for (qNum, query) in enumerate(queries):
        query = query.replace(" ", "+")
        outFile.write(
            "# Q%d - %s\n" % (
                qNum + 1 + last_queryId, query
            )
        )
        response = urllib.request.urlopen(
            "http://%s:%d/solr/nutch/select?wt=json&%sq=%s&fl=title,url,score,[features%%20efi.query=%s%%20store=myfeature_store]" % (
                host,
                port,
                "rows=%d&" % N if N > 0 else "",
                query,
                query
            )
        )
        if response.getcode() != 200:
            raise Exception("Request Failed!\nCode %d: %s" % (
                response.getcode(),
                response.read()
            ))

        solrResp = json.loads(
            response.read()
        )

        relevance = len(solrResp["response"]["docs"])
        for doc in solrResp["response"]["docs"]:
            features = sorted([f.split("=") for f in doc["[features]"].split(",")])
            features = ["%d:%s" % (i + 1, v) for (i, (_, v)) in enumerate(features)]
            outFile.write("%d qid:%d %s # %s\n" % (
                relevance,
                qNum + 1 + last_queryId,
                " ".join(features),
                doc["url"]
            ))
            relevance -= 1
    outFile.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generates query with corresponding responses')
    parser.add_argument('-n', '--documents_per_query', type=int, default=-1)
    parser.add_argument('-q', '--queries', nargs='*', type=str)
    parser.add_argument('-o', '--out_file', nargs='?', type=argparse.FileType('a+'), default=sys.stdout)
    parser.add_argument('-i', '--in_file', type=argparse.FileType('r'))
    parser.add_argument('-H', '--host', type=str, default="localhost")
    parser.add_argument('-P', '--port', type=int, default=8983)

    args = parser.parse_args()

    queries = args.queries or []
    if args.in_file:
        additions = [ line.strip() for line in args.in_file if line.strip() ]
        queries.extend(additions)
    
    if len(queries) == 0:
        print("Please provide 1 or more queries via --abfragen or --in_file", file=sys.stderr)
        exit(1)

    print(queries)
    main(queries, args.out_file, args.host, args.port, args.documents_per_query)
    