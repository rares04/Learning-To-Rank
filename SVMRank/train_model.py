# Script used for training the model with Thorestens SVMRank

from subprocess import call
import argparse, sys
import json
import timeit

modelTemplate = {
  "store" : "myfeature_store",
  "name" : "my_svm_model",
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
      "originalScore" : 0.0,
      "titleLength" : 0.0,
      "contentLength" : 0.0,
      "titleScore" : 0.0,
      "contentScore" : 0.0,
      "freshness" : 0.0,
      "clickCount" : 0.0
    }
  }
}

def main(svmDir, trainingFile, modelFile):
    start = timeit.default_timer()
    call(["%s/svm_rank_learn" % svmDir, "-c", "171", "-w", "1", "-k", "5", trainingFile, "model"], stdout=sys.stderr)

    # Read in the parameters from the generated model file
    params = None
    with open("model") as fmodel:
        lines = fmodel.read().split("\n")
        # We only really care about the last line
        for lineNo in range(-1, -len(lines)-1, -1):
            param_line = lines[lineNo]
            if param_line:
                break
        param_line = param_line.split()
        params = [p.split(":")[1] for p in param_line if ":" in p]

    # Plug the parameters into our template
    model = modelTemplate
    for i in range(len(model["features"])):
        model["params"]["weights"][model["features"][i]["name"]] = params[i]
    # Save the model save the world
    json.dump(model, modelFile)

    stop = timeit.default_timer()
    print('Time elapsed in seconds: ', stop - start)  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the SVMRank model for Solr.')
    parser.add_argument('trainingFile', type=argparse.FileType('r'), help="file with training data")
    parser.add_argument('-m', '--modelOutFile', nargs='?', type=argparse.FileType('w+'), help="where to save the Solr compatible model file", default=sys.stdout)
    parser.add_argument('--svmDir', type=str, help="path to directory where we can find the binary svm_rank_learn", default="svm_rank")

    args = parser.parse_args()
    
    main(args.svmDir, args.trainingFile.name, args.modelOutFile)