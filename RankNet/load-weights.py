# This script is used to extract the weights of the layers to build the model that will be uploaded into Solr

from keras.models import load_model
import json


model = load_model("tensorflow-ranknet-model-20")
print(model.summary())

weights = model.get_weights()

solr_model = {"store" : "myfeature_store",
              "name" : "my_tensorflow_ranknet_model",
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
first_layer_weights = model.layers[0].get_weights()[0]
first_layer_biases  = model.layers[0].get_weights()[1]
layers.append({"matrix": weights[0].T.tolist(),
               "bias": weights[1].tolist(),
               "activation": "leakyrelu"})

second_layer_weights = model.layers[1].get_weights()[0]
second_layer_biases  = model.layers[1].get_weights()[1]
layers.append({"matrix": weights[2].T.tolist(),
               "bias": weights[3].tolist(),
               "activation": "leakyrelu"})

third_layer_weights = model.layers[2].get_weights()[0]
third_layer_biases  = model.layers[2].get_weights()[1]
layers.append({"matrix": weights[4].T.tolist(),
               "bias": weights[5].tolist(),
               "activation": "identity"})

solr_model["params"]["layers"] = layers

with open("my_tensorflow_ranknet_model.json", "w") as out:
    json.dump(solr_model, out, indent = 2)

print('First layer weights:', model.get_weights()[0].shape)
print('First layer biases:', model.get_weights()[1].shape)
print('Second layer weights:', model.get_weights()[2].shape)
print('Second layer biases:', model.get_weights()[3].shape)
print('Third layer weights:', model.get_weights()[4].shape)
print('Third layer biases:', model.get_weights()[5].shape)