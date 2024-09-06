import numpy as np
import pandas as pd

data = pd.read_csv("./data/clean_weather.csv", index_col = 0)
data = data.ffill()
tInputs = np.array([np.array(data["tmax"]), np.array(data["tmin"])]).T
tOutputs = np.array([data["tmax_tomorrow"]])
tInputsNorm = (tInputs - np.min(tInputs, axis=0)) / (np.max(tInputs, axis=0) - np.min(tInputs, axis=0))
tOutputsNorm = (tOutputs - np.min(tOutputs)) / (np.max(tOutputs) - np.min(tOutputs))
learning_rate = 0.01

def sigmoid (x):
    return (2 / (1 + np.exp(-x))) - 1

def calculateError (expected, outputs):
    return expected - outputs

def trainNeuralNetwork (iterations, synapticWeights):
    weights = synapticWeights
    for iteration in range(iterations):
        outputs = np.dot(tInputsNorm, weights)
        error = calculateError(tOutputsNorm.reshape(-1, 1), outputs)
        adjustments = np.dot(tInputsNorm.T, error) * learning_rate / len(tInputsNorm)
        weights += adjustments
    print(f"La presición del programa es del : {round((1 - np.mean(np.abs(error))) * 100, 2)}%")
    return weights

def predict (inputs, synapticWeights):
    inputsNorm = (inputs - np.min(tInputs, axis=0)) / (np.max(tInputs, axis=0) - np.min(tInputs, axis=0))
    weightedSum = np.dot(inputsNorm, synapticWeights)
    outputs = weightedSum * (np.max(tOutputs) - np.min(tOutputs)) + np.min(tOutputs)
    return outputs