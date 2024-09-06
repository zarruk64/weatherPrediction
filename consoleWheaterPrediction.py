import weatherPrediction as wp
import temperatureCalculator as tc
import numpy as np
import pandas as pd

temp = ["C", "F"]

def main():
    iterations = int(input("Digite la cantidad para entrenar al modelo (a mayor cantidad, mejor entrenado): \n"))
    print("Entrenando el programa ...")
    synapticWeights = wp.trainNeuralNetwork(iterations)
    print("El programa se ha terminado de entrenar.")
    opt = int(input("Elija una opción para el funcionamiento del programa:\n1.Celcuis\n2.Fahrenheit\n"))
    tMax = []
    tMin = []
    for i in range(int(input("Digite cuantos días quiere predecir: \n"))):
        if (opt == 1):
            tMax.append(tc.celsiusToFahrenheit(float(input("Digite la temperatura máxima del día anterior a predecir: \n"))))
            tMin.append(tc.celsiusToFahrenheit(float(input("Digite la temperatura minima del día anterior a predecir: \n"))))
        else:
            tMax.append(float(input("Digite la temperatura máxima del día anterior a predecir: \n")))
            tMin.append(float(input("Digite la temperatura minima del día anterior a predecir: \n")))
    inputs = pd.DataFrame({"tmax": tMax, "tmin": tMin})
    prediction = wp.predict(inputs, synapticWeights)
    for i in range(len(prediction)):
        if (opt == 1):
            pred = round(tc.fahrenheitToCelsius(prediction[i][0]))
        print(f"La temperatura máxima estimada para el día #{i+1} es:", pred, f"°{temp[opt - 1]}")
    
main()