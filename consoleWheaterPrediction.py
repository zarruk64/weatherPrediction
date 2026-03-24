import weatherPrediction as wp
import temperatureCalculator as tc
import networkVisualizer as nv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

temp = ["C", "F"]

def main():
    synapticWeights = wp.trainNeuralNetwork(1)
    entrenar = input("Desea entrenar el programa (Si o No): \n")
    if entrenar == "Si":
        iterations = int(input("Digite la cantidad para entrenar al modelo: \n"))
        print("Entrenando el programa ...")
        synapticWeights = wp.trainNeuralNetwork(iterations)
        print("El programa se ha terminado de entrenar.")

    opt = int(input("Elija una opción:\n1. Celsius\n2. Fahrenheit\n"))
    tMax, tMin = [], []

    for i in range(int(input("Digite cuantos días quiere predecir: \n"))):
        if opt == 1:
            tMax.append(tc.celsiusToFahrenheit(float(input(f"Día {i+1} - Temp máxima (°C): \n"))))
            tMin.append(tc.celsiusToFahrenheit(float(input(f"Día {i+1} - Temp mínima (°C): \n"))))
        else:
            tMax.append(float(input(f"Día {i+1} - Temp máxima (°F): \n")))
            tMin.append(float(input(f"Día {i+1} - Temp mínima (°F): \n")))

    inputs = pd.DataFrame({"tmax": tMax, "tmin": tMin})
    prediction = wp.predict(inputs, synapticWeights)

    for i in range(len(prediction)):
        pred_f = prediction[i][0]
        pred = round(tc.fahrenheitToCelsius(pred_f)) if opt == 1 else round(pred_f)
        print(f"La temperatura máxima estimada para el día #{i+1} es: {pred} °{temp[opt-1]}")

        # ✅ Dibujar la red con los valores de este día
        input_norm = wp.tInputsNorm[0]  # placeholder shape, reemplazamos abajo
        tmax_n = (tMax[i] - np.min(wp.tInputs[:,0])) / (np.max(wp.tInputs[:,0]) - np.min(wp.tInputs[:,0]))
        tmin_n = (tMin[i] - np.min(wp.tInputs[:,1])) / (np.max(wp.tInputs[:,1]) - np.min(wp.tInputs[:,1]))

        nv.visualizeNetwork(
            synapticWeights,
            sampleInput=[tmax_n, tmin_n],
            sampleRaw=[tMax[i], tMin[i]],
            samplePrediction=float(prediction[i][0]),
            day=i+1,
            unit=temp[opt-1]
        )

main()