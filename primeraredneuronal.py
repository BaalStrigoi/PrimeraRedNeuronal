"""Escenario con una red neuronal, simple, lineal, Y=mx+b
en este caso, la formula de tranformar de celcius a fahrenheit es una ecuacion lineal
por eso lo simple de la red neuronal"""

# -*- coding: utf-8 -*-
import os
os.system('chcp 65001')
import tensorflow as tf
import numpy as np
print(tf.__version__)

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(

    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print("Comenzando Entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs = 1000, verbose = False)
print("Modelo Entrenado")

import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de Perdida")
plt.plot(historial.history["loss"])
plt.show()


print("Hagamos una Prediccion")
resultado = modelo.predict(np.array([[100.0]]))
print("El resultado es: "+ str(resultado[0][0])+" fahrenheit")

print("Variables Internas del Modelo")
print(capa.get_weights())

