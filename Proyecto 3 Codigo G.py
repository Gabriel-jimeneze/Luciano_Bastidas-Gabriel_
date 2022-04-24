# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:25:54 2022

@author: jimen
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

datos = pd.read_csv('')
datos.head()
predictores= datos.keys()
predictores= predictores.drop("")
objetivo = ""
X=datos[predictores]
Y=datos[objetivo]

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_entreno, Y_entreno, X_testeo, Y_testeo=  train_test_split(X,Y, test_size=0.3)
GB = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01)
GB.fit(X_entreno, Y_entreno)
r2_puntaje= r2_score(Y_entreno, GB.predict(X_entreno))
