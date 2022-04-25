# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 18:39:39 2022

@author: Lbasper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#nivel profesores
datos1 = pd.read_csv("Porcen_doc_2022_1.csv",encoding='latin-1',delimiter=";").fillna(0)
llav1=datos1.keys()
a=datos1[llav1[1]]/datos1[llav1[4]]
b=datos1[llav1[2]]/datos1[llav1[4]]
c=datos1[llav1[3]]/datos1[llav1[4]]
datos1=pd.concat([a,b,c],axis=1, join ='inner')


#sexo profesores de catedra
datos2 = pd.read_csv("Porcen_sexo_catedra_2022_1.csv",encoding='latin-1',delimiter=";").fillna(0)
llav2=datos2.keys()
a=datos2[llav2[2]]/datos2[llav2[4]]
b=datos2[llav2[3]]/datos2[llav2[4]]
datos2=pd.concat([a,b],axis=1, join ='inner').fillna(0)


#sexo profesores de planta
datos3 = pd.read_csv("Porcen_sexo_planta_2022_1.csv",encoding='latin-1',delimiter=";").fillna(0)
llav3=datos3.keys()
a=datos3[llav3[1]]/datos3[llav3[3]]
b=datos3[llav3[2]]/datos3[llav3[3]]
datos3=pd.concat([a,b],axis=1, join ='inner')


datospredic = pd.read_csv("Porcen_est_doble_2022_1.csv",encoding='latin-1',delimiter=";").fillna(0)
llavpredic=datospredic.keys()
sumar=datospredic[llavpredic[1]].sum()
datospredic=datospredic[llavpredic[1]]/sumar

# Columna 1_%doc,2_%maes,3_%esp/pre,4_%hombre_cat,5_%mujeres_cat,6_%hombre_plan,7_%mujer_plan
X=pd.concat([datos1,datos2,datos3],axis=1, join ='inner')


Y=datospredic

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_entreno, Y_entreno, X_testeo, Y_testeo=  train_test_split(X,Y, test_size=0.3)
GB = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01)
GB.fit(X_entreno, Y_entreno)
# r2_puntaje= r2_score(Y_entreno, GB.predict(X_entreno))