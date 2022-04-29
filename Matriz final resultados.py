# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 22:44:45 2022

@author: Lbasper
"""

#imagen
import matplotlib.pyplot as plt

plt.figure()
plt.errorbar(R2f, P0f, yerr=er_P0, xerr=er_R2, fmt='o')
for i in range(31):
    plt.annotate(Nombres[i],(R2f[i], P0f[i]),fontsize=8)
plt.xlim([0,1])
plt.xlabel("$R^2$")
plt.ylabel("Importancia Doctorado")

plt.figure()
plt.errorbar(R2f, P1f, yerr=er_P1, xerr=er_R2, fmt='o')
for i in range(31):
    plt.annotate(Nombres[i],(R2f[i], P1f[i]),fontsize=8)
plt.xlim([0,1])
plt.xlabel("$R^2$")
plt.ylabel("Importancia Maestria")

plt.figure()
plt.errorbar(R2f, P2f, yerr=er_P2, xerr=er_R2, fmt='o')
for i in range(31):
    plt.annotate(Nombres[i],(R2f[i], P2f[i]),fontsize=8)
plt.xlim([0,1])
plt.xlabel("$R^2$")
plt.ylabel("Importancia Especializacion")

plt.figure()
plt.errorbar(R2f, P3f, yerr=er_P3, xerr=er_R2, fmt='o')
for i in range(31):
    plt.annotate(Nombres[i],(R2f[i], P3f[i]),fontsize=8)
plt.xlim([0,1])
plt.xlabel("$R^2$")
plt.ylabel("Importancia Hombre catedra")

plt.figure()
plt.errorbar(R2f, P4f, yerr=er_P4, xerr=er_R2, fmt='o')
for i in range(31):
    plt.annotate(Nombres[i],(R2f[i], P4f[i]),fontsize=8)
plt.xlim([0,1])
plt.xlabel("$R^2$")
plt.ylabel("Importancia Mujer catedra")

plt.figure()
plt.errorbar(R2f, P5f, yerr=er_P5, xerr=er_R2, fmt='o')
for i in range(31):
    plt.annotate(Nombres[i],(R2f[i], P5f[i]),fontsize=8)
plt.xlim([0,1])
plt.xlabel("$R^2$")
plt.ylabel("Importancia Hombre planta")

plt.figure()
plt.errorbar(R2f, P6f, yerr=er_P6, xerr=er_R2, fmt='o')
for i in range(31):
    plt.annotate(Nombres[i],(R2f[i], P6f[i]),fontsize=8)
plt.xlim([0,1])
plt.xlabel("$R^2$")
plt.ylabel("Importancia Mujer planta")