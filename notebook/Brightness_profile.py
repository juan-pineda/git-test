#!/usr/bin/env python
# coding: utf-8

#librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import ellipses as el
import cv2
import copy


def Draw_galaxia(Datos, cmap = 'viridis', norm = None, Scale = [None, None]):
    #Datos = datos a gráficar
    #cmap =  escala de colores
    #norm = normaliación de la escala
    #Scale = escala de la normalizaión
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.figure(figsize=(12,12))
    ax = plt.gca()
    im = ax.imshow(Datos, cmap=cmap, norm = norm)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    im.set_clim(Scale[0],Scale[1])
    plt.yticks(size = 15)
    plt.show()
    #retorna la gráfica


def Draw_countour(Datos, min_cielo, min_datos):
    #Datos = datos a gráficar
    #min_cielo = límite para tomar las isofotas
    #min_datos = límite superior
    plt.figure(figsize=(12,6))
    plt.contour(Datos, levels = np.linspace(min_cielo,min_datos,1))
    plt.gca().invert_yaxis()
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.show()
    #retorna la gráfica


# In[9]:


def Draw_countour1(Datos, min_cielo, min_datos, numiso = 60 ):
    #Datos = datos a gráficar
    #min_cielo = límite inferior para tomar las isofotas
    #min_datos = límite superior
    #numiso = número de isofotas
    plt.figure(figsize=(12,6))
    cs = plt.contour(Datos, levels = np.linspace(min_datos,min_cielo,numiso))
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    #retorna la información de las isofotas
    return cs



def points(cs):
    #cs = información de las isofotas
    p = 0
    v = 0
    x = []
    y = []

    listasx = []
    listasy = []

    for j in range(0,len(cs.collections)):
        x = []
        y = []
        for i in range(0,len(cs.collections[j].get_paths())):

            p = cs.collections[j].get_paths()[i]
            v = p.vertices
            f = v[:,0]
            h = v[:,1]

            for i in range(0,len(f)):
                x.append(f[i])

            for i in range(0,len(h)):
                y.append(h[i])

        listasx.append(x)
        listasy.append(y)
        #retorna los puntos (x,y) de las isofotas
    return listasx,listasy



def Draw_points(listasx,listasy):
    #listasx = lista de los puntos en x
    #listasy = lista de los puntos en y
    plt.figure(figsize=(12,6))
    plt.plot(listasx[59],listasy[59],'o')
    plt.gca().invert_yaxis()
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.show()
    #retorna la gráfica de  los puntos


def ellipses(listasx,listasy):
    #listasx = lista de los puntos en x
    #listasy = lista de los puntos en y
    data = np.ones(len(listasx)-2)
    centros = np.ones((len(listasx)-2, 2))
    anchos = np.ones(len(listasx)-2)
    altos = np.ones(len(listasx)-2)
    angulos = np.ones(len(listasx)-2)

    for i in range(2,len(listasx)):
        data = [listasx[i],listasy[i]]
        lsqe = el.LSqEllipse()
        lsqe.fit(data)
        center, width, height, phi = lsqe.parameters()
        centros[i-2] = np.asanyarray(center)
        anchos[i-2] = width
        altos[i-2] = height
        angulos[i-2] = phi
    allD = [data, centros, anchos, altos, angulos]
    #retorna una lista de arreglos [(x,y),(xo.yo),[semi_mayor],[semi_menor],[ángulos]]
    return allD


def Draw_ellipses(allD):
    #allD = lista de todos los parámetros de las elipses
    from matplotlib.patches import Ellipse
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_xlim(40, 450)
    ax.set_ylim(10, 180)
    ells = [Ellipse(xy=allD[1][i], width=2*allD[2][i],height=2*allD[3][i], angle=np.rad2deg(allD[4][20]), edgecolor='k', fc='None', lw=1, zorder = 2)
            for i in range(len(allD[2]))]
    for e in ells:
        ax.add_patch(e)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.show()
    #retorna la gráfica de las elipses


# In[14]:


def mu(semi_mayor, Intensidad, datos_disco):
    #semi_mayor = lista de semi-eje mayor
    #Intensidad = lista de intensidades
    #datos_disco = limite inferior para hacer el ajuste
    from scipy.optimize import curve_fit
    semi_mayor_disco = semi_mayor[datos_disco:len(semi_mayor)]
    Intensidad_disco = Intensidad[datos_disco:len(Intensidad)]

    def brightness_profile (x,muo,h):
        return muo + 1.09*(x/h)
    popt, pcov = curve_fit(brightness_profile, semi_mayor, Intensidad, [1,1])
    print(popt)

    xfit_disk = np.linspace(semi_mayor.min(), 80, 1000)
    yfit_disk = brightness_profile(xfit_disk,popt[0],popt[1])
    #Retorna una lista con los datos de la regresión y los parámetros del ajuste
    #[xfit_disk, yfit_disk, muo, h]
    return [xfit_disk, yfit_disk, popt[0], popt[1]]


# In[15]:


def Image_galaxi(promedio_centrox, promedio_centroy, Datos, minimo_cielo, promedio_PA,Intensidad,semi_mayor, semi_menor):  
    #promedio_centrox = promedio de los centros de las elipses en x
    #promedio_centroy = promedio de los centros de las elipses en y
    #Datos = datos de la galaxia
    #minimo_cielo = mínimo del cielo
    #promedio_PA = ángulo promedio de inclinación de las elipses
    #Intesidad = lista de las intensidades de las isofotas
    #semi_mayor = lista de los semi ejes mayores
    #semi_menor = lista de los semi ejes menores
    centropro = (int(promedio_centrox/0.385),int(promedio_centroy/0.385))
    imagen = np.ones(Datos.shape)*(minimo_cielo+0.00000001)
    angle = int(promedio_PA*180/np.pi)
    thickness = 2
    Intensidad2 = Intensidad[::-1]
    semi_mayor2 = semi_mayor[::-1]
    semi_menor2 = semi_menor[::-1]
    for x in range(0,len(semi_mayor2)):
        axes = (int(semi_mayor2[x]/0.385),int(semi_menor2[x]/0.385))
        color = (Intensidad2[x], Intensidad2[x],Intensidad2[x]) 
        imagen = cv2.ellipse(imagen, centropro, axes, angle, 0, 360, color ,thickness)
        posiciones = np.where(imagen == Intensidad2[x])
        posx = []
        posy = []
        for b in range(0,len(posiciones[0])):
            if posiciones[0][b]> min(posiciones[0]) and posiciones[0][b]< max(posiciones[0]):
                posx.append(posiciones[0][b])
                posy.append(posiciones[1][b])
        for f in range(min(posiciones[0])+1,max(posiciones[0])):
            posxpos = [ i for i, v in enumerate(posx) if v==f]
            for j in range(posy[posxpos[0]],posy[posxpos[-1]]):
                imagen[f][j]= Intensidad2[x]
    #retorna la imagen de la galaxia creada con las isofotas
    return imagen


def Draw_ellipses_galaxia(allD, Data, cmap = 'viridis', norm = None, Scale = [None, None]):
    #allD = información de las isofotas
    #Datos = datos a gráficar
    #cmap =  escala de colores
    #norm = normaliación de la escala
    #Scale = escala de la normalizaión
    from matplotlib.patches import Ellipse
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.set_xlim(40, 450)
    ax.set_ylim(0, 180)

    im = ax.imshow(Data, cmap=cmap, norm = norm)
    
    ells = [Ellipse(xy=allD[1][i], width=2*allD[2][i],height=2*allD[3][i], angle=np.rad2deg(allD[4][20]), edgecolor='k', fc='None', lw=1, zorder = 2)
            for i in range(len(allD[2]))]

    for e in ells:
        ax.add_patch(e)
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.show()
    #retorna la gráfica de la galaxia y las isofotas


def SDBP(xfit_disk, muo, h, Mass_to_light):
    #xfit_disk = lista de los semi ejes mayores de las isofotas
    #muo = intensidad superficial central
    #h = scale lenght
    #Mass_to_light
    def brightness_profile(x,muo,h):
            return muo + 1.09*(x/h)
    PSDE = Mass_to_light*brightness_profile(xfit_disk,muo,h)
    #retorna un arreglo con el perfil superficial de masa estelar
    return PSDE


