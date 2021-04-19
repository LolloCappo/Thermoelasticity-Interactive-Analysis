import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from scipy import ndimage

import scipy

import RoiSelect


path_base = f'C:/Users/Rodo/Dropbox/Il mio PC (LAPTOP-SA2HR7TC)/Desktop/Tesi/Dati/res/'

lista_file = RoiSelect.list_all_files(path_base,ext = '.npy')

def rapporto(mappa,limiti_base,limiti_picco):
    # limiti = (yi,yf,xi,xf)
    media_base  = np.mean(mappa[limiti_base[0]:limiti_base[1],limiti_base[2]:limiti_base[3]])
    media_picco = np.mean(mappa[limiti_picco[0]:limiti_picco[1],limiti_picco[2]:limiti_picco[3]])
    return media_picco/media_base


def seleziona_cordinate_puntuale(mappa,size_base = 9,size_picco=9):
    flag = 'n'
    while flag == 'n':
        fig,ax = plt.subplots()
        (t_lim_inf,t_lim_sup)=RoiSelect.set_cmap(mappa)
        ax.imshow(mappa,cmap='magma',clim = [t_lim_inf,t_lim_sup])
        cordinata_base = RoiSelect.selectROI_point(fig,ax,titolo='Seleziona punto base')
        # cordinate = (x,y)
        fig,ax = plt.subplots()
        ax.imshow(mappa,cmap='magma',clim = [t_lim_inf,t_lim_sup])
        limiti_base = np.array([cordinata_base[1]-window_size_base//2,cordinata_base[1]+window_size_base//2,cordinata_base[0]-window_size_base//2,cordinata_base[0]+window_size_base//2], dtype = int)

        rect_1 = Rectangle((limiti_base[2],limiti_base[0]),limiti_base[3]-limiti_base[2],limiti_base[1]-limiti_base[0],linewidth=1,edgecolor='r',facecolor='none') # ricorda verso immagine e verso matrice
        ax.add_patch(rect_1)
        cordinata_picco = RoiSelect.selectROI_point(fig,ax,titolo='Seleziona punto picco')
        limiti_picco = np.array([cordinata_picco[1]-window_size_picco//2,cordinata_picco[1]+window_size_picco//2,cordinata_picco[0]-window_size_picco//2,cordinata_picco[0]+window_size_picco//2],dtype=int)

        plottaggio(mappa,limiti_base,limiti_picco)
        while (flag:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
    return limiti_base,limiti_picco

def seleziona_cordinate_rettangolo(mappa):
    flag = 'n'
    while flag == 'n':
        cordinata_base = RoiSelect.selectROI(mappa,titolo='Seleziona punto base')
        # cordinate = (xi,yi,xf,yf)
        limiti_base = np.array([cordinata_base[1],cordinata_base[3],cordinata_base[0],cordinata_base[2]], dtype = int)
        # limiti = (yi,yf,xi,xf)
        cordinata_picco = RoiSelect.selectROI(mappa,titolo='Seleziona punto base')
        limiti_picco = np.array([cordinata_picco[1],cordinata_picco[3],cordinata_picco[0],cordinata_picco[2]], dtype = int)
        print(limiti_picco)
        print(limiti_base)
        plottaggio(mappa,limiti_base,limiti_picco)
        while (flag:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
    return limiti_base,limiti_picco

def plottaggio(mappa,limiti_base,limiti_picco):
    _,ax = plt.subplots()
    (t_lim_inf,t_lim_sup)=RoiSelect.set_cmap(mappa)
    ax.imshow(mappa,cmap='magma',clim = [t_lim_inf,t_lim_sup])
    rect_1 = Rectangle((limiti_base[2],limiti_base[0]),limiti_base[3]-limiti_base[2],limiti_base[1]-limiti_base[0],linewidth=1,edgecolor='r',facecolor='none') # ricorda verso immagine e verso matrice
    rect_2 = Rectangle((limiti_picco[2],limiti_picco[0]),limiti_picco[3]-limiti_picco[2],limiti_picco[1]-limiti_picco[0],linewidth=1,edgecolor='r',facecolor='none') # ricorda verso immagine e verso matrice
    ax.add_patch(rect_1)
    ax.add_patch(rect_2)
    plt.show()
#print(lista_file[0][-21:-4])
#print(lista_file[0][-10])
#print(lista_file[0][-5])
#print(lista_file[0][-8:-6])

data = []
res = 0
window_size_base = 30
window_size_picco = 9
footprint = ndimage.generate_binary_structure(2, 1)
footprint = np.matrix([[1,1,1],[1,2,1],[1,1,1]])
print(footprint)
flag_filtraggio = False
for mappa_analisi in lista_file:
    mappa = np.load(mappa_analisi)[0,:,:]
    if flag_filtraggio:
        mappa = ndimage.gaussian_filter(mappa,sigma=3)
        mappa = ndimage.grey_erosion(mappa,footprint=footprint)
        mappa = ndimage.grey_dilation(mappa,footprint=footprint)
        #mappa = ndimage.gaussian_filter(mappa,sigma=3)
    flag = 'n'
    if mappa_analisi != lista_file[0]:
        plottaggio(mappa,limiti_base,limiti_picco)
        while (flag:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
    if flag == 'n':
        (limiti_base,limiti_picco) = seleziona_cordinate_rettangolo(mappa) #seleziona_cordinate_punutale(mappa,size_base=window_size_base,size_picco =window_size_picco)
        #(limiti_base,limiti_picco) = seleziona_cordinate_puntuale(mappa,size_base=window_size_base,size_picco =window_size_picco) #seleziona_cordinate_punutale(mappa,size_base=window_size_base,size_picco =window_size_picco)

    res = rapporto(mappa,limiti_base,limiti_picco)
    data.append({'strato':mappa_analisi[-10],'force':mappa_analisi[-5],'fr':int(mappa_analisi[-8:-6]),'cordinata_base':limiti_base,'cordinata_picco':limiti_picco,'rapporto':res})

#import json
#file_res = open(path_base+'res_modulo_complessivo.json',"w")
#for i in data:
#    res_fr = json.dumps(i)
#    file_res.write(res_fr)
#    file_res.write("\n")
#file_res.close()
import pandas as pd
data = pd.DataFrame(data)

_,ax_0 = plt.subplots()
ax_0.set(title= 'Mappa Modulo',xlabel='f [Hz]')
data[data['force']=='L'].plot(x = 'fr',y = 'rapporto',c='r',ax =ax_0,label='L')
data[data['force']=='M'].plot(x = 'fr',y = 'rapporto',c='g',ax =ax_0,label='M')
data[data['force']=='H'].plot(x = 'fr',y = 'rapporto',ax = ax_0,label='H')

data.to_csv(path_base+'data.csv',index = False, header=True)
plt.show()
print(data)
