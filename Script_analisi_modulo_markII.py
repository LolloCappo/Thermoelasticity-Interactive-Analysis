import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy
from scipy import interpolate,ndimage

import RoiSelect

def seleziona_cordinate_mista(mappa,window_size_picco=9):
    flag = 'n'
    while flag == 'n':
        cordinata_base = RoiSelect.selectROI(mappa,titolo='Seleziona base')       # cordinate = (xi,yi,xf,yf)
        limiti_base = np.array([cordinata_base[1],cordinata_base[3],cordinata_base[0],cordinata_base[2]], dtype = int)   # limiti = (yi,yf,xi,xf)
        (dx,dy) = mappa.shape
        fig,ax = plt.subplots(figsize=(15*dy/(dx+dy),15*dx/(dx+dy)))
        ax = coutour_plot(mappa,ax)
        cordinata_picco = RoiSelect.selectROI_point(fig,ax,titolo='Seleziona punto picco')
        limiti_picco = np.array([cordinata_picco[1]-window_size_picco//2,cordinata_picco[1]+window_size_picco//2,cordinata_picco[0]-window_size_picco//2,cordinata_picco[0]+window_size_picco//2],dtype=int)
        while (flag:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
    return limiti_base,limiti_picco

def seleziona_cordinate_rettangolo(mappa):
    flag = 'n'
    while flag == 'n':
        cordinata_base = RoiSelect.selectROI(mappa,titolo='Seleziona punto base',contour_plot_flag=True)
        # cordinate = (xi,yi,xf,yf)
        limiti_base = np.array([cordinata_base[1],cordinata_base[3],cordinata_base[0],cordinata_base[2]], dtype = int,)
        # limiti = (yi,yf,xi,xf)
        cordinata_picco = RoiSelect.selectROI(mappa,titolo='Seleziona punto picco',contour_plot_flag=True)
        limiti_picco = np.array([cordinata_picco[1],cordinata_picco[3],cordinata_picco[0],cordinata_picco[2]], dtype = int)
        mappa_view = mappa.copy()
        max_view = np.max(mappa_view)
        mappa_view[limiti_base[0]:limiti_base[1],limiti_base[2]:limiti_base[3]] = 1.5*max_view
        mappa_view[limiti_picco[0]:limiti_picco[1],limiti_picco[2]:limiti_picco[3]] = 1.5*max_view
        plt.imshow(mappa_view)

        while (flag:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
    return limiti_base,limiti_picco

def selezione_blob(mappa,flag_limiti,limiti_base=[0,0,0,0],limiti_picco=[0,0,0,0],fattore_scala=0.8):
    mappa_view = mappa.copy()
    flag_while = 'n'
    while flag_while == 'n':
        if flag_limiti:
            limiti_base,limiti_picco = seleziona_cordinate_rettangolo(mappa_view)
        media_base  = np.mean(mappa[limiti_base[0]:limiti_base[1],limiti_base[2]:limiti_base[3]])
        base_std = np.std(mappa[limiti_base[0]:limiti_base[1],limiti_base[2]:limiti_base[3]])
        ### selezione picco
        # taglio la mappa
        mappa  = mappa[limiti_picco[0]:limiti_picco[1],limiti_picco[2]:limiti_picco[3]]
        (dx,dy) = mappa.shape
        N = dx*dy
        # filtraggio
        mappa = ndimage.gaussian_filter(mappa,sigma=1)
        footprint = np.matrix([[1,1,1],[1,2,1],[1,1,1]])
        mappa = ndimage.grey_erosion(mappa,footprint=footprint)
        mappa = ndimage.grey_dilation(mappa,footprint=footprint)
        # analisi e selezione blob
        istogramma = np.sort(np.reshape(mappa,N))
        mask = np.empty((dx,dy))
        flag_mask = 'n'
        while flag_mask =='n': 
            #soglia = istogramma[int(fattore_scala*N)]
            soglia = media_base + fattore_scala*base_std
            mask = mappa > soglia
            labels, _ = ndimage.label(mask)
            fig,ax = plt.subplots(1,2)        
            plt.gcf().text(0.5,0.001, f' blob isolivello per {fattore_scala*100:.{2}f} [%] ', fontsize=14)
            coutour_plot(mappa,ax[0],levels = [soglia],filter_flag=False)
            ax[1].imshow(labels)
            cordinata_picco = RoiSelect.selectROI_point(fig,ax[1],titolo='Seleziona punto picco')
            try :
                media_picco = mappa[labels != labels[cordinata_picco[1],cordinata_picco[0]]].mean()
            except:
                media_picco = np.mean(mappa)
            #while (flag_mask:= input(f"Vanno bene i livelli? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
            flag_mask = 'y'

            if flag_mask == 'n':
                scale_livello = float(input("percentile del livello: ... "))/100
        flag_while = 'y'
        #while (flag_while:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
    return media_base,media_picco
    
def coutour_plot(immagine,ax,levels = [],filter_flag=True):
    mappa = immagine.copy()
    (dx,dy) = mappa.shape
    N = dx*dy
    istogramma = np.zeros(N)
    istogramma = np.sort(np.reshape(mappa,N))
    if not levels:
        levels = [istogramma[int(0.2*N)],istogramma[int(0.6*N)],istogramma[int(0.8*N)],istogramma[int(0.9*N)],istogramma[int(0.95*N)]]
    footprint = np.matrix([[1,1,1],[1,2,1],[1,1,1]])
    if filter_flag:
        mappa = ndimage.gaussian_filter(mappa,sigma=2)
        mappa = ndimage.grey_erosion(mappa,footprint=footprint)
        mappa = ndimage.grey_dilation(mappa,footprint=footprint)
    ax.imshow(mappa,cmap = 'inferno',alpha=0.5)
    CS = ax.contour(np.arange(dy),np.arange(dx),mappa,levels,cmap = 'inferno')
    ax.clabel(CS, inline=True, fontsize=10)
    return ax

path_base = f'C:/Users/Rodo/Dropbox/Il mio PC (LAPTOP-SA2HR7TC)/Desktop/Tesi/Dati/res/'
lista_file = RoiSelect.list_all_files(path_base,ext = '.npy')
lista_file.reverse()
N_prove = ['dx','dy']
flag_visualizzazione = False
#

# inizializzazione salvo i dati
data_mean = [] 
data_max = []
data_cv = []
limiti_base = np.empty(4)
limiti_picco = np.empty(4)
#
for prova in N_prove:
    flag_selezione_roi = True
    #limiti_base = np.array([317,474,16,172])
    #limiti_picco = np.array([262,280,126,141])
    for file_corrente in lista_file:
        print(file_corrente[-21:-4])
        mappa = np.load(file_corrente)[0,:,:]
        if flag_selezione_roi:
            (limiti_base,limiti_picco) = seleziona_cordinate_mista(mappa)
             #(limiti_base,limiti_picco) = seleziona_cordinate_rettangolo(mappa)
        flag_selezione_roi = False # prendo solo la prima
        if True:
            mappa = ndimage.gaussian_filter(mappa,sigma=10)
            footprint = np.matrix([[1,1,1],[1,2,1],[1,1,1]])
            mappa = ndimage.grey_dilation(mappa,footprint=footprint)
            mappa = ndimage.grey_erosion(mappa,footprint=footprint)
        plt.imshow(mappa)
        plt.show()
        media_base  = np.mean(mappa[limiti_base[0]:limiti_base[1],limiti_base[2]:limiti_base[3]])
        std_base = np.std(mappa[limiti_base[0]:limiti_base[1],limiti_base[2]:limiti_base[3]])
        media_picco = np.mean(mappa[limiti_picco[0]:limiti_picco[1],limiti_picco[2]:limiti_picco[3]])
        std_picco = np.std(mappa[limiti_picco[0]:limiti_picco[1],limiti_picco[2]:limiti_picco[3]])
        _,ax = plt.subplots()
        ax.imshow(mappa[limiti_picco[0]:limiti_picco[1],limiti_picco[2]:limiti_picco[3]],cmap='inferno')
        plt.show()
        max_picco = np.percentile(mappa[limiti_picco[0]:limiti_picco[1],limiti_picco[2]:limiti_picco[3]],90)
    
        data_mean.append({'strato':file_corrente[-10],'force':file_corrente[-5],'fr':int(file_corrente[-8:-6]),'cordinata_base':limiti_base,'cordinata_picco':limiti_picco,'rapporto':media_picco/media_base,'media_base':media_base,'std_base':std_base,'media_picco':media_picco,'std_picco':std_picco,'lato':prova})
        data_max.append({'strato':file_corrente[-10],'force':file_corrente[-5],'fr':int(file_corrente[-8:-6]),'cordinata_base':limiti_base,'cordinata_picco':limiti_picco,'rapporto':max_picco/media_base,'media_base':media_base,'std_base':std_base,'max_picco':max_picco,'std_picco':std_picco,'lato':prova})
        media_base,cv_picco = selezione_blob(mappa,flag_selezione_roi,limiti_base=limiti_base,limiti_picco=limiti_picco)
        #cv_picco = 0
        data_cv.append({'strato':file_corrente[-10],'force':file_corrente[-5],'fr':int(file_corrente[-8:-6]),'rapporto':cv_picco/media_base,'std_base':std_base,'media_base':media_base,'cv_picco':cv_picco,'lato':prova})
    print('cambio lato')
    #limiti_base = np.array([303,475,24,169])
    #limiti_picco = [260,276,61,74]
# esporto i dati
data_mean = pd.DataFrame(data_mean)
data_max = pd.DataFrame(data_max)
data_cv = pd.DataFrame(data_cv)
data_mean.to_csv(path_base+f'data_mean_{file_corrente[-10]}.csv',index = False, header=True)
data_max.to_csv(path_base+f'data_max_{file_corrente[-10]}.csv',index = False, header=True)
data_cv.to_csv(path_base+f'data_cv_{file_corrente[-10]}.csv',index = False, header=True)

_,ax_0 = plt.subplots()
f = [10,20,30,40,50]
ff = np.arange(f[0],f[-1],0.2)

ax_0.set(title= 'Mappa Modulo',xlabel='f [Hz]')
k = 0
colori = [{'name':'data mean','L':'#A8F017','M':'#2211BA','H':'#E41414'},{'name':'data max','L':'#3F871B','M':'#493F9E','H':'#AF2020'},{'name':'data cv','L':'#87D77B','M':'#847BD7','H':'##914C4C'}]
nomi = [' data_mean',' data_max',' data_cv']
for data in [data_mean,data_max]:#,data_cv]:
    nome = nomi[k]
    colore = colori[k]
    for prova in N_prove:
        for livello in ['L','M','H']:
            data[data['force']==livello].plot.scatter(x = 'fr',y = 'rapporto',c=colore[livello],ax =ax_0,label=livello+' '+colore['name']+nome)
            f_temp = np.array(data[data['force']==livello].groupby('fr')['rapporto'].mean()).flatten()
            ax_0.plot(ff,interpolate.BarycentricInterpolator(f,f_temp)(ff),c=colore[livello])
    k =+ 1
plt.show()




