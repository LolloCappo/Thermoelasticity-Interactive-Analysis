import pysfmov
import matplotlib.pyplot as plt
import numpy as np
import analisi_tsa # modulo
import json
import time

import RoiSelect

fa = 100 # freq ac1uisiszione
fr = [5,10,15,20,25,30,35,40,45,50]
path = f'C:/Users/Rodo/Dropbox/Il mio PC (LAPTOP-SA2HR7TC)/Desktop/Tesi/Dati/'
flag = False # da selezionare?
flag_utente = 'y'

if flag:
    Analisi = analisi_tsa.TSA(fa,path+f'cfrp{fr[0]}hz.sfmov')
    _,ax = plt.subplots()
    ax.imshow(Analisi.finestra_video[:,:,0])
    ax.set(title='Seleziona finestra')
    cordinate_finestra = RoiSelect.selectROI(ax) # (xi,yi,dx,dy)
    for f in fr:
        Analisi = analisi_tsa.TSA(fa,path+f'cfrp{fr[0]}hz.sfmov')
        _,ax = plt.subplots()
        ax.imshow(Analisi.finestra_video[cordinate_finestra[1]:cordinate_finestra[3],cordinate_finestra[0]:cordinate_finestra[2],0])
        ax.set(title=f'Confermare la Finestra a {f} [Hz]')
        plt.show()
        while (flag_utente:= input("Va bene la ROI? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
        if flag_utente == 'n':
            _,ax = plt.subplots()
            ax.imshow(Analisi.finestra_video[:,:,0])
            cordinate_finestra = RoiSelect.selectROI(ax)
            Analisi.taglio_video(cordinate_finestra[1],cordinate_finestra[0],cordinate_finestra[3],cordinate_finestra[2],view = False,save=True,save_path=path+f'cpfr{f}.npy')
        else:
            Analisi.taglio_video(cordinate_finestra[1],cordinate_finestra[0],cordinate_finestra[3],cordinate_finestra[2],view = False,save=True,save_path=path+f'cpfr{f}.npy')

Analisi = analisi_tsa.TSA(fa,path+f'cpfr{fr[0]}.npy')
_,ax = plt.subplots()
ax.imshow(Analisi.finestra_video[:,:,0])
ax.set(title='Seleziona ROI')
cordinate_roi = RoiSelect.selectROI(ax)
Analisi.selezione_ROI(cordinate_roi[1],cordinate_roi[0],cordinate_roi[3],cordinate_roi[2],view = False)
_,ax = plt.subplots()
ax.imshow(Analisi.finestra_video_roi[:,:,0]+Analisi.finestra_video_roi_offset)
ax.set(title='Seleziona finestra di compensazione')
cordinate_cop = RoiSelect.selectROI(ax)

file_res = open(path+'res.json',"w")
for f in fr:
    Analisi = analisi_tsa.TSA(fa,path+f'cpfr{f}.npy')
    Analisi.selezione_ROI(cordinate_roi[1],cordinate_roi[0],cordinate_roi[3],cordinate_roi[2],view = True)
    while (flag_utente:= input("Va bene la ROI? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
    if flag_utente == 'n':
        _,ax = plt.subplots()
        ax.imshow(Analisi.finestra_video[:,:,0])   
        cordinate = RoiSelect.selectROI(ax)
        Analisi.selezione_ROI(cordinate[1],cordinate[0],cordinate[3],cordinate[2],view = True)
        _,ax = plt.subplots()
        ax.imshow(Analisi.finestra_video_roi[:,:,0]+Analisi.finestra_video_roi_media)
        ax.set(title='Seleziona finestra di compensazione')
        cordinate = RoiSelect.selectROI(ax)
        (fr_compensata,DTmedia,fase) = Analisi.selezione_ROI_compensazione(cordinate[1],cordinate[0],cordinate[3],cordinate[2],f,df = 5,view = False)
    else:
        (DTmedia,fr_compensata) = Analisi.selezione_ROI_compensazione(cordinate_cop[1],cordinate_cop[0],cordinate_cop[3],cordinate_cop[2],f,df = 5,view = False)
    [mappa_modulo,mappa_fase]=Analisi.lockin(view = True)
    temp = {'Ora analisi':time.ctime(),'fa[Hz]':fa,'fr[Hz]':f,'fr_lockin[Hz]':fr_compensata,'DTmediArea[K]':DTmedia}
    np.save(path+f'res{f}Hz.npy',np.array([mappa_modulo,mappa_fase]))
    print(temp)
    res_fr = json.dumps(temp)
    file_res.write(res_fr)
    file_res.write("\n")
file_res.close()
