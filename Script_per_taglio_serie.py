import pysfmov
import matplotlib.pyplot as plt
import numpy as np
import analisi_tsa # modulo
import json
import time

import RoiSelect

# dati
fa = 120 # freq ac1uisiszione
fr = [40,50] # 10,20,30,
force = ['L','M','H']
label_nome = "strato_0"
# Inizio il vettore Nomi
names = []
path = f'C:/Users/Rodo/Dropbox/Il mio PC (LAPTOP-SA2HR7TC)/Desktop/Tesi/Dati/'
for f in fr:
    for l in force:
        names.append({'name':f"cfrp_{label_nome}_{fa}Hz_f{f}_{l}",'fa':fa,'fr':f,'force':l,'path':path+f"{label_nome}/cfrp_{label_nome}_{fa}Hz_f{f}_{l}"})
#
flag_auto = 'n' # da selezionare?
while (flag_auto:= input("Ritaglio automatico del video? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass

flag = 'y'
if flag == 'y':
    Analisi = analisi_tsa.TSA(fa,f"{names[0]['path']}.sfmov")
    cordinate_finestra = RoiSelect.selectROI(Analisi.finestra_video[:,:,0],titolo='Seleziona finestra') # (xi,yi,dx,dy)
    for name in names:
        if name != names[0]:
            Analisi = analisi_tsa.TSA(fa,name['path']+".sfmov")
        if flag_auto == 'n':
            _,ax = plt.subplots()
            ax.imshow(Analisi.finestra_video[cordinate_finestra[1]:cordinate_finestra[3],cordinate_finestra[0]:cordinate_finestra[2],0])
            ax.set(title=f"Confermare la Finestra a {name['fr']} [Hz] e l = {name['force']}")
            plt.show()
            while (flag_utente:= input(f"Va bene il taglio f = {name['fr']} e L = {name['force']}? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
            if flag_utente == 'n':
                cordinate_finestra = RoiSelect.selectROI(Analisi.finestra_video_roi_offset,titolo='Seleziona finestra')
        analisi_tsa.taglio_video(Analisi,cordinate_finestra[1],cordinate_finestra[0],cordinate_finestra[3],cordinate_finestra[2],save_path=path+name['name']+'.npy')



