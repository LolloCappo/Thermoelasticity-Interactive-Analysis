import matplotlib.pyplot as plt
import numpy as np
import json
import time
import sys
sys.path.insert(0,'..')
import RoiSelect
import pytsa # modulo

# dati
fa = 120 # freq ac1uisiszione
fr = [10,20,30,40,50]
force = ['L','M','H']


label_nome = "strato_1"
# Inizio il vettore Nomi
names = []
path = f'C:/Users/Rodo/Dropbox/Il mio PC (LAPTOP-SA2HR7TC)/Desktop/Tesi/Dati/'
flag_utente = 'npy'
while (flag_utente:= input("Caricare il .sfmov o in .npy il video? (Enter sfomv/npy) : ... ").lower()) not in {"sfmov",".sfmov",".npy","npy"}: pass
for f in fr:
    for l in force:
        if flag_utente == 'npy' or flag_utente == '.npy':
            print('file .npy')
            names.append({'name':f"cfrp_{label_nome}_{fa}Hz_f{f}_{l}",'fa':fa,'fr':f,'force':l,'path':path+f'{label_nome}/cfrp_{label_nome}_{fa}Hz_f{f}_{l}.npy'})
        else:
            print('file .sfmov')
            names.append({'name':f"cfrp_{label_nome}_{fa}Hz_f{f}_{l}",'fa':fa,'fr':f,'force':l,'path':path+f'{label_nome}/cfrp_{label_nome}_{fa}Hz_f{f}_{l}.sfmov'})
for name in names:
    print(name['name'])
print(len(names))#
path_res = f"{path}/res/"
Analisi = pytsa.TSA(fa,names[0]['path'])

cordinate_roi = RoiSelect.selectROI(Analisi.get_roi_offset(),titolo ='Seleziona ROI')
Analisi.set_ROI(cordinate_roi[1],cordinate_roi[0],cordinate_roi[3],cordinate_roi[2],view = False)
cordinate_crop = RoiSelect.selectROI(Analisi.get_roi_offset(),titolo = 'Seleziona finestra di compensazione')
dict_res = []
for name in names:
    print(name)
    theta_offset = 0
    flag_analisi = 'y'
    if name != names[0]:
        Analisi = pytsa.TSA(name['fa'],name['path'])
        Analisi.set_ROI(cordinate_roi[1],cordinate_roi[0],cordinate_roi[3],cordinate_roi[2],view = False)
    while flag_analisi == 'y':
        (fr_real,_,_) = Analisi.freq_detection(name['fr'],cordinate_crop[1],cordinate_crop[0],cordinate_crop[3],cordinate_crop[2],df = 5,view = True)
        while (flag_utente:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
        while flag_utente == 'n':
            cordinate = RoiSelect.selectROI(Analisi.get_roi_offset(),titolo='Seleziona finestra di compensazione')
            (fr_real,_,_) = Analisi.freq_detection(name['fr'],cordinate_crop[1],cordinate_crop[0],cordinate_crop[3],cordinate_crop[2],df = 4,view = True)
            while (flag_utente:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass

        [mappa_modulo,mappa_fase]=Analisi.lockin(view = True)
        t_lim_inf = None
        t_lim_sup = None
        while (flag_utente:= input(f"Va bene la zona cmap? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
        while flag_utente == 'n':
            (t_lim_inf,t_lim_sup) = Analisi.set_cmap_lim(interactive=True)
            while (flag_utente:= input(f"Va bene la zona cmap? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
        Analisi.view_result(t_lim_inf=t_lim_inf,t_lim_sup = t_lim_sup, save = True,path=path_res+'cfrp_force_'+name['force']+'_')
        while (flag_analisi:= input(f"Ripetere l'analisi? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
    dict_res.append({'Ora analisi':time.ctime(),'fa[Hz]':name['fa'],'fr[Hz]':name['fr'],'fr_real':fr_real,'carico':name['force'],'name':name['name'],'modulo_limiti_cmap':(t_lim_inf,t_lim_sup),'fase offset':theta_offset})    
    np.save(path_res + f"res_{label_nome}_{name['fr']}_" + name['force'] +'.npy',np.array([mappa_modulo,mappa_fase]))

file_res = open(path_res+'res.json',"w")
for i in dict_res:   
    file_res.write(json.dumps(i))
    file_res.write("\n")
file_res.close()


