import matplotlib.pyplot as plt
import numpy as np
import pytsa # modulo

from scipy import interpolate,ndimage

import RoiSelect


fa = 120 # freq q
fr = 30
label_nome = 'strato_1'
livello_carico = 'H'
path_base = f'C:/Users/Rodo/Dropbox/Il mio PC (LAPTOP-SA2HR7TC)/Desktop/Tesi/Dati/{label_nome}/'

path = path_base + f'cfrp_{label_nome}_{fa}Hz_f{fr}_{livello_carico}.sfmov'
path_npy = path_base + f'cfrp_{label_nome}_{fa}Hz_f{fr}_{livello_carico}.npy' 


flag = False # da ritagliare .npy? o video interno
if flag:
    Analisi = pytsa.TSA(fa,path)
else:
    Analisi = pytsa.TSA(fa,path_npy)
cordinate = RoiSelect.selectROI(Analisi.get_roi_offset(),titolo='split video?')
pytsa.split_video(Analisi,cordinate[1],cordinate[0],cordinate[3],cordinate[2],50,250,save_path='video')
path_npy = f'video.npy'
Analisi = pytsa.TSA(fa,path_npy)
Analisi.view()
Analisi.view_animate()

cordinate = RoiSelect.selectROI(Analisi.get_roi_offset(),titolo='Seleziona ROI')
Analisi.set_ROI(cordinate[1],cordinate[0],cordinate[3],cordinate[2],view = False)

cordinate = RoiSelect.selectROI_ellipse(Analisi.get_roi_offset(),titolo='Seleziona ROI foro')
Analisi.set_hole(cordinate[1],cordinate[0],cordinate[3],cordinate[2])

flag_utente = 'n'
while flag_utente == 'n':
    cordinate = RoiSelect.selectROI(Analisi.get_roi_offset(),titolo='Seleziona finestra di compensazione')
    Analisi.freq_detection(fr,cordinate[1],cordinate[0],cordinate[3],cordinate[2],df = 5,view=True)
    while (flag_utente:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
fr = Analisi.get_freq()
print(cordinate)


(mappa_modulo,mappa_fase)=Analisi.lockin()
Analisi.view_result()

flag_utente = 'n'
while flag_utente == 'n':
    mappa_view = mappa_modulo
    cordinate = RoiSelect.select_line(mappa_view,titolo='ROI')
    Analisi.result_line(cordinate[3],cordinate[2],cordinate[1],cordinate[0],view = True)
    while (flag_utente:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
Analisi.set_phase_offset()
Analisi.view_result(phase_reverse=True)
Analisi.set_cmap_lim(interactive=True)
Analisi.view_result()
#Analisi.save()

Analisi.view_result_coutour_plot()








