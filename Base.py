import matplotlib.pyplot as plt
import numpy as np
import analisi_tsa # modulo

import RoiSelect


fa = 120 # freq ac1uisiszione
fr = 30
label_nome = 'strato_0'
livello_carico = 'H'
path_base = f'C:/Users/Rodo/Dropbox/Il mio PC (LAPTOP-SA2HR7TC)/Desktop/Tesi/Dati/{label_nome}/'

path = path_base + f'cfrp_{label_nome}_{fa}Hz_f{fr}_{livello_carico}.sfmov'
path_npy = path_base + f'cfrp_{label_nome}_{fa}Hz_f{fr}_{livello_carico}.npy' 


flag = False # da ritagliare .npy? o video interno
if flag:
    Analisi = analisi_tsa.TSA(fa,path)
else:
    Analisi = analisi_tsa.TSA(fa,path_npy)

cordinate = RoiSelect.selectROI(Analisi.finestra_video[:,:,1],titolo='Seleziona ROI')
Analisi.set_ROI(cordinate[1],cordinate[0],cordinate[3],cordinate[2],view = False)
#cordinate = RoiSelect.selectROI_ellipse(Analisi.finestra_video_roi_offset,titolo='Seleziona ROI foro')
#Analisi.set_hole(cordinate[1],cordinate[0],cordinate[3],cordinate[2])
flag_utente = 'n'
while flag_utente == 'n':
    cordinate = RoiSelect.selectROI(Analisi.finestra_video_roi_offset,titolo='Seleziona finestra di compensazione')
    Analisi.freq_detection(fr,cordinate[1],cordinate[0],cordinate[3],cordinate[2],df = 5,view=True)
    while (flag_utente:= input(f"Va bene la zona compensazione? (Enter y/n) : ... ").lower()) not in {"y", "n"}: pass
print(cordinate)



#map_std = Analisi.view_std()
#RoiSelect.interactive_cmap(map_std)

#Analisi.view()
#Analisi.view_animate()

(mappa_modulo,mappa_fase)=Analisi.lockin(view = True,t_lim_inf = None, t_lim_sup = None)

analisi_tsa.intercative_phase(mappa_fase)


Analisi.view_result_coutour_plot()

#cordinate = RoiSelect.select_line(mappa_modulo,titolo='Seleziona res')
#Analisi.view_result_line(cordinate[1],cordinate[0],cordinate[3],cordinate[2])
plt.show()
RoiSelect.interactive_cmap(np.abs(mappa_modulo))


