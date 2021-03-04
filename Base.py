import matplotlib.pyplot as plt
import numpy as np
import analisi_tsa # modulo

import RoiSelect


fa = 100 # freq ac1uisiszione
fr = 35
path_base = r'C:/Users/Rodo/Dropbox/Il mio PC (LAPTOP-SA2HR7TC)/Desktop/Tesi/Dati/'
path = path_base + f'cfrp{fr}hz.sfmov'
path_npy = path_base + f'cpfr{fr}.npy'

flag = False # da ritagliare .npy? o video interno
if flag:
    Analisi = analisi_tsa.TSA(fa,path)
    cordinate = RoiSelect.selectROI(Analisi.finestra_video[:,:,1],titolo='Seleziona Area Analisi')
    Analisi.taglio_video(cordinate[1],cordinate[0],cordinate[3],cordinate[2],save=True,view = True,save_path=path_npy)
else:
    Analisi = analisi_tsa.TSA(fa,path_npy)

cordinate = RoiSelect.selectROI(Analisi.finestra_video[:,:,1],titolo='Seleziona ROI')
Analisi.selezione_ROI(cordinate[1],cordinate[0],cordinate[3],cordinate[2],view = True)


#cordinate = RoiSelect.selectROI_ellipse(Analisi.finestra_video_roi_offset,titolo='Seleziona ROI foro')
#Analisi.selezione_ROI_foro(cordinate[1],cordinate[0],cordinate[3],cordinate[2])

cordinate = RoiSelect.selectROI(Analisi.finestra_video_roi_offset,titolo='Seleziona finestra di compensazione')
print(cordinate)
Analisi.selezione_ROI_compensazione(fr,cordinate[1],cordinate[0],cordinate[3],cordinate[2],df = 3,view=True)


#Analisi.view_std()

#Analisi.view()
#Analisi.view_animate()

(mappa_modulo,mappa_fase)=Analisi.lockin(view = True,t_lim_inf = None, t_lim_sup = None)
#cordinate = RoiSelect.select_line(mappa_modulo,titolo='Seleziona res')
#Analisi.view_result_line(cordinate[1],cordinate[0],cordinate[3],cordinate[2])


from matplotlib.widgets import Slider

def update(val):
    lim_inf = cmap_lim_inf.val
    lim_sup = cmap_lim_sup.val
    if lim_inf<lim_sup:
        l.set_clim([lim_inf,lim_sup])
    fig.canvas.draw_idle()



fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
lim_inf = 1.1*np.min(mappa_fase)
lim_sup = 0.9*np.max(mappa_fase)
l = plt.imshow(np.abs(mappa_fase),cmap='magma')
fig.colorbar(l,orientation = 'vertical',fraction = 0.5)
ax.margins(x=0)

axcolor = 'lightgoldenrodyellow'
ax_lim_inf = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
ax_lim_sup = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

cmap_lim_inf = Slider(ax_lim_inf, 'sunf', 0, np.max(mappa_fase), valinit=lim_inf)
cmap_lim_sup = Slider(ax_lim_sup, 'sup', 0, np.max(mappa_fase), valinit=lim_sup)
cmap_lim_inf.on_changed(update)
cmap_lim_sup.on_changed(update)

plt.show()

def applica_filtro(image,kernel,name=''): # non decomposto (ignorante)
    print('Esecuzione filtro',name)
    (R,C) = image.shape
    (w,_) = kernel.shape
    w = w//2
    image_f = np.zeros((R,C))
    for r in range(w,R-w):
        for c in range(w,C-w): # scorro punto
            for i in range(2*w+1): # scorro indice
                for j in range(2*w+1):
                    image_f[r,c] += image[r-i,c-j]*kernel[i,j]
    return image_f

h = 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]]) # Gaussiano
Analisi.map_phase = applica_filtro(Analisi.map_phase,h)
Analisi.map_amplitude = applica_filtro(Analisi.map_phase,h)

Analisi.view_result()

