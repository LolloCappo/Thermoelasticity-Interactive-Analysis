import matplotlib.pyplot as plt
import numpy as np
import analisi_tsa # modulo
from scipy import interpolate,ndimage

import RoiSelect


fa = 120 # freq ac1uisiszione
fr = 10
label_nome = 'strato_1'
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
fr = Analisi.get_freq()
print(cordinate)

import matplotlib.animation as animation 
from scipy import signal

#yi,xi,yf,xf = RoiSelect.select_line(Analisi.finestra_video_roi_offset,titolo='Seleziona res')
yi,xi,yf,xf = 99,220,100,270
print([yi,xi,yf,xf])
dx = (xf-xi)
dy = (yf-yi)
if dx != 0 and dy != 0:
    u = np.arange(int(dx*dy))/int(dx*dy)
elif dx != 0:
    u = np.arange(int(dx))/int(dx)
elif dy != 0:
    u = np.arange(int(dy))/int(dy)
res = []
for i in range(int(dx*dy)):
    x_v = xi + dx*u[i]
    y_v = yi + dy*u[i]
    if not([int(x_v),int(y_v)] in res):
        res.append([int(x_v),int(y_v)])
res_max = []
fig,(ax0,ax1) = plt.subplots(1,2)
video = Analisi.finestra_video_roi.copy() 
offset = Analisi.finestra_video_roi_offset.copy()
(_,_,N) = video.shape
ims = []
res_2_total = []
max_video = np.max(video)
for t in range(N):
    res_2 = []
    for i in res:
        res_2.append(Analisi.finestra_video_roi[i[0],i[1],t]+offset[i[0],i[1]])
        #video[i[0],i[1]] = max_video*1.2
    res_2_total.append(res_2)
    res_filtrato = ndimage.gaussian_filter1d(res_2, 2)
    res_filtrato = signal.convolve(res_filtrato,np.array([-1,0,1]),mode = 'valid')
    res_filtrato = abs(res_filtrato)
    soglia = 0.04
    res_filtrato[res_filtrato<soglia] = 0
    res_arg_max = np.argmax(res_filtrato)
    res_max.append(res_arg_max)
    ims0 = ax0.imshow(video[:,:,t]+ offset ,cmap = 'inferno', animated=True)   
    ims1, = ax1.plot(res_2,animated=True)
    ims.append([ims0,ims1])
np.save('mod.npy',res_2_total)
video[250:254,98:101,:] = max_video*1.2

from numba import jit

@jit(nopython=True)
def sposta_video(video,fa,fr,ampitude,phase=0):
    (x_lim,y_lim,N) = video.shape
    video_res = np.zeros((x_lim,y_lim,N))
    time = (1/fa)*np.arange(N)
    dx = ampitude*np.cos(2*np.pi*fr*time+phase)
    ampitude = abs(ampitude)
    for t in range(N):
        for y in range(y_lim):
            for x in range(int(ampitude)+1,x_lim-int(ampitude)-1):
                video_res[x,y,t] = (video[int(np.floor(x+dx[t])),y,t]*(x+dx[t])//2+video[int(np.ceil(x+dx[t])),y,t]*(x+dx[t])%2)/(2*(x+dx[t]))
    return video_res

def lockin(fr,fa,signal):
    time = (1/fa)*np.arange(len(signal))
    wr = fr*(2*np.pi)
    SgnSF = np.sin(wr*time)*signal
    SgnSG = np.cos(wr*time)*signal
    sf = np.abs(np.mean(SgnSF)*2)
    sg = np.abs(np.mean(SgnSG)*2)
    return np.sqrt(sf**2+sg**2),np.arctan2(sg,sf)
    
animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=900)
plt.show()
fig,(ax0,ax1) = plt.subplots(1,2)

ax0.plot(res_max[10:200])
f = (fa/N)*np.arange((N)//2+1)
S_fft = np.fft.rfft(res_max)/len(res_max)
print('max fft')
ampitude,phase = lockin(fr,fa,res_max)
print([ampitude,phase*180/np.pi])

ax1.plot(f[1:],2*np.abs(S_fft[1:]))
plt.show()
#
signale = Analisi.finestra_video_roi[209:320,99,1]+offset[209:320,99]
signale += - np.mean(signale)
#plt.plot(signale)
#plt.show()
signale = signal.medfilt(signale,9)
signale = signal.convolve(signale,np.array([-1,0,1]),mode = 'valid')
plt.plot(signale)

df = 0.2
n_lim = (np.array([fr-df/2,fr+df/2])*(N/fa)).astype(int)
std_res = []
for i in range(209,320):
    signale = np.array(Analisi.finestra_video_roi[i,99,:])
    std_res.append(np.std(signale))
np.save('mod_std_res.npy',std_res)
plt.plot(std_res)
plt.show()
signale = Analisi.finestra_video_roi[254,102,:]
f = (fa/N)*np.arange((N)//2+1)
S_fft = 2*np.abs(np.fft.rfft(signale)/N)
fig,(ax0,ax1) = plt.subplots(1,2)   
ax0.plot(signale)
ax1.plot(f,S_fft)
plt.show()

for t in range(N):
    video[:,:,t] += offset 
#video = sposta_video(video,fa,fr=10.395,ampitude = 0.9,phase = -(48.71)/180*np.pi+np.pi)
video = sposta_video(video,fa,fr=fr,ampitude = ampitude,phase = phase)

signale = video[254,102,:]

f = (fa/N)*np.arange((N)//2+1)
S_fft = 2*np.abs(np.fft.rfft(signale)/N)
fig,(ax0,ax1) = plt.subplots(1,2)
ax0.plot(signale)
ax1.plot(f[10:],S_fft[10:])
plt.show()
Analisi = analisi_tsa.TSA(fa,video_npy=video,flag_npy=True)

cordinate = RoiSelect.selectROI(Analisi.finestra_video[:,:,1],titolo='Seleziona ROI')
Analisi.set_ROI(cordinate[1],cordinate[0],cordinate[3],cordinate[2],view = False)
Analisi.view_animate()
Analisi.lockin(fr=fr,view = True)
#map_std = Analisi.view_std()
#RoiSelect.interactive_cmap(map_std)
print(ok)

#Analisi.view()
#Analisi.view_animate()
(mappa_modulo,mappa_fase)=Analisi.lockin(view = True,t_lim_inf = None, t_lim_sup = None)
Analisi.result_line(xi, yi, xf, yf,view = True)

analisi_tsa.intercative_phase(mappa_fase)


Analisi.view_result_coutour_plot()

#cordinate = RoiSelect.select_line(mappa_modulo,titolo='Seleziona res')
#Analisi.view_result_line(cordinate[1],cordinate[0],cordinate[3],cordinate[2])
plt.show()
RoiSelect.interactive_cmap(np.abs(mappa_modulo))




