import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'..')
import pytsa 
import RoiSelect

fa = 100
fr = 45

x = 300
y = 300
z = 1032
matrice = np.zeros([x,y,z])
A = 1
th_offset = (np.pi/180)*65
th = -90*(np.pi/180)+th_offset
# costruiamo la matrice per la prova, divisa in tre zone : zona con fase nulla (solo rumore), zona con fase di th_offset (ampiezza A) e zona con th da individuare (ampezza 2A)
print(f'phase offset : {th_offset*180/np.pi} \nphase obj: {(th-th_offset)*180/np.pi} \nphase_res {th*180/np.pi}')
wr = fr*(2*np.pi)
t = np.arange(z)*(1/fa)
for i in range(x):
    for j in range(y//2):
        matrice[i,j,:] = A*np.sin(wr*t+th_offset)
for i in range(2*x//3,x):
    for j in range(2*y//3,y):
        matrice[i,j,:] = 2*A*np.sin(wr*t+th)
matrice += 2*np.random.rand(x,y,z)
np.mean(matrice)
Analisi = pytsa.TSA(fa,video_npy=matrice)

cordinate = RoiSelect.selectROI(Analisi.get_roi_offset(),titolo='Seleziona finestra di compensazione')
Analisi.freq_detection(fr,cordinate[1],cordinate[0],cordinate[3],cordinate[2],df = 3,view=True)

(mappa_modulo,mappa_fase)=Analisi.lockin(view = True,t_lim_inf = None, t_lim_sup = None)

Analisi.set_phase_offset()

