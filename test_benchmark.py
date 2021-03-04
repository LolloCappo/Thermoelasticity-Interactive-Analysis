import matplotlib.pyplot as plt
import numpy as np
import analisi_tsa # modulo

import RoiSelect

fa = 100 # freq ac1uisiszione
fr = 35

x = 300
y = 300
z = 1032
matrice = np.zeros([x,y,z])
A = 1
th_offset = (np.pi/180)*25
th = -50*(np.pi/180)+th_offset
print(f'phase offset : {th_offset*180/np.pi} \nphase obj: {(th-th_offset)*180/np.pi} \nphase_res {th*180/np.pi}')
wr = fr*(2*np.pi)
t = np.arange(z)*(1/fa)
flag = False
if flag:
    for i in range(x):
        for j in range(y//2):
            matrice[i,j,:] = A*np.sin(wr*t+th_offset)
    for i in range(2*x//3,x):
        for j in range(2*y//3,y):
            matrice[i,j,:] = 2*A*np.sin(wr*t+th)
    matrice += 2*np.random.rand(x,y,z)
    np.save('data',matrice)
Analisi = analisi_tsa.TSA(fa,'data.npy')
cordinate = RoiSelect.selectROI(Analisi.finestra_video[:,:,1],titolo='Seleziona ROI')
Analisi.selezione_ROI(cordinate[1],cordinate[0],cordinate[3],cordinate[2],view = False)

cordinate = RoiSelect.selectROI(Analisi.finestra_video_roi_offset,titolo='Seleziona finestra di compensazione')
Analisi.selezione_ROI_compensazione(fr,cordinate[1],cordinate[0],cordinate[3],cordinate[2],df = 3,view=True)

(mappa_modulo,mappa_fase)=Analisi.lockin(view = True,t_lim_inf = None, t_lim_sup = None)


