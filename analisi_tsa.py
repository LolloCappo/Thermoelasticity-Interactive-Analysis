import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation 

import pysfmov
from scipy.io import loadmat

class TSA:
    def __init__(self,fa,file_path='data.mat'):
        self.__file_path = file_path
        if file_path[-3:] == 'mat': # rivedi fai con funzione di Os!
            self.finestra_video = loadmat(file_path,squeeze_me = True)['video']
        elif file_path[-5:] == 'sfmov':
            self.finestra_video = np.moveaxis(pysfmov.get_data(self.__file_path),0,-1) # (t,x,y) => (x,y,t)
        elif file_path[-3:] == 'npy':
            self.finestra_video = np.load(file_path)
        else:
            print('** Path non valido') # devi da errore
        (self.__dx,self.__dy,self.__N) = self.finestra_video.shape
        self.__cordinate_roi = (0,0,self.__dx,self.__dy)
        self.finestra_video_roi = np.zeros([self.__dx,self.__dy,self.__N])
        self.finestra_video_roi_media = np.zeros([self.__dx,self.__dy])
        (self.__dx_roi,self.__dy_roi,self.__N_roi) = (self.__dx,self.__dy,self.__N)
        self.__fa = fa # f acquiszione Hz
        self.__fr = 0 # f riferimento Hz
        self.__df = 0 # banda filtro Hz
        self.tag = 'Nessuna analisi svolta'
        self.__tag_roi = False # true roi selezionata. false roi non selezionata
        self.mappa_modulo = np.empty([self.__dx,self.__dy])
        self.mappa_fase = np.empty([self.__dx,self.__dy])
    
    def __repr__(self):
        return f"Video per TSA:'{self.__file_path}'\nAnalisi: {self.tag}\nfa [Hz] = {self.__fa} , fr [Hz] = {str(self.__fr)}, df [Hz] = {str(self.__df)}"
    
    def selezione_ROI_compensazione(self,xi,yi,xf,yf,fr,df = 5,view = False):
        '''
            (xi,yi) --> coordinate del vertice rettangolo [pixel]
            (xf,yf) --> coordinate del vertice opposto rettangolo [pixel]
            fr --> frequenza di riferimento [Hz]
            df --> banda di senisibiltà attorno fr [Hz]
        '''
        dx = xf-xi
        dy = yf-yi
        fr_originario = fr
        self.__df = df
        self.__fr = fr
        if xi+dx < self.__dx_roi and yi+dy < self.__dy_roi:
            if df:                
                n = 0
                f_lim = (fr-df/2,fr+df/2)
                f = (self.__fa/self.__N_roi)*np.arange(self.__N_roi//2+1)
                segnale = np.zeros(self.__N_roi)
                
                Numero_non_nan = np.count_nonzero(self.finestra_video_roi[:,:,1]) # elementi non nan
                for x in range(dx): # media
                    for y in range(dy):
                        segnale += self.finestra_video_roi[xi+x,yi+y,:]/(Numero_non_nan)
                (modulo_picco,fase_picco,n) = self.__Seleziona_banda(segnale,f,*f_lim)       
                self.__fr = n*(self.__fa/self.__N_roi)
                print(f"frequenza di carico compensata :'{self.__fr})' [Hz] con df = 1/T = {self.__fa/self.__N_roi}")
                if view:
                    _,ax = plt.subplots(2,1,figsize=(16,9))
                    fft = np.abs(np.fft.rfft(segnale))/self.__N_roi
                    ax[0].plot(f,2*fft)
                    ax[0].grid(True)
                    ax[0].set(title='F = FFT[f(x_0,y_0,t)] ',xlabel='f [Hz]')
                    ax[0].vlines(fr_originario,*[0,np.max(2*fft)])
                    ax[0].vlines(f_lim[0],*[0,np.max(2*fft)])
                    ax[0].vlines(f_lim[1],*[0,np.max(2*fft)])

                    t = (1/self.__fa)*np.arange(self.__N_roi)
                    ax[1].plot(t,segnale)
                    ax[1].grid(True)
                    ax[1].set(title='f(x_0,y_0,t) ',xlabel='t [s]')
                    plt.show()
                return self.__fr,modulo_picco,fase_picco
                
    def selezione_ROI(self,xi,yi,xf,yf,ni = 0,view = False): # aggiungi controllo se xi None
        ''' Seleziono la ROI su cui fare l'analisi TSA, rispetto al caso precedente

            (xi,yi) --> coordinate del vertice rettangolo [pixel]
            (xf,yf) --> coordinate del vertice opposto rettangolo [pixel]
            ni --> cosidero dal n-esimo frame
            view --> attiva o meno la visualizzazione
        '''
        dx = xf-xi
        dy = yf-yi
        if xi+dx <= self.__dx and yi+dy <= self.__dy and ni<self.__N and ni >= 0 and xi >= 0 and yi >= 0 and dx>0 and dy>0:
            self.__cordinate_roi = (xi,yi,xf,yf)
            (self.__dx_roi,self.__dy_roi) = (dx,dy)
            self.mappa_modulo = np.empty([dx,dy])
            self.mappa_fase = np.empty([dx,dy])
            self.tag = 'Nessuna analisi svolta'
            self.__tag_roi = True
            self.__N_roi = self.__N - ni
            self.finestra_video_roi = np.empty([dx,dy,self.__N_roi]) 
            matrice_temporanea = self.finestra_video[xi:xi+dx,yi:yi+dy,ni:]
            self.finestra_video_roi_media = np.empty([dx,dy])

            for x in range(dx):
                for y in range(dy):
                    self.finestra_video_roi_media[x,y] = np.mean(matrice_temporanea[x,y,:]) #matrice_temporanea[x,y,0]#np.mean(self.finestra_video_roi[x,y,:]) #matrice_temporanea[x,y,0]#
                    self.finestra_video_roi[x,y,:] = matrice_temporanea[x,y,:] - np.mean(matrice_temporanea[x,y,:]) 

            if view:
                fig_1 = plt.figure(constrained_layout=True)
                spec_1 = fig_1.add_gridspec(1,2)
                ax_0 = fig_1.add_subplot(spec_1[0,0])
                ax_0.set(title='Frame originale')
                ax_0.imshow(self.finestra_video[:,:,1])
                rect = patches.Rectangle((yi,xi),dy,dx,linewidth=1,edgecolor='r',facecolor='none') # ricorda verso immagine e verso matrice
                ax_0.add_patch(rect)
                ax_1 = fig_1.add_subplot(spec_1[0,1])
                ax_1.imshow(self.finestra_video_roi[:,:,1]+self.finestra_video_roi_media)
                ax_1.set(title='Finestra ROI f(x,y,t = 0)')
                plt.show()
        else:
            print('Roi non valida')

    def selezione_ROI_foro(self,x1,y1,x2,y2,reset=True):
        if reset:
            self.selezione_ROI(*self.__cordinate_roi) # da gesti meaglio
        a = (x2 - x1)/2
        b = (y2 - y1)/2
        x_0 = (x2 + x1)/2
        y_0 = (y2 + y1)/2
        for x in range(int(2*a)+1):
            for y in range(int(2*b)+1):
                if ((x+x1+1/2-x_0)/a)**2+((y1+y+1/2-y_0)/b)**2 <= 1:
                    self.finestra_video_roi[x+x1,y+y1,:] = 0

    def taglio_video(self,xi,yi,xf,yf,ni = 0,save = False,view=False,save_path = 'data'):
        ''' Permette di selezionare una regione d'interesse rettangolare su cui svolgere l'analisi
        (xi,yi) --> coordinate del vertice rettangolo [pixel]
        (xf,yf) --> coordinate del vertice opposto rettangolo [pixel]
        view --> attiva o meno la visualizzazione
        '''
        dx = xf-xi
        dy = yf-yi
        (dx_originale,dy_originale,_) = self.finestra_video.shape
        if xi+dx < dx_originale and yi+dy < dy_originale:
            self.finestra_video = self.finestra_video[xi:xi+dx,yi:yi+dy,:]
            self.finestra_video_roi = self.finestra_video 
            (self.__dx,self.__dy) = (dx,dy)
            self.mappa_modulo = np.empty([dx,dy])
            self.mappa_fase = np.empty([dx,dy])
            self.tag = 'Nessuna analisi svolta'
            self.__tag_roi = False
            if view:
                plt.imshow(self.finestra_video[:,:,0])
                plt.title('Finestra f(x,y,t = 0)')
                plt.show()
            if save:
                np.save(save_path,self.finestra_video)
            return self.finestra_video[xi:xi+dx,yi:yi+dy,:]
        else:
            print('Finestra non valida')

    def ottieni_mappe_lockin(self,fr=None,view = False,t_lim_inf = None,t_lim_sup = None):
        ''' Ottengo la mappa di modulo e fase, tramite lock-in.
        fr --> frequenza di carico [Hz]
        view --> attiva o meno la visualizzazione delle mappe
        Output:
        mappa modulo T [K]
        mappa fase [deg]
        '''
        if fr:
            self.__fr = fr
        if not(self.__tag_roi):
            self.selezione_ROI(0,0,self.__dx,self.__dy)
            self.__tag_roi = True
        self.tag = 'lock-in'
        wr = (2*np.pi)*self.__fr
        Ix = np.empty([self.__dx_roi,self.__dy_roi])
        Iy = np.empty([self.__dx_roi,self.__dy_roi])
        t = (1/self.__fa)*np.arange(self.__N_roi)

        for x in range(self.__dx_roi):
            for y in range(self.__dy_roi):
                Ix[x,y] = np.mean(self.finestra_video_roi[x,y,:]*np.sin(wr*t)*2) 
                Iy[x,y] = np.mean(self.finestra_video_roi[x,y,:]*np.cos(wr*t)*2)
                #Ix[x,y] = np.mean(self.__filtro_passabasso(self.finestra_video_roi[x,y,:]*np.sin(wr*t)*2,self.__fr*1.5)) 
                #Iy[x,y] = np.mean(self.__filtro_passabasso(self.finestra_video_roi[x,y,:]*np.cos(wr*t)*2,self.__fr*1.5))
        self.mappa_modulo = np.sqrt(Ix**2 + Iy**2)
        self.mappa_fase = np.arctan2(Iy,Ix)*(180/np.pi)
        if view:
            self.view_result(t_lim_inf,t_lim_sup)        
        return self.mappa_modulo,self.mappa_fase

    def __filtro_passabasso(self,S_t,ft:float,n:int=3):
        ''' Dato il segnale nel tempo, lo filtra e restituisce nel tempo
        S_t --> segnale originale
        ft  --> frequenza di taglio
        '''
        S_fft = np.fft.rfft(S_t) # fft da normalizzare /N
        frq = (self.__fa/self.__N_roi)*np.arange(self.__N_roi//2+1)
        #h = [0.5 + 0.5*np.cos(np.pi*(f/ft)) if f<ft else 0 for f in frq ] # hamming
        h = np.empty(self.__N//2+1,dtype=complex)
        h = 1/(1 + 1j*(frq/ft))**n
        return np.fft.irfft(h*S_fft) # dato ho fft non /N, resta così

    def __Seleziona_banda(self,S_t,frq,ft_inf:float,ft_sup:float):
        ''' Dato il segnale nel tempo, converte in fft, restringe nel campo di frequenza
        imposta e restituisce (Come modulo e fase e frequenza esatta) l'armonica massima
        S_t --> segnale originale
        frq --> vettore frequenze
        ft_inf,ft_sup  --> frequenza di taglio inferiore e superiore
        '''
        N = S_t.size
        S_fft = np.fft.rfft(S_t)/N
        h = [(1-0.16)/2 - 1/2*np.cos(2*np.pi*(f-ft_inf)/(ft_sup-ft_inf)) + 0.16/2*np.cos(4*np.pi*(f-ft_inf)/(ft_sup-ft_inf)) if f<ft_sup and f>ft_inf else 0 for f in frq]
        S_fft *= h
        n = np.argmax(np.abs(S_fft)) 
        return (2*np.abs(S_fft[n]),- np.angle(S_fft[n],deg=True),n)

    def view_result(self,t_lim_inf = None,t_lim_sup = None):
        ''' Visualizza area selezionata o la mappa del modulo e fase
        '''
        if t_lim_sup == None:
            # definisco i limiti
            p_sup = 0.96 # [%] dei valori limite sup
            p_inf = 0.4
            istogramma = np.zeros(self.__dx_roi*self.__dy_roi)
            istogramma = np.sort(np.reshape(self.mappa_modulo,self.__dx_roi*self.__dy_roi))
            t_lim_sup = istogramma[int(p_sup*self.__dx_roi*self.__dy_roi)]
            t_lim_inf = istogramma[int(p_inf*self.__dx_roi*self.__dy_roi)]
            
            t_mean = np.mean(istogramma)
            p = 0.7
            dt_max = (1-p)*(istogramma[-1] - t_mean)
            dt_min = p*(t_mean - istogramma[0])
            t_lim_sup = istogramma[(istogramma-t_mean>=dt_max).argmax(axis=0)]
            t_lim_inf = istogramma[(t_mean-istogramma<=dt_min).argmax(axis=0)]
            # Fase
            p_sup = 0.92 # [%] dei valori limite sup
            p_inf = 0.4
            istogramma = np.zeros(self.__dx_roi*self.__dy_roi)
            istogramma = np.sort(np.reshape(self.mappa_fase,self.__dx_roi*self.__dy_roi))
            t_lim_sup_fase = istogramma[int(p_sup*self.__dx_roi*self.__dy_roi)]
            t_lim_inf_fase = istogramma[int(p_inf*self.__dx_roi*self.__dy_roi)]
        #plt.plot(istogramma)
        #plt.show()
        #box = np.ones(9)/9
        #diff_istogramma = np.convolve(istogramma, box, mode='same')
        #diff_istogramma = np.diff(istogramma)
        #plt.plot(diff_istogramma)
        #plt.show()
        #soglia = 0.0005
        #diff_istogramma[diff_istogramma < soglia] = 0
        #t_lim_sup = istogramma[np.argmax(diff_istogramma)]
        #t_lim_sup = istogramma[(diff_istogramma!=0).argmax(axis=0)]
        #print(t_lim_sup)
        #plt.plot(diff_istogramma)
        #plt.show()
        #
        istogramma = np.sort(np.reshape(self.mappa_fase,self.__dx_roi*self.__dy_roi))
        #plt.plot(istogramma)
        #plt.show()

        fig = plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(1,2)
        fig.suptitle(f'risultati per fr = {self.__fr:.{2}f} [Hz]', fontsize=16)
        ax_0 = fig.add_subplot(spec[0,0])
        temp = ax_0.imshow(self.mappa_modulo,cmap = 'inferno',clim = [t_lim_inf,t_lim_sup])  # magma o inferno
        fig.colorbar(temp,orientation = 'vertical',fraction = 0.10) 
        ax_0.set(title=f'Modulo {self.tag} ')
    
        ax_1 = fig.add_subplot(spec[0,1])
        mappa_fase_sogliata = np.zeros(self.mappa_fase.shape)
        mappa_fase_sogliata = self.mappa_fase - np.mean(self.mappa_fase)*np.ones(self.mappa_fase.shape)
        istogramma = np.sort(np.reshape(mappa_fase_sogliata,self.__dx_roi*self.__dy_roi))
        t_lim_sup_fase = istogramma[int(p_sup*self.__dx_roi*self.__dy_roi)]
        t_lim_inf_fase = istogramma[int(p_inf*self.__dx_roi*self.__dy_roi)]

        (X,Y) = self.mappa_fase.shape
        for x in range(X):
            for y in range(Y):
                if mappa_fase_sogliata[x,y] < -180 :#-180:
                    mappa_fase_sogliata[x,y] = -mappa_fase_sogliata[x,y]-180
                elif mappa_fase_sogliata[x,y] > 180:
                    mappa_fase_sogliata[x,y] = -mappa_fase_sogliata[x,y]+180
        print([np.max(mappa_fase_sogliata),np.min(mappa_fase_sogliata)])
        soglia = 0.007
        print(soglia)
        #mappa_fase_sogliata[self.mappa_modulo<soglia] = 0

        temp = ax_1.imshow(mappa_fase_sogliata,cmap='twilight',clim = [-70,70])#[t_lim_inf_fase,t_lim_sup_fase]
        fig.colorbar(temp,orientation = 'vertical',fraction = 0.5)
        ax_1.set(title=f'fase {self.tag} [deg]')
        plt.show()

        plt.imshow(np.abs(self.mappa_fase)-np.mean(self.mappa_fase)*np.ones(self.mappa_fase.shape),cmap='magma',clim = [0,t_lim_sup_fase])
        plt.show()

    
    def view(self,animate = True,t_lim_inf = None,t_lim_sup = None): 
        ''' Visualizza finestra corrente
            t_lim_inf --> limite inf color bar
            t_lim_sup --> limite sup color bar
        '''
        _,ax = plt.subplots()
        temp = ax.imshow(self.finestra_video[:,:,1],cmap = 'inferno',clim = [t_lim_inf,t_lim_sup])
        plt.colorbar(temp,orientation = 'vertical',fraction = 0.5)
        plt.show()

    def view_std(self,xi=0,yi=0,xf=0,yf=0):
        ''' deviazione std e altro nella finestra selezionata
        '''
        if xf == 0:
            xf = self.__dx_roi
        if yf == 0:
            yf = self.__dy_roi
        dx = xf-xi
        dy = yf-yi
        if xi+dx <= self.__dx and yi+dy <= self.__dy and xi >= 0 and yi >= 0 and dx>0 and dy>0:
            map_std = np.empty([dx,dy])
            for x in range(dx):
                for y in range(dy):
                    map_std[x,y] = np.std( self.finestra_video_roi[x,y,:])
        plt.imshow(map_std,'inferno')
        plt.show    

    def view_animate(self,ROI=True):

        fig = plt.figure()
        ims = []
        temp = []
        if ROI and self.__tag_roi:
            for i in range(self.__N_roi):
                temp = plt.imshow(self.finestra_video_roi[:,:,i]+self.finestra_video_roi_media, animated=True)
                ims.append([temp])
        else:
            for i in range(self.__N):
                temp = plt.imshow(self.finestra_video[:,:,i], animated=True)
                ims.append([temp])
        animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=100)
        plt.show()

    def view_result_line(self,xi,yi,xf,yf):
        '''
        '''
        dx = (xf-xi)
        dy = (yf-yi)
        res = []
        if dx != 0 and dy != 0:
            u = np.arange(int(dx*dy))/int(dx*dy)
        elif dx != 0:
            u = np.arange(int(dx))/int(dx)
        elif dy != 0:
            u = np.arange(int(dy))/int(dy)
        for i in range(int(dx*dy)):
            x_v = xi + dx*u[i]
            y_v = yi + dy*u[i]
            if not([int(x_v),int(y_v)] in res):
                res.append([int(x_v),int(y_v)])
        res_2 = []
        for i in res:
            res_2.append(self.mappa_modulo[i[0],i[1]])
        plt.plot(res_2)
        plt.plot(np.mean(res_2)*np.ones(len(res_2)))
        plt.show()



def main():
    fa = 100
    fr = 5
    im1 = TSA(fa)
    im1.ottieni_mappe_lockin(fr = fr,view=True)

if __name__ == '__main__':
    main()


