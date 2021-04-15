import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation 
from matplotlib.widgets import Slider

import pysfmov
from scipy.io import loadmat

class TSA:
    def __init__(self,fa:float,file_path='data.mat',video_npy = None):
        ''' Nota. il video è un oggetto numpy ed è utilizzato è organizzato come (x,y,t)
        '''
        self.__file_path = file_path
        if video_npy != None:
            self.finestra_video = video_npy
        else:
            if file_path[-3:] == 'mat': # rivedi fai con funzione di Os!
                self.finestra_video = loadmat(file_path,squeeze_me = True)['video']
            elif file_path[-5:] == 'sfmov':
                print(file_path)
                self.finestra_video = np.moveaxis(pysfmov.get_data(self.__file_path),0,-1) # (t,x,y) => (x,y,t)
                print(pysfmov.get_meta_data(self.__file_path))
            elif file_path[-3:] == 'npy':
                self.finestra_video = np.load(file_path)
            else:
                print('** Path non valido') # devi da errore
        # Dati
        self.__fa = fa # f acquiszione Hz
        self.__fr = 0 # f riferimento corretta Hz
        (self.__dx,self.__dy,self.__N) = self.finestra_video.shape
        self.finestra_video_roi_offset = self.finestra_video[:,:,0].copy()
        self.finestra_video_roi = np.zeros([self.__dx,self.__dy,self.__N])
        self.finestra_video_roi_mask = np.ones([self.__dx,self.__dy])
        self.map_amplitude = np.empty([self.__dx,self.__dy])
        self.map_phase = np.empty([self.__dx,self.__dy])
        # utilità
        self.__cordinate_roi = (0,0,self.__dx,self.__dy)
        (self.__dx_roi,self.__dy_roi,self.__N_roi) = (self.__dx,self.__dy,self.__N)
        self.__fa = fa # f acquiszione Hz
        self.__fr = 0 # f riferimento corretta Hz
        self.__fr_original = 0 # # f riferimento Hz
        self.__df = 0 # banda filtro Hz
        self.__tag_analysis = False # analisi svolta? [true/false]
        self.__tag_roi = False # roi selezionata? [true/false] 
        # temp
        self.__fase_picco = 0

    
    def __repr__(self):
        return f"Video per TSA:'{self.__file_path}'\nfa [Hz] = {self.__fa} , fr [Hz] = {str(self.__fr)}, df [Hz] = {str(self.__df)}"

    def freq_detection(self,fr,xi=0,yi=0,xf=0,yf=0,df = 4,view = False):
        '''
            (xi,yi) --> coordinate del vertice rettangolo [pixel]
            (xf,yf) --> coordinate del vertice opposto rettangolo [pixel]
            fr --> frequenza di riferimento [Hz]
            df --> banda di senisibiltà attorno fr [Hz]
        '''
        if xf == 0 or xf == None:
            xf = self.__dx_roi
        if yf == 0 or yf == None:
            yf = self.__dy_roi
        dx = xf-xi
        dy = yf-yi
        self.__df = df
        self.__fr = fr
        if xi+dx < self.__dx_roi and yi+dy < self.__dy_roi:
            if df:                
                ni = int(self.__N_roi%(self.__fa/self.__fr)) # riduco leakage
                N = self.__N_roi-ni
                roi_detection = self.finestra_video_roi[xi:xi+dx,yi:yi+dy,ni:].copy()
                n_lim = (np.array([fr-df/2,fr+df/2])*(N/self.__fa)).astype(int)
                signal = np.zeros(N)
                f = (self.__fa/N)*np.arange((N)//2+1)
                time = (1/self.__fa)*np.arange(N)
                N_media = np.count_nonzero(self.finestra_video_roi_mask[xi+dx,yi+dy]) #dx*dy
                for x in range(dx): # media
                    for y in range(dy):
                        if self.finestra_video_roi_mask[xi+x,yi+y] == 1:
                            signal[:] += roi_detection[x,y,:]/N_media
                S_fft = np.abs(np.fft.rfft(signal)/N)
                k_scale = 0.3
                n_list = get_local_max(S_fft[n_lim[0]:n_lim[1]],k_scale=k_scale) + n_lim[0]
                print(n_list)
                flag_max = False
                if n_list.size == 1:
                    n = np.float(n_list)
                else:
                    n_list_size = n_list.size
                    print(f'* sono presenti {n_list_size} alternative! soglia al {k_scale*100} [%]')
                    flag_max = True
                #
                if view:
                    s_fft_max = np.max(2*S_fft)
                    _,ax = plt.subplots(2,1,figsize=(16,9))
                    ax[0].plot(f,2*S_fft)
                    ax[0].grid(True)
                    ax[0].set(title=r'F = FFT[f(x_0,y_0,t)] ',xlabel='f [Hz]')
                    ax[0].vlines(self.__fr_original,*[0,s_fft_max])
                    ax[0].vlines(n_lim[0]*self.__fa/N,*[0,s_fft_max])
                    ax[0].vlines(n_lim[1]*self.__fa/N,*[0,s_fft_max])
                    for n in n_list:
                        ax[0].plot(f[n],np.abs(2*S_fft[n]),'r*')
                    ax[1].plot(time,signal)
                    ax[1].grid(True)
                    ax[1].set(title='f(x_0,y_0,t) ',xlabel='t [s]')
                    plt.show()
                if flag_max:
                    flag_utente = 0
                    while (flag_utente:= input(f"Scegli la frequenza: (Enter numero {np.arange(n_list_size)} per f = {n_list*(self.__fa/N)}) : ... ").lower()) not in [str(i) for i in range(n_list_size)]: pass
                    n = n_list[int(flag_utente)]

                #print(n*(self.__fa/N))
                self.__fr = (self.__fa/N)*(S_fft[n]*n+S_fft[n-1]*(n-1)+S_fft[n+1]*(n+1))/(S_fft[n]+S_fft[n-1]+S_fft[n+1])
                wr = self.__fr*(2*np.pi)
                SgnSF = np.sin(wr*time)*signal
                SgnSG = np.cos(wr*time)*signal
                sf = np.abs(np.mean(SgnSF)*2)
                sg = np.abs(np.mean(SgnSG)*2)
                fase_picco = np.arctan2(sg,sf)
                self.__fase_picco = fase_picco
                print(f"frequenza di carico compensata : {self.__fr} [Hz] con df = 1/T = {self.__fa/self.__N_roi} e angolo di fase th = {self.__fase_picco*(180/np.pi)}")
                return self.__fr,_,self.__fase_picco
            else: 
                print(' roi compensazione non valida')
                self.__fr_original = fr


    def set_ROI(self,xi,yi,xf,yf,ni = 0,view = False): # aggiungi controllo se xi None
        ''' Seleziono la ROI su cui fare l'analisi TSA
            (xi,yi) --> coordinate del vertice rettangolo [pixel]
            (xf,yf) --> coordinate del vertice opposto rettangolo [pixel]
            ni --> cosidero dal n-esimo frame
            view --> attiva o meno la visualizzazione
        '''
        dx = xf-xi
        dy = yf-yi
        self.__tag_analysis = False
        self.__tag_roi = True
        if xi+dx <= self.__dx and yi+dy <= self.__dy and xi >= 0 and yi >= 0 and dx>0 and dy>0:
            self.__cordinate_roi = (xi,yi,xf,yf)
            (self.__dx_roi,self.__dy_roi) = (dx,dy)
            self.finestra_video_roi = self.finestra_video[xi:xi+dx,yi:yi+dy,:].copy()
            self.finestra_video_roi_offset = self.finestra_video[xi:xi+dx,yi:yi+dy,0].copy()
            self.finestra_video_roi_mask = np.ones((self.__dx_roi,self.__dy_roi))
        else:
            print('Roi non impostata')
            (dx,dy) = (self.__dx,self.__dy)
            self.finestra_video_roi = self.finestra_video.copy()
            self.finestra_video_roi_offset = np.zeros([dx,dy])
        plt.imshow(self.finestra_video_roi_offset)
        plt.show()
        for x in range(dx):
            for y in range(dy):
                self.finestra_video_roi_offset[x,y] =  self.finestra_video_roi[x,y,0].copy() #matrice_temporanea[x,y,0]
                self.finestra_video_roi[x,y,:] += - self.finestra_video_roi_offset[x,y] #np.mean(matrice_temporanea[x,y,:])
        if view:
            fig_1 = plt.figure(constrained_layout=True)
            spec_1 = fig_1.add_gridspec(1,2)
            ax_0 = fig_1.add_subplot(spec_1[0,0])
            ax_0.set(title='Frame originale')
            ax_0.imshow(self.finestra_video[:,:,1])
            rect = patches.Rectangle((yi,xi),dy,dx,linewidth=1,edgecolor='r',facecolor='none') # ricorda verso immagine e verso matrice
            ax_0.add_patch(rect)
            ax_1 = fig_1.add_subplot(spec_1[0,1])
            ax_1.imshow(self.finestra_video_roi[:,:,1]+self.finestra_video_roi_offset)
            ax_1.set(title='Finestra ROI f(x,y,t = 0)')
            plt.show()

    def selezione_ROI_foro(self,x1,y1,x2,y2,reset=True):
        if reset:
            self.set_ROI(*self.__cordinate_roi) # da gesti meaglio
        a = (x2 - x1)/2
        b = (y2 - y1)/2
        x_0 = (x2 + x1)/2
        y_0 = (y2 + y1)/2
        self.__N_ROI_hole = 0
        for x in range(int(2*a)+1):
            for y in range(int(2*b)+1):
                if ((x+x1+1/2-x_0)/a)**2+((y1+y+1/2-y_0)/b)**2 <= 1:
                    self.finestra_video_roi_mask[x+x1,y+y1,:] = 0

    def lockin(self,fr=None,view = False,t_lim_inf = None,t_lim_sup = None):
        ''' Ottengo la mappa di modulo e fase, tramite lock-in.
        fr --> frequenza di carico [Hz]
        view --> attiva o meno la visualizzazione delle mappe
        Output:
        mappa modulo T [K]
        mappa fase [deg]
        '''
        # contollo e utility
        if fr:
            self.__fr = fr
            self.__fr_original = fr
        print(f'* lockin a fr = {self.__fr}')
        if not(self.__tag_roi):
            self.set_ROI(0,0,self.__dx,self.__dy)
        self.__tag_analysis = True
        # dati
        wr = (2*np.pi)*self.__fr
        Ix = np.empty([self.__dx_roi,self.__dy_roi])
        Iy = np.empty([self.__dx_roi,self.__dy_roi])
        t = (1/self.__fa)*np.arange(self.__N_roi)
        SgnF = np.sin(wr*t+self.__fase_picco+np.pi)
        SgnG = np.cos(wr*t+self.__fase_picco+np.pi)
        for x in range(self.__dx_roi):
            for y in range(self.__dy_roi):
                if self.finestra_video_roi_mask[x,y]:
                    Ix[x,y] = np.mean(self.finestra_video_roi[x,y,:]*SgnF) 
                    Iy[x,y] = np.mean(self.finestra_video_roi[x,y,:]*SgnG)
                else:
                    Ix[x,y] = 0
                    Iy[x,y] = 0
        self.map_amplitude = 2*np.sqrt(Ix**2 + Iy**2)
        self.map_phase = np.arctan2(Iy,Ix)*(180/np.pi)
        if view:
            self.view_result(t_lim_inf,t_lim_sup)        
        return self.map_amplitude,self.map_phase

    def view_result(self,t_lim_inf = None,t_lim_sup = None,interactive = False,save=False,path=''):
        ''' Visualizza area selezionata o la mappa del modulo e fase
        '''
        # definisco i limiti
        p_sup = 0.96 # [%] dei valori limite sup
        p_inf = 0.4
        istogramma = np.zeros(self.__dx_roi*self.__dy_roi)
        istogramma = np.sort(np.reshape(self.map_amplitude,self.__dx_roi*self.__dy_roi))
        if t_lim_sup is None:
            t_lim_sup = istogramma[int(p_sup*self.__dx_roi*self.__dy_roi)]
        if t_lim_inf is None:
            t_lim_inf = istogramma[int(p_inf*self.__dx_roi*self.__dy_roi)]

        if interactive:
            (t_lim_inf,t_lim_sup) = interactive_cmap(self.map_amplitude,titolo = 'Mappa modulo',lim_inf=t_lim_inf,lim_sup=t_lim_sup)
        else:
            fig = plt.figure(constrained_layout=True)
            spec = fig.add_gridspec(1,2)
            fig.suptitle(f'risultati per fr = {self.__fr:.{2}f} [Hz]', fontsize=16)
            ax_0 = fig.add_subplot(spec[0,0])
            temp = ax_0.imshow(self.map_amplitude,cmap = 'inferno',clim = [t_lim_inf,t_lim_sup])  # magma o inferno
            fig.colorbar(temp,orientation = 'vertical',fraction = 0.10) 
            ax_0.set(title=f'Modulo ')
    
            ax_1 = fig.add_subplot(spec[0,1])
            temp = ax_1.imshow(self.map_phase,cmap='twilight',clim = [-180,180])
            fig.colorbar(temp,orientation = 'vertical',fraction = 0.5)
            ax_1.set(title=f'fase [deg]')
            if save:
                plt.savefig(path+f'fa{self.__fa}_fr{self.__fr:.{2}f}.png')
            else:
                plt.show()

        return (t_lim_inf,t_lim_sup)

    
    def view(self,t_lim_inf = None,t_lim_sup = None): 
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
        plt.imshow(map_std,cmap = 'inferno')
        plt.show  
        return map_std

    def view_animate(self,ROI=True):
        fig = plt.figure()
        ims = []
        temp = []
        if ROI and self.__tag_roi:
            for i in range(self.__N_roi):
                temp = plt.imshow(self.finestra_video_roi[:,:,i]+self.finestra_video_roi_offset,cmap = 'inferno', animated=True)
                ims.append([temp])
        else:
            for i in range(self.__N):
                temp = plt.imshow(self.finestra_video[:,:,i],cmap = 'inferno', animated=True)
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
            res_2.append(self.map_amplitude[i[0],i[1]])
        plt.plot(res_2)
        plt.plot(np.mean(res_2)*np.ones(len(res_2)))
        plt.show()

    def get_result(self):
        return self.map_amplitude,self.map_phase

    def get_window(self):
        return self.finestra_video

    def get_roi(self):
        return self.finestra_video_roi[:,:,1]

    def get_roi_offset(self):
        return self.finestra_video_roi_offset[:,:,1]
    
def __update(cmap_lim_inf,cmap_lim_sup,AxesImage,fig):
    lim_inf = cmap_lim_inf.val
    lim_sup = cmap_lim_sup.val
    if lim_inf<lim_sup:
        AxesImage.set_clim([lim_inf,lim_sup])
    fig.canvas.draw_idle()

def interactive_cmap(matrice_immagine,titolo='Immagine',lim_inf=None,lim_sup=None):
    ''' 
    input:
        image matrix
    output:
        (x1,y1,x2,y2)
    '''
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set(title=titolo)
    if lim_inf is None:
        lim_inf = 1.1*np.min(matrice_immagine)
    if lim_sup is None:
        lim_sup = 0.9*np.max(matrice_immagine)
    AxesImage = plt.imshow(matrice_immagine,cmap='magma')
    fig.colorbar(AxesImage,orientation = 'vertical',fraction = 0.5)
    

    axcolor = 'lightgoldenrodyellow'
    ax_lim_inf = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
    ax_lim_sup = plt.axes([0.15, 0.25, 0.0225, 0.63], facecolor=axcolor)

    cmap_lim_inf = Slider(ax_lim_inf, 'inf', lim_inf/1.1, lim_sup/0.9, valinit=lim_inf,orientation="vertical")
    cmap_lim_sup = Slider(ax_lim_sup, 'sup', lim_inf/1.1, lim_sup/0.9, valinit=lim_sup,orientation="vertical")
    cmap_lim_inf.on_changed(lambda temp: __update(cmap_lim_inf,cmap_lim_sup,AxesImage,fig))
    cmap_lim_sup.on_changed(lambda temp: __update(cmap_lim_inf,cmap_lim_sup,AxesImage,fig))
    plt.show()
    return (cmap_lim_inf.val,cmap_lim_sup.val)

def get_local_max(signal,k_scale = 0.5):
    N = len(signal)
    n_max = []
    value_max = []
    for i in range(1,N-1):
        if (signal[i] > signal[i-1]) and (signal[i] > signal[i+1]):
            n_max.append(i)
            value_max.append(signal[i])
    n_max = np.array(n_max,dtype=int)
    n_max = n_max[np.argsort(value_max)]
    value_max = np.sort(value_max)
    return n_max[value_max>(k_scale*signal[n_max[-1]])]

def taglio_video(analysis_object,xi,yi,xf,yf,view=False,save_path = 'data'):
    ''' Permette di selezionare una regione d'interesse rettangolare su cui svolgere l'analisi
        (xi,yi) --> coordinate del vertice rettangolo [pixel]
        (xf,yf) --> coordinate del vertice opposto rettangolo [pixel]
        view --> attiva o meno la visualizzazione
    '''
    dx = xf-xi
    dy = yf-yi
    video = analysis_object.get_window()
    (dx_originale,dy_originale,_) = video.shape
    if xi+dx < dx_originale and yi+dy < dy_originale:
        np.save(save_path,video[xi:xi+dx,yi:yi+dy,:])
        if view:
            plt.imshow(video[xi:xi+dx,yi:yi+dy,1])
            plt.title('Finestra f(x,y,t = 0)')
            plt.show()
    else:
        print('Finestra non valida')

def main():
    pass

if __name__ == '__main__':
    main()



