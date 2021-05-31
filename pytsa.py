__version__ = '0.1'

import pysfmov
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation 
import matplotlib.patches as mpatches

from scipy import ndimage
from scipy.io import loadmat
from numba import jit
from matplotlib.widgets import Slider


class TSA:
    ''' Thermelastic signal 
    '''
    def __init__(self,fs:float,file_path='data.mat',video_npy = None):
        ''' 
        The function use the video (x,y,t)
        '''
        self.__file_path = file_path
        if not(video_npy is None):
            self.video = video_npy
        else:
            if file_path[-3:] == 'mat':
                self.video = loadmat(file_path,squeeze_me = True)['video']
            elif file_path[-5:] == 'sfmov':
                self.video = np.moveaxis(pysfmov.get_data(self.__file_path),0,-1) # (t,x,y) => (x,y,t)
                print(pysfmov.get_meta_data(self.__file_path))
            elif file_path[-3:] == 'npy':
                self.video = np.load(file_path)
            else:
                print('** Path non valido') # devi da errore
        # Dati
        self.__fs = fs # Sampling rate [Hz]
        self.__fr = 0 # reference frequency [Hz]
        (self.__dx,self.__dy,self.__N) = self.video.shape
        self.__video_roi_offset = self.video[:,:,0].copy()
        self.__video_roi = np.zeros([self.__dx,self.__dy,self.__N])
        self.__mask = np.ones([self.__dx,self.__dy]) # 1 on, 0 off
        self.__map_amplitude = np.empty([self.__dx,self.__dy])
        self.__map_phase = np.empty([self.__dx,self.__dy])
        self.__phase_offset = 0
        # utilità
        self.__cordinate_roi = (0,0,self.__dx,self.__dy)
        (self.__dx_roi,self.__dy_roi,self.__N_roi) = (self.__dx,self.__dy,self.__N)
        self.__fr_original = 0 # # set reference frequency  [Hz]
        self.__df = 0 # bandwidth frequency [Hz]
        self.__tag_analysis = False # analisi svolta? [true/false]
        self.__tag_roi = False # roi selezionata? [true/false] 
        self.__tag_cmap = False
        self.__t_lim_inf = None
        self.__t_lim_sup = None

    def __repr__(self):
        return f"Video TSA:'{self.__file_path}'\nfa [Hz] = {self.__fs} , fr [Hz] = {str(self.__fr)}, df [Hz] = {str(self.__df)}"

    def freq_detection(self,fr:float(),xi:int=0,yi:int=0,xf:int=0,yf:int=0,df:float = 4,k_scale:float() = 0.3,view:bool = False):
        '''
        Fuction to selects the peak from the FFT of the thermal signal, in the sets area, close to the set frequency (fr).
        Input:
            (xi,yi,xf,yf) --> coordinates edges of rettangle [pixel]
            fr            --> set the frequency for the reference [Hz]
            df            --> the bandwidth [Hz]
            k_scale       --> scale factor. All the peak between the max and 
                              k_ scale*max are taken
            view          --> plot
        Output:
            fr           --> the real frequency, compensated
            phase_offset --> phase between the median value of the 
                             area and the temporal origin of the video
        '''
        if xf == 0 or xf == None:
            xf = self.__dx_roi
        if yf == 0 or yf == None:
            yf = self.__dy_roi
        dx = xf-xi
        dy = yf-yi
        self.__df = df
        self.__fr = fr
        if not(self.__tag_roi):
            self.set_ROI(0,0,self.__dx,self.__dy)
        self.__tag_analysis = True            
        if xi+dx < self.__dx_roi and yi+dy < self.__dy_roi:
            if df:
                self.__fr_original = fr
                ni = int(self.__N_roi%(self.__fs/self.__fr)) # riduco leakage
                N = self.__N_roi-ni
                if ni == 0:
                    roi_detection = self.__video_roi[xi:xi+dx,yi:yi+dy,:].copy()
                else:
                    roi_detection = self.__video_roi[xi:xi+dx,yi:yi+dy,:-ni].copy()
                n_lim = (np.array([fr-df/2,fr+df/2])*(N/self.__fs)).astype(int)
                signal = np.zeros(N)
                f = (self.__fs/N)*np.arange((N)//2+1)
                time = (1/self.__fs)*np.arange(N)
                N_media = np.count_nonzero(self.__mask[xi:xi+dx,yi:yi+dy])
                for x in range(dx): # mean
                    for y in range(dy):
                        if self.__mask[xi+x,yi+y] == 1:
                            signal[:] += roi_detection[x,y,:]/N_media
                S_fft = np.abs(np.fft.rfft(signal)/N)
                n_list = get_local_max(S_fft[n_lim[0]:n_lim[1]],k_scale=k_scale)[::-1] + n_lim[0]
                flag_max = False
                if n_list.size == 1:
                    n = np.float(n_list)
                else:
                    n_list_size = n_list.size
                    print(f'* sono presenti {n_list_size} alternative! soglia al {k_scale*100} [%]')
                    flag_max = True
                if view:
                    s_fft_max = np.max(2*S_fft)
                    _,ax = plt.subplots(2,1,figsize=(16,9))
                    ax[0].plot(f,2*S_fft)
                    ax[0].grid(True)
                    ax[0].set(title=r'$FFT[ T( x_0 , y_0 , t)]$',xlabel='Frequency [Hz]')
                    ax[0].vlines(self.__fr_original,*[0,s_fft_max],color = 'r')
                    ax[0].vlines(n_lim[0]*self.__fs/N,*[0,s_fft_max],color = 'black',linestyle='--')
                    ax[0].vlines(n_lim[1]*self.__fs/N,*[0,s_fft_max],color = 'black',linestyle='--')
                    for n in n_list:
                        ax[0].plot(f[n],np.abs(2*S_fft[n]),'r*')
                    ax[1].plot(time,signal)
                    ax[1].grid(True)
                    ax[1].set(title=r'$T(x_0,y_0,t)$',xlabel='Time [s]')
                    plt.show()
                if flag_max:
                    flag_utente = 0
                    while (flag_utente:= input(f"Select the frequency: (Enter {np.arange(n_list_size)} for f = {n_list*(self.__fs/N)}) : ... ").lower()) not in [str(i) for i in range(n_list_size)]: pass
                    n = n_list[int(flag_utente)]
                self.__fr = (self.__fs/N)*(2*S_fft[n]*n+S_fft[n-1]*(n-1)+S_fft[n+1]*(n+1))/(2*S_fft[n]+S_fft[n-1]+S_fft[n+1])
                phase_map = np.empty((dx,dy))
                _,phase_map = lockin_2D(self.__video_roi[xi:xi+dx,yi:yi+dy,-ni:].copy(),np.ones([dx,dy]),self.__fr,self.__fs)
                plt.plot(np.percentile(phase_map,np.arange(0,100))*180/np.pi)
                plt.show()
                self.__phase_offset = np.mean(np.percentile(np.abs(phase_map),np.arange(45,55)))
                print(f"Frequency : {self.__fr:.{2}f} Hz df = 1/T = {self.__fs/self.__N_roi:.{2}f} s, phase = {self.__phase_offset*(180/np.pi):.{2}f} deg")
                return self.__fr,_,self.__phase_offset

            else: 
                print('invalid')
                self.__fr_original = fr
                self.__fr = fr

    def set_ROI(self,xi,yi,xf,yf,ni = 0,view = False):
        ''' 
        Function to sets the regions of interest (ROIs) in which to perform the analysis.
        Arguments:
            - (xi,yi) --> coordinate [pixel]
            - (xf,yf) --> coordinate [pixel]
            - ni --> save from the ni-th frame
            - view --> attiva o meno la visualizzazione            
        '''
        dx = xf-xi
        dy = yf-yi
        self.__tag_analysis = False
        self.__tag_roi = True
        if xi+dx <= self.__dx and yi+dy <= self.__dy and xi >= 0 and yi >= 0 and dx>0 and dy>0:
            self.__cordinate_roi = (xi,yi,xf,yf)
            (self.__dx_roi,self.__dy_roi) = (dx,dy)
            self.__video_roi = self.video[xi:xi+dx,yi:yi+dy,:].copy()
            self.__video_roi_offset = self.video[xi:xi+dx,yi:yi+dy,0].copy()
            self.__mask = self.__mask[xi:xi+dx,yi:yi+dy].copy()
        else:
            print('invalid')
            (dx,dy) = (self.__dx,self.__dy)
            self.__video_roi = self.video.copy()
            self.__video_roi_offset = np.zeros([dx,dy])
        for x in range(dx):
            for y in range(dy):
                self.__video_roi_offset[x,y] =  self.__video_roi[x,y,0].copy() 
                self.__video_roi[x,y,:] += - self.__video_roi_offset[x,y]
        if view:
            fig_1 = plt.figure(constrained_layout=True)
            spec_1 = fig_1.add_gridspec(1,2)
            ax_0 = fig_1.add_subplot(spec_1[0,0])
            ax_0.set(title='Original frame')
            ax_0.imshow(self.video[:,:,1])
            rect = patches.Rectangle((yi,xi),dy,dx,linewidth=1,edgecolor='r',facecolor='none') # ricorda verso immagine e verso matrice
            ax_0.add_patch(rect)
            ax_1 = fig_1.add_subplot(spec_1[0,1])
            ax_1.imshow(self.__video_roi[:,:,1]+self.__video_roi_offset)
            ax_1.set(title=' ROI f(x,y,t = 0)')
            plt.show()

    def set_hole(self,x1,y1,x2,y2,reset=True):
        if reset:
            self.__mask = np.ones([self.__dx_roi,self.__dy_roi])            
        dx = x2-x1
        dy = y2-y1
        if x1+dx <= self.__dx_roi and y1+dy <= self.__dy_roi and x1 >= 0 and y1 >= 0 and dx>0 and dy>0:
            a = (x2 - x1)/2
            b = (y2 - y1)/2
            x_0 = (x2 + x1)/2
            y_0 = (y2 + y1)/2
            for x in range(int(2*a)+1):
                for y in range(int(2*b)+1):
                    if ((x+x1+1/2-x_0)/a)**2+((y1+y+1/2-y_0)/b)**2 <= 1:
                        self.__mask[x+x1,y+y1] = 0
        else:
            print('invalid')

    def lockin(self,fr=None,phase_offset=None,view = False,t_lim_inf = None,t_lim_sup = None):
        ''' 
        Function to obtain magnitude and phase of a thermal acquisition usign a lock-in analyzer

        used to measure phase shift, even when the signals are large, have a high signal-to-noise ratio and do not need further improvement. 
        returns the thermelastic signal 
        - fr   --> reference frequency [Hz]
        - view --> attiva o meno la visualizzazione delle mappe
        - phase_offset --> phase of the reference signal [rad]
        Output:
        - ampitude map [K]
        - phase map [deg]
        '''
        # contollo e utility
        if fr:
            self.__fr = fr
            self.__fr_original = fr
        if phase_offset is not None:
            self.__phase_offset = phase_offset
        print(f'* lockin fr = {self.__fr}')
        if not(self.__tag_roi):
            self.set_ROI(0,0,self.__dx,self.__dy)
        self.__tag_analysis = True
        # dati
        self.__map_amplitude,self.__map_phase = lockin_2D(self.__video_roi,self.__mask,self.__fr,self.__fs,phase_offset=self.__phase_offset)
        self.__map_phase *= (180/np.pi)
        if view:
            self.view_result(t_lim_inf,t_lim_sup)        
        return self.__map_amplitude,self.__map_phase

    def view_result(self,t_lim_inf = None,t_lim_sup = None,phase_reverse = False,save=False,path='',name_file=''):
        ''' Visualizza area selezionata o la mappa del modulo e fase
        -
        - phase_reverse --> if is True, the phase is between [0,180] deg (abs of the phase map). Else, it is between [-180,180] 
        '''
        self.set_cmap_lim(t_lim_inf,t_lim_sup)
  
        fig = plt.figure(constrained_layout=True)
        spec = fig.add_gridspec(1,2)
        fig.suptitle(f'fr = {self.__fr:.{2}f} [Hz]', fontsize=16)
        ax_0 = fig.add_subplot(spec[0,0])
        temp = ax_0.imshow(self.__map_amplitude,cmap = 'inferno',clim = [self.__t_lim_inf,self.__t_lim_sup])  # magma o inferno
        fig.colorbar(temp,orientation = 'vertical',fraction = 0.10) 
        ax_0.set(title=f'Magnitude \n [units of thermal video]')
    
        ax_1 = fig.add_subplot(spec[0,1])
        if phase_reverse:
            temp = ax_1.imshow(np.abs(self.__map_phase),cmap='magma',clim = [0,180])
        else:
            temp = ax_1.imshow(self.__map_phase,cmap='twilight',clim = [-180,180])
        fig.colorbar(temp,orientation = 'vertical',fraction = 0.5)
        ax_1.set(title=f'Phase \n [deg]')
        if save:
            if name_file == '':
                name_file = f'fs{self.__fs}_fr{self.__fr_original_:.{2}f}'
            plt.savefig(path+name_file+'.png')
        else:
            plt.show()

    def set_cmap_lim(self,lim_inf=None,lim_sup=None,reset=False,interactive=False):
        t_lim_inf_temp,t_lim_sup_temp = set_clim(self.__map_amplitude)
        if not(reset):
            if lim_sup is None:
                self.__t_lim_sup = t_lim_sup_temp
            if lim_inf is None:
                self.__t_lim_inf = t_lim_inf_temp
            if interactive:
                (self.__t_lim_inf,self.__t_lim_sup) = interactive_cmap(self.__map_amplitude,titolo = 'interactive cmap',lim_inf=t_lim_inf_temp,lim_sup=t_lim_sup_temp)
        return (self.__t_lim_inf,self.__t_lim_sup)


    def set_phase_offset(self):
        theta_offset = interactive_phase(self.__map_phase,self.__mask,titolo='Mappa fase')
        self.__phase_offset += theta_offset
        for x in range(self.__dx_roi):
            for y in range(self.__dy_roi):   
                self.__map_phase[x,y] = abs(self.__map_phase[x,y] + theta_offset)%180 *(-1)*np.sign(self.__map_phase[x,y] + theta_offset)

    def save(self,path = '',name_file=''):
        ''' save
        '''
        if name_file == '':
            name_file = f'fs{self.__fs}_fr{self.__fr:.{2}f}'
        self.view_result(save=True,path=path,name_file=name_file)
        np.save(path + name_file +'.npy',np.array([self.__map_amplitude,self.__map_phase]))
    
    def view_result_coutour_plot(self,sigma=1.5,levels = []):
        N = self.__dx_roi*self.__dy_roi
        istogramma = np.zeros(N)
        istogramma = np.sort(np.reshape(self.__map_amplitude,N))
        if not levels:
            levels = [istogramma[int(0.2*N)],istogramma[int(0.5*N)],istogramma[int(0.65*N)],istogramma[int(0.80*N)],istogramma[int(0.95*N)]]
        mappa = self.__map_amplitude.copy()
        footprint = np.matrix([[1,1,1],[1,2,1],[1,1,1]])

        mappa = ndimage.gaussian_filter(mappa,sigma=sigma)
        mappa = ndimage.grey_dilation(mappa,footprint=footprint)
        mappa = ndimage.grey_erosion(mappa,footprint=footprint)
        _,ax= plt.subplots(figsize=(15*self.__dy_roi/(self.__dx_roi+self.__dy_roi),15*self.__dx_roi/(self.__dx_roi+self.__dy_roi)))
        ax.imshow(mappa,cmap = 'inferno',alpha=0.5)
        CS = ax.contour(np.arange(self.__dy_roi),np.arange(self.__dx_roi),mappa,levels,cmap = 'inferno')
        ax.clabel(CS, inline=True, fontsize=10)
        plt.show()
   
    def view(self,t_lim_inf = None,t_lim_sup = None): 
        ''' Visualizza finestra corrente
            t_lim_inf --> limite inf color bar
            t_lim_sup --> limite sup color bar
        '''
        t_lim_inf_temp,t_lim_sup_temp = set_clim(self.__video_roi_offset)
        if t_lim_sup is None:
            t_lim_sup = t_lim_sup_temp
        if t_lim_inf is None:
            t_lim_inf = t_lim_inf_temp
        _,ax = plt.subplots()
        temp = ax.imshow(self.video[:,:,1],cmap = 'inferno',clim = [t_lim_inf,t_lim_sup])
        plt.colorbar(temp,orientation = 'vertical',fraction = 0.5)
        plt.show()

    def view_animate(self,ROI=True):
        fig = plt.figure()
        ims = []
        temp = []
        if ROI and self.__tag_roi:
            for i in range(self.__N_roi):
                temp = plt.imshow(self.__video_roi[:,:,i]+self.__video_roi_offset,cmap = 'inferno', animated=True)
                ims.append([temp])
        else:
            for i in range(self.__N):
                temp = plt.imshow(self.video[:,:,i],cmap = 'inferno', animated=True)
                ims.append([temp])
        animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=100)
        plt.show()

    def result_line(self,xi:float,yi:float,xf:float,yf:float,view:bool= True):
        '''
        '''
        dx = (xf-xi)
        dy = (yf-yi)
        cordinate = []
        if dx != 0 and dy != 0:
            N = int(dx*dy)
        elif dx != 0:
            N = int(dx)
        elif dy != 0:
            N = int(dy)
        u = np.arange(N)/N
        for i in range(N):
            x_v = xi + dx*u[i]
            y_v = yi + dy*u[i]
            if not([int(x_v),int(y_v)] in cordinate):
                cordinate.append([int(x_v),int(y_v)])
        value_line = []
        for i in cordinate:
            value_line.append(self.__map_amplitude[i[0],i[1]])
        if view:
            t_lim_inf,t_lim_sup = set_clim(self.__map_amplitude)
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(self.__map_amplitude,cmap = 'inferno',clim = [t_lim_inf,t_lim_sup],interpolation='hermite')
            ax[0].plot(yi + dy*u,xi + dx*u,c='w')
            ax[1].plot(value_line)
            ax[1].plot(np.mean(value_line)*np.ones(len(cordinate)))
            ax[1].grid()
            plt.show()
        return value_line     

    def get_freq(self):
        return self.__fr

    def get_result(self):
        return self.__map_amplitude,self.__map_phase

    def get_window(self):
        return self.video

    def get_roi(self):
        return self.__video_roi[:,:,1]

    def get_roi_offset(self):
        return self.__video_roi_offset
        
##---------------------------------------------------------------------------------------------

def __update(cmap_lim_inf,cmap_lim_sup,AxesImage,fig):
    lim_inf = cmap_lim_inf.val
    lim_sup = cmap_lim_sup.val
    if lim_inf<lim_sup:
        AxesImage.set_clim([lim_inf,lim_sup])
    fig.canvas.draw_idle()

def __update_phase(slider_lim,AxesImage,fig,map_phase,mask):
    th_offset = slider_lim.val
    (dx,dy) = map_phase.shape
    map_phase_temp = map_phase.copy()
    for x in range(dx):
        for y in range(dy):
            if mask[x,y]:
                map_phase_temp[x,y] = abs(map_phase_temp[x,y] + th_offset)%180 *(-1)*np.sign(map_phase_temp[x,y] + th_offset)
    AxesImage.imshow(map_phase_temp,cmap='twilight',clim = [-180,180])
    fig.canvas.draw_idle()

def interactive_phase(matrice_immagine,mask=None,titolo='Immagine'):
    '''
    input:
        - image --> npy matrix
        - mask  --> zeros and ones matrix, if 1 change the phase if 0 change the 
    output:
        - (x1,y1,x2,y2) cordinate
    '''
    if mask is None:
        mask = np.ones(matrice_immagine.shape)
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set(title=titolo)
    temp = ax.imshow(matrice_immagine,cmap='twilight',clim = [-180,180])
    plt.colorbar(temp,orientation = 'vertical',fraction = 0.5)
    ax_slider = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor='lightgoldenrodyellow')
    slider_lim = Slider(ax_slider, 'phase', -180, 180,orientation="vertical")
    slider_lim.on_changed(lambda temp: __update_phase(slider_lim,ax,fig,matrice_immagine,mask))
    plt.show()
    return  slider_lim.val

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
    if n_max.size == 0:
        return n_max
    else:
        return n_max[value_max>(k_scale*signal[n_max[-1]])]

def split_video(analysis_object,xi,yi,xf,yf,view=False,save_path = 'data'):
    ''' Permette di selezionare una regione d'interesse rettangolare su cui svolgere l'analisi
    Input:
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
        print('invalid')

@jit(nopython=True)
def lockin_2D(video_roi,video_roi_mask,fr,fs,phase_offset=0):
    ''' 
    Multiple-channel digital lock-in amplifier
        - signal --> .npy array 3D
        - fr     --> reference frequency [Hz]
        - fs     --> Sampling rate [Hz]
        - phase_offset --> phase of the reference signal [rad]
    return;
        - ampitude
        - phase [deg]
    ''' 
    (dx_roi,dy_roi,N_roi) = video_roi.shape
    wr = (2*np.pi)*fr
    Ix = np.zeros((dx_roi,dy_roi))
    Iy = np.zeros((dx_roi,dy_roi))
    t = (1/fs)*np.arange(N_roi)
    SgnF = np.sin(wr*t+phase_offset)
    SgnG = np.cos(wr*t+phase_offset)
    for x in range(dx_roi):
        for y in range(dy_roi):
            if video_roi_mask[x,y]:
                Ix[x,y] = np.mean(video_roi[x,y,:]*SgnF) 
                Iy[x,y] = np.mean(video_roi[x,y,:]*SgnG)
            else:
                Ix[x,y] = 0
                Iy[x,y] = 0
    map_amplitude = 2*np.sqrt(Ix**2 + Iy**2)
    map_phase = np.arctan2(Iy,Ix)
    return map_amplitude,map_phase

@jit(nopython=True)
def lockin_1D(signal,fr,fs,phase_offset:float=0):
    ''' 
    A basic  Multiple-channel digital lock-in amplifier.
    input:
        - signal --> .npy array 1d
        - fr     --> reference frequency
        - fs     --> Sampling rate
        - phase_offset --> phase of the reference signal
    return;
        - ampitude
        - phase [deg]
    ''' 
    time = (1/fs)*np.arange(len(signal))
    wr = fr*(2*np.pi)
    SgnSF = np.sin(wr*time+phase_offset)*signal
    SgnSG = np.cos(wr*time+phase_offset)*signal
    sf = np.abs(np.mean(SgnSF)*2)
    sg = np.abs(np.mean(SgnSG)*2)
    return np.sqrt(sf**2+sg**2),np.arctan2(sg,sf)

def set_clim(matrix,p_sup = 0.96,p_inf = 0.1):
    ''' set the cmap limit for the matrix from the percentile
    Input:
        - p_sup -->
        - p_inf -->
    Output:
        - p 
        - 
    '''
    (dx,dy) = matrix.shape
    N = dx*dy
    istogramma = np.zeros(N)
    istogramma = np.sort(np.reshape(matrix,N))
    return istogramma[int(p_inf*N)],istogramma[int(p_sup*N)]


def main():
    pass

if __name__ == '__main__':
    main()



