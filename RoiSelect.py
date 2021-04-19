import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import RectangleSelector,Button,EllipseSelector,Slider,Cursor

cordinate = (0,0,0,0)
cordinate_punto = [0,0]


def __onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    #print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    global cordinate
    cordinate = (int(x1),int(y1),int(x2),int(y2))

def __toggle_selector(event):
    print(' Key pressed. q to exit, a activeted/ w deactived select')
    if event.key in ['A', 'a'] and __toggle_selector.RS.active:
        print(' Selector deactivated.')
        __toggle_selector.RS.set_active(False)
    if event.key in ['W', 'w'] and not __toggle_selector.RS.active:
        print(' Selector activated.')
        __toggle_selector.RS.set_active(True)

def __update(cmap_lim_inf,cmap_lim_sup,AxesImage,fig):
    lim_inf = cmap_lim_inf.val
    lim_sup = cmap_lim_sup.val
    if lim_inf<lim_sup:
        AxesImage.set_clim([lim_inf,lim_sup])
    fig.canvas.draw_idle()

def selectROI(matrice_immagine,titolo='Immagine'):
    ''' 
    input:
        ax --> axis su cui fare la selezione della roi
    output:
        (xi,yi,dx,dy)
    '''
    (t_lim_inf,t_lim_sup)=set_cmap(matrice_immagine)

    _,ax = plt.subplots()
    ax.imshow(matrice_immagine,cmap = 'inferno',clim = [t_lim_inf,t_lim_sup])
    ax.set(title=titolo)
    __toggle_selector.RS = RectangleSelector(ax, __onselect,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
    plt.connect('key_press_event', __toggle_selector)
    plt.show()
    return cordinate


def select_line(matrice_immagine,titolo='Immagine'):
    ''' 
    input:
        ax --> axis su cui fare la selezione della roi
    output:
        (xi,yi,dx,dy)
    '''
    (t_lim_inf,t_lim_sup)=set_cmap(matrice_immagine)

    _,ax = plt.subplots()
    ax.imshow(matrice_immagine,clim = [t_lim_inf,t_lim_sup])
    ax.set(title=titolo)

    __toggle_selector.RS = RectangleSelector(ax, __onselect,
                                       drawtype='line', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
    plt.connect('key_press_event', __toggle_selector)
    plt.show()
    return cordinate

def selectROI_ellipse(matrice_immagine,titolo='Immagine'):
    ''' 
    input:
        ax --> axis su cui fare la selezione della roi
    output:
        (x1,y1,x2,y2)
    '''
    (t_lim_inf,t_lim_sup)=set_cmap(matrice_immagine)
    _,ax = plt.subplots()
    ax.imshow(matrice_immagine,clim = [t_lim_inf,t_lim_sup])
    ax.set(title=titolo)
    __toggle_selector.ES = EllipseSelector(ax, __onselect,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
    plt.connect('key_press_event', __toggle_selector)
    plt.show()
    return cordinate


def interactive_cmap(matrice_immagine,titolo='Immagine'):
    ''' 
    input:
        matrice_immagine
    output:
        (x1,y1,x2,y2)
    '''
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.set(title=titolo)
    lim_inf = 1.1*np.min(matrice_immagine)
    lim_sup = 0.9*np.max(matrice_immagine)
    AxesImage = plt.imshow(matrice_immagine,cmap='magma')
    fig.colorbar(AxesImage,orientation = 'vertical',fraction = 0.5)
    

    axcolor = 'lightgoldenrodyellow'
    ax_lim_inf = plt.axes([0.1, 0.25, 0.0225, 0.63], facecolor=axcolor)
    ax_lim_sup = plt.axes([0.2, 0.25, 0.0225, 0.63], facecolor=axcolor)

    cmap_lim_inf = Slider(ax_lim_inf, 'sunf', lim_inf/1.1, lim_sup/0.9, valinit=lim_inf,orientation="vertical")
    cmap_lim_sup = Slider(ax_lim_sup, 'sup', lim_inf/1.1, lim_sup/0.9, valinit=lim_sup,orientation="vertical")
    cmap_lim_inf.on_changed(lambda temp: __update(cmap_lim_inf,cmap_lim_sup,AxesImage,fig))
    cmap_lim_sup.on_changed(lambda temp: __update(cmap_lim_inf,cmap_lim_sup,AxesImage,fig))
    plt.show()
    return (cmap_lim_inf,cmap_lim_sup)

def __onclick(event):
    "onclick are matplotlib events at press and release."
    x, y = event.xdata, event.ydata
    global cordinate_punto
    cordinate_punto = [int(x),int(y)]


def selectROI_point(fig,ax,titolo='Immagine'):
    ''' 
    input:
        image matrix
    output:
        (x1,y1,x2,y2)
    '''
    ax.set(title=titolo)    
    Cursor(ax,horizOn=True,vertOn=True, color='red', linewidth=2)
    fig.canvas.mpl_connect('button_press_event',__onclick)

    plt.show()
    return cordinate_punto

def set_cmap(image,p_sup = 0.96,p_inf = 0.4):
    p_sup = 0.96 # [%] dei valori limite sup
    p_inf = 0.4
    N = image.size
    istogramma = np.zeros(N)
    istogramma = np.sort(np.reshape(image,N))
    t_lim_sup = istogramma[int(p_sup*N)]
    t_lim_inf = istogramma[int(p_inf*N)]
    return (t_lim_inf,t_lim_sup)


def list_all_files(path_dir,ext='.npy'):
    list_file = []
    for file in os.listdir(path_dir):
        if file.endswith(ext):
            list_file.append(os.path.join(path_dir, file))
    return list_file

if __name__ == '__main__':
    print('Modulo per roi interattiva')

