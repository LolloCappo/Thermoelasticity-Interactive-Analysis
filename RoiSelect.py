import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector,EllipseSelector,Slider

cordinate = (0,0,0,0)


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
    _,ax = plt.subplots()
    ax.imshow(matrice_immagine,cmap = 'inferno')
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
    _,ax = plt.subplots()
    ax.imshow(matrice_immagine)
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
    _,ax = plt.subplots()
    ax.imshow(matrice_immagine)
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



if __name__ == '__main__':
    print('Modulo per roi interattiva')

