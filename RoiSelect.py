import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector,EllipseSelector,Slider
import pandas as pd
cordinate = (0,0,0,0)


def __onselect(eclick, erelease):
    "eclick and erelease are matplotlib events at press and release."
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))

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

if __name__ == '__main__':
    print('Modulo per roi interattiva')

