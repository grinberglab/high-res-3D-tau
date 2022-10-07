'''
Created on May 24, 2016

@author: Maryana Alegro
'''

from matplotlib.widgets import Lasso
from matplotlib.colors import colorConverter
from matplotlib.collections import RegularPolyCollection
from matplotlib import path

import matplotlib.pyplot as plt
import numpy as np
from numpy import nonzero
from numpy.random import rand

class LassoManager(object):
    def __init__(self, ax, data):
        self.axes = ax
        self.canvas = ax.figure.canvas
        self.data = data
        self.ind = [];
        #fig = ax.figure

        self.cid = self.canvas.mpl_connect('button_press_event', self.onpress)

    def callback(self, verts):

        p = path.Path(verts)
        self.ind = p.contains_points(self.data)
        #self.ind = nonzero([p.contains_point(xy) for xy in self.data])[0]
        
        self.canvas.draw_idle()
        self.canvas.widgetlock.release(self.lasso)
        del self.lasso


    def onpress(self, event):
        if self.canvas.widgetlock.locked():
            return
        if event.inaxes is None:
            return
        self.lasso = Lasso(event.inaxes,
                           (event.xdata, event.ydata),
                           self.callback,useblit=False)
        # acquire a lock on the widget drawing
        self.canvas.widgetlock(self.lasso)
        
        
    def getSelected(self):
        return self.ind;