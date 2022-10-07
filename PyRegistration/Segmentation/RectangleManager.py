'''
Created on May 25, 2016

@author: Maryana Alegro
'''

import matplotlib
from matplotlib.widgets import RectangleSelector
import numpy as np
import matplotlib.pyplot as plt

class RectangleManager(object):
    '''
    classdoc
    '''


    def __init__(self, ax):

        self.ax = ax
        self.RS = RectangleSelector(self.ax, self.line_select_callback,
                                       drawtype='box', useblit=False,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels')
     #                                 interactive=True)
                                       
        self.selectBack = []
        self.selectFore = []
        self.x1 = []
        self.x2 = []
        self.y1 = []
        self.y2 = []
        
    def line_select_callback(self, eclick, erelease):
        self.x1, self.y1 = eclick.xdata, eclick.ydata
        self.x2, self.y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (self.x1, self.y1, self.x2, self.y2))

        if eclick.button == 1: # left buttom = foreground
            self.selectFore = (self.x1, self.y1, self.x2, self.y2)
        elif eclick.button == 3: # right button = background
            self.selectBack = (self.x1, self.y1, self.x2, self.y2)

        print(" The button you used was: %s %s" % (eclick.button, erelease.button))

        
    def toggle_selector(self,event):
        print(' Key pressed.')
        if event.key in ['B', 'b'] and self.RS.active:
            print('Background region set.')
            self.selectBack = (self.x1,self.y1,self.x2,self.y2)
        if event.key in ['R', 'r'] and self.RS.active:
            print('Foreground region set.')
            self.selectFore = (self.x1,self.y1,self.x2,self.y2)

    '''
    Returns the top left and bottom right points selected regions
    '''
    def getSelection(self):
        return self.selectBack, self.selectFore