#!/usr/bin/python3 -s

FNAME='figure%d.png'


import pylab
import matplotlib.pyplot as plt

_show=pylab.show

def newshow(*l, **k):
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(FNAME % i)
        print("saved figure %d as '%s'." % (i, FNAME % i))
    _show()		# uncomment this line if you want to display the plot on the screen as well
pylab.show = newshow

# from plot_gff import *
exec(open("/bin/plot_gff.py").read())
