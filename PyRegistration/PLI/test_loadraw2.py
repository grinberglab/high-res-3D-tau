import rawpy
import numpy as np
import matplotlib.pyplot as plt


filename = '/Volumes/SUSHI_HD2/SUSHI/Posdoc/PLI/2017/P2694/block-4/slice_0010_0006.CR2'

raw = rawpy.imread(filename)
rgb = raw.postprocess(output_bps=16)
plt.imshow(rgb)