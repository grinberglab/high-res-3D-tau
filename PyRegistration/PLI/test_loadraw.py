import numpy as np
from rawkit.raw import Raw
from rawkit.metadata import Metadata
import matplotlib.pyplot as plt
from rawkit.options import WhiteBalance, colorspaces, gamma_curves

filename = '/Volumes/SUSHI_HD2/SUSHI/Posdoc/PLI/2017/P2694/block-4/slice_0010_0006.CR2'
raw_image = Raw(filename)
info = raw_image.metadata
rows = info.height
cols = info.width
buff_img = np.array(raw_image.to_buffer())
buff_img = buff_img.reshape([rows,cols,3])
plt.imshow(buff_img)

