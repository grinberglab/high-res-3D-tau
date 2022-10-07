'''
Import images from memory card
==============================
'''

import os
import shutil
import pandas as pd


csv_file = '/media/maryana/PENPEN/csv_pli/optical_chiasm_2370.10.csv'
img_dir = '/home/maryana/storage/Posdoc/OPT_CHIASM_CANON_PLI'
out_dir = '/home/maryana/storage/Posdoc/PLI/2017/2372.10_Optic_Chiasm/PLI/raw'

#load CSV
file_list = pd.read_csv(csv_file)
slice_id = file_list.iloc[:,0].tolist()
slice_1st = file_list.iloc[:,1].tolist()
slice_last = file_list.iloc[:,2].tolist()
nSlices = len(slice_id)

#create slice map
file_map = {}
for s in range(nSlices):
    sid = slice_id[s]
    #check for duplicates, they may be mislabels
    if sid in file_map:
        print "Warning slice {} may be duplicated or mislabeled".format(sid)
        print "Skipping"
        continue
    sbegin = slice_1st[s]
    send = slice_last[s]
    file_map[sid] = (sbegin,send)

img_name = 'IMG_{:04d}.{}'
file_name = 'slice_{}_{:02d}.{}'
file_ext = 'CR2'

for item in file_map:
    begin = file_map[item][0]
    end = file_map[item][1]
    counter = 0

    #check if the number of images per slice is correct
    if (end - begin) != 17:
        print "Warning: Wrong number of images for file {}".format(file_name)
        print "Skipping"
        continue

    for ind in range(begin,end+1):
        old_name = img_name.format(ind,file_ext)
        T = type(item)
        if T == int:
            item_str = "{:04d}".format(item)
        else:
            item_str = item
        new_name = file_name.format(item_str,counter,file_ext)
        file1 = os.path.join(img_dir,old_name)
        file2 = os.path.join(out_dir,new_name)
        #print('Copying {} to {}'.format(old_name,new_name))
        print('Copying {} to {}'.format(file1,file2))
        counter += 1
        #copy file
        shutil.copyfile(file1,file2)
        # try:
        #     shutil.copyfile(file1,file2)
        # except:
        #     print('Error copying {}.'.format(file2))










