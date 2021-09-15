import numpy as np
import nibabel as nib
import skimage.io as io
import sys


def load_file(file_name):
    ext = file_name[:-3]
    if ext == 'nii':
        nii = nib.load(file_name)
        img = nii.get_data()
    else:
        img = io.imread(file_name)

    return img

def create_volume(name_str,ind1,ind2,dims,output_file):
    first = load_file(name_str.format(ind1))
    vol = np.zeros((first[0],first[1],(ind2-ind1)))
    for idx in range(ind1,ind2+1):
        file_name = name_str.format(idx)
        img = load_file(file_name)
        vol[:,:,idx] = img

    M = np.array([[dims[0], 0, 0, 0], [0, dims[1], 0, 0], [0, 0, dims[2], 0], [0, 0, 0, 1]])
    nii = nib.Nifti1Image(vol, M)
    nib.save(nii,output_file)


if __name__ == '__main__':
    if len(sys.argv) != 8:
        print('Usage: create_3d_volume <file_name_pattern> <1st_index> <last_index> <x_dim> <y_dim> <z_dim> <output_file>')
        print('Example: ')
        exit()


