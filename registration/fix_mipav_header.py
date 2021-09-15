import matplotlib.pyplot as plt
import nibabel as nib
import sys



def fix_header(block_file,reg_file,out_file):
    nii1 = nib.load(block_file) #load blockface nii
    nii2 = nib.load(reg_file) #load file saved by MIPAV
    img1 = nii1.get_data() #get blockface image matrix
    img2 = nii2.get_data() #get histology image matrix
    #plt.imshow(img2) #display image
    #plt.imshow(img1) #display image
    M = nii1.affine #get affine matrix from blockface
    nii3 = nib.Nifti1Image(img2,M) #create new nifti file with histology image and blockface affine matrix
    nib.save(nii3,out_file)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: fix_mipav_header.py <blockface> <reg_img> <out_file>')
        exit()

    block_file = sys.argv[1]
    reg_file = sys.argv[2]
    out_file = sys.argv[3]
    fix_header(block_file,reg_file,out_file)





