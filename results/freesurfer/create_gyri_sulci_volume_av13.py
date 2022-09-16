import nibabel.freesurfer as fs
import nibabel as nib
import numpy as np


names = ['gyri','sulci']
ctab = np.zeros((2,4),dtype = 'int')
ctab[0,:] = [20,10,100,255]
ctab[1,:] = [200,10,20,255]

#AV1

#left hemisphere
lh_curv_file = '/home/maryana/bin/freesurfer/subjects/av13_pm/surf/lh.curv'
lh_annot_file = '/home/maryana/bin/freesurfer/subjects/av13_pm/label/lh_sulci_gyri.annot'

lh_curv = fs.read_morph_data(lh_curv_file)
lh_sulci = np.zeros(lh_curv.shape,dtype='int')
lh_sulci[lh_curv <= 0] = 0
lh_sulci[lh_curv > 0] = 1

fs.io.write_annot(lh_annot_file,lh_sulci,ctab,names)

#right hemisphere
rh_curv_file = '/home/maryana/bin/freesurfer/subjects/av13_pm/surf/rh.curv'
rh_annot_file = '/home/maryana/bin/freesurfer/subjects/av13_pm/label/rh_sulci_gyri.annot'

rh_curv = fs.read_morph_data(rh_curv_file)
rh_sulci = np.zeros(rh_curv.shape,dtype='int')
rh_sulci[rh_curv <= 0] = 0
rh_sulci[rh_curv > 0] = 1

fs.io.write_annot(rh_annot_file,rh_sulci,ctab,names)


# #create left hemisphere sulci and gyri volumes(use freesurfer command line  tools)
# mri_annotation2label --subject av13 --hemi lh --annotation av13/label/lh_sulci_gyri.annot --outdir av13/label
# mri_label2vol --label av13/label/lh.sulci.label --subject av13 --hemi lh --identity --temp av13/mri/T1.mgz --o av13/mri/lh.sulci.nii.gz --proj frac 0 1 0.01
# mri_binarize --dilate 1 --erode 1 --i av13/mri/lh.sulci.nii.gz --o av13/mri/lh.sulci2.nii.gz --min 1
# mris_calc -o av13/mri/lh.sulci_final.nii.gz av13/mri/lh.sulci2.nii.gz mul av13/mri/lh.ribbon.mgz
#
# mri_label2vol --label av13/label/lh.gyri.label --subject av13 --hemi lh --identity --temp av13/mri/T1.mgz --o av13/mri/lh.gyri.nii.gz --proj frac 0 1 0.01
# mri_binarize --dilate 1 --erode 1 --i av13/mri/lh.gyri.nii.gz --o av13/mri/lh.gyri2.nii.gz --min 1
# mris_calc -o av13/mri/lh.gyri_final.nii.gz av13/mri/lh.gyri2.nii.gz mul av13/mri/lh.ribbon.mgz
#
# #crete right hemisphere sulci and gyri volumes
# mri_annotation2label --subject av13 --hemi rh --annotation av13/label/rh_sulci_gyri.annot --outdir av13/label
# mri_label2vol --label av13/label/rh.sulci.label --subject av13 --hemi rh --identity --temp av13/mri/T1.mgz --o av13/mri/rh.sulci.nii.gz --proj frac 0 1 0.01
# mri_binarize --dilate 1 --erode 1 --i av13/mri/rh.sulci.nii.gz --o av13/mri/rh.sulci2.nii.gz --min 1
# mris_calc -o av13/mri/rh.sulci_final.nii.gz av13/mri/rh.sulci2.nii.gz mul av13/mri/rh.ribbon.mgz
#
# mri_label2vol --label av13/label/rh.gyri.label --subject av13 --hemi rh --identity --temp av13/mri/T1.mgz --o av13/mri/rh.gyri.nii.gz --proj frac 0 1 0.01
# mri_binarize --dilate 1 --erode 1 --i av13/mri/rh.gyri.nii.gz --o av13/mri/rh.gyri2.nii.gz --min 1
# mris_calc -o av13/mri/rh.gyri_final.nii.gz av13/mri/rh.gyri2.nii.gz mul av13/mri/rh.ribbon.mgz


#join lh and rh sulci/gyri volumes into a single file
lh_sulci_nii = nib.load('/home/maryana/bin/freesurfer/subjects/av13_pm/mri/lh.sulci_final.nii.gz')
lh_sulci = lh_sulci_nii.get_data()
lh_gyri_nii = nib.load('/home/maryana/bin/freesurfer/subjects/av13_pm/mri/lh.gyri_final.nii.gz')
lh_gyri = lh_gyri_nii.get_data()

rh_sulci_nii = nib.load('/home/maryana/bin/freesurfer/subjects/av13_pm/mri/rh.sulci_final.nii.gz')
rh_sulci = rh_sulci_nii.get_data()
rh_gyri_nii = nib.load('/home/maryana/bin/freesurfer/subjects/av13_pm/mri/rh.gyri_final.nii.gz')
rh_gyri = rh_gyri_nii.get_data()

vol = np.zeros(lh_sulci.shape)
vol[lh_sulci > 0] = 255
vol[rh_sulci > 0] = 255
vol[lh_gyri > 0] = 100
vol[rh_gyri > 0] = 100
sulci_gyri_nii = nib.Nifti1Image(vol,lh_sulci_nii.affine)
nib.save(sulci_gyri_nii,'/home/maryana/bin/freesurfer/subjects/av13_pm/mri/sulci_gyri_final.nii.gz')




#visualize annotations
# freeview -f  av13_pm/surf/lh.pial:annot=aparc.annot:name=pial_aparc:visible=0 \
# av13_pm/surf/lh.pial:annot=aparc.a2009s.annot:name=pial_aparc_des:visible=0 \
# av13_pm/surf/lh.inflated:overlay=lh.thickness:overlay_threshold=0.1,3::name=inflated_thickness:visible=0 \
# av13_pm/surf/lh.inflated:visible=0 \
# av13_pm/surf/lh.white:visible=0 \
# av13_pm/surf/lh.pial \
# --viewport 3d

#from: https://ggooo.wordpress.com/2014/10/12/extracting-a-volumetric-roi-from-an-annotation-file/
# #convert annotations to labels
# 'mri_annotation2label --subject 1001bl --hemi lh --annotation 1001bl/label/lh.TPJ.annot --outdir 1001bl/label'
#  #freeview -f 1001bl/surf/lh.white:label=1001bl/label/lh.temporal-parietal-junction-p.label
#
# #map the surface-based label to volume
# 'mri_label2vol --label 1001bl/label/lh.temporal-parietal-junction-p.label --subject 1001bl --hemi lh --identity --temp 1001bl/mri/T1.mgz --o 1001bl/mri/lh.TPJ.nii.gz'
# #freeview 1001bl/mri/T1.mgz 1001bl/mri/lh.TPJ.nii.gz -f 1001bl/surf/lh.white:label=1001bl/label/lh.temporal-parietal-junction-p.label -f 1001bl/surf/lh.pial:label=1001bl/label/lh.temporal-parietal-junction-p.label
#
# #enhance ROI
# 'mri_label2vol --label 1001bl/label/lh.temporal-parietal-junction-p.label --subject 1001bl --hemi lh --identity --temp 1001bl/mri/T1.mgz --o 1001bl/mri/lh.TPJc.nii.gz --proj frac 0 1 0.01'
# #freeview 1001bl/mri/T1.mgz 1001bl/mri/lh.TPJc.nii.gz -f 1001bl/surf/lh.white -f 1001bl/surf/lh.pial:label=1001bl/label/lh.temporal-parietal-junction-p.label
#
# 'mri_binarize --dilate 1 --erode 1 --i 1001bl/mri/lh.TPJc.nii.gz --o 1001bl/mri/lh.TPJf.nii.gz --min 1'
# #freeview 1001bl/mri/T1.mgz 1001bl/mri/lh.TPJf.nii.gz -f 1001bl/surf/lh.white -f 1001bl/surf/lh.pial:label=1001bl/label/lh.temporal-parietal-junction-p.label
#
# 'mris_calc -o 1001bl/mri/lh.TPJm.nii.gz 1001bl/mri/lh.TPJf.nii.gz mul 1001bl/mri/lh.ribbon.mgz'
# #freeview 1001bl/mri/T1.mgz 1001bl/mri/lh.TPJm.nii.gz -f 1001bl/surf/lh.white -f 1001bl/surf/lh.pial:label=1001bl/label/lh.temporal-parietal-junction-p.label



