import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib as mpl
import matplotlib.cm as cm
from skimage import img_as_ubyte

nii = nib.load('/home/maryana/storage2/Posdoc/AVID/AV23/blockface/nii/A13-1181-002-BlockFace-CorrectHeader.nii.gz')
#nii = nib.load('/home/maryana/storage2/Posdoc/AVID/AV13/blockface/nii/av13_blockface.nii')

vol = nii.get_data()
vol2 = np.zeros(vol.shape)

#AV1 AT100
#slices = [280,296,312,328,344,360,376,392,408,424,440,457,472,488,504,520,536,552,568,584,600,616,632,648]

#AV1 AT8
# slices = [162,202,242,282,286,290,295,298,306,310,311,314,318,322,326,327,330,334,338,342,343,346,350,354,358,362,366,
#           370,374,378,382,386,390,394,398,402,406,410,414,418,422,430,434,438,442,448,450,454,458,462,466,470,474,478,
#           482,486,490,494,498,502,506,510,514,518,522,526,530,534,538,542,546,550,554,558,562,566,570,574,578,582,586,
#           590,594,598,602,606,610,614,618,622,626,630,634,638,642,650,654,658,702]

#AV2 AT100
#slices = [84,92,98,100,116,124,132,140,148,156,164,172,180,188,196,204,212,220,228,236,244,252,260,268,276,284,292,300,
#          308,316,324,332,340,348,356,364,372,380,396,404,412,420,428,436,444,452,468,476,484,492,500,508,516,532,540,
#          548,556,564,580,588,596,612,620,628,636,652,660,700,708,724,732,737,740,745,748,753,756]
#AV2_AT8
slices = [71,75,79,83,87,91,95,99,103,107,111,115,119,123,127,131,135,139,143,147,151,155,159,163,167,171,175,179,183,
          187,191,195,199,203,207,211,215,219,223,227,231,235,243,247,251,255,259,263,267,271,275,279,283,287,291,295,
          299,303,307,311,315,319,323,327,331,335,339,343,347,351,355,359,363,367,371,375,379,383,387,391,395,399,403,
          407,411,415,419,423,427,431,435,439,443,447,451,455,459,463,467,471,475,479,483,487,491,495,499,503,507,511,
          515,519,523,527,531,535,539,543,547,551,555,559,563,567,571,575,579,583,587,591,595,599,603,607,611,615,619,
          623,627,631,635,639,643,647,651,655,659,663,667,671,675,679,683,687,691,695,699,703,707,711,715,719,723,727,
          731,735,739,743,747,751,755]

#slice_name = '/home/maryana/storage2/Posdoc/AVID/AV13/AT8/registered/AV1AT8_{}_heatmap.nii'
slice_name = '/home/maryana/storage2/Posdoc/AVID/AV23/AT8/registered/registered_AV2AT8_{}_heatmap.nii'


nSlices = len(slices)

for s in range(nSlices):
    id = slices[s]
    if s == nSlices-1:
        id2 = slices[s]
    else:
        id2 = slices[s+1]

    try:
        name = slice_name.format(str(id),str(id))
        slice = nib.load(name)
        simg = slice.get_data()
        # norm = mpl.colors.Normalize(vmin=simg.min(),vmax=simg.max())
        # cmap = cm.gray
        # img = cmap(norm(simg))
        # img2 = img_as_ubyte(img)
        # hmap = img2[:, :, 0]    #
        # vol2[:,:,id-1]=hmap

        # for ss in range(id-1,id2-1):
        #     vol2[:,:,ss] = simg
        vol2[:,:,id-1] = simg
    except Exception as e:
        print("Error loading slice {}".format(id))
        print(e)


#hm_name='/home/maryana/storage2/Posdoc/AVID/AV23/blockface/nii/AT8_heatmap2blockface_072019_AAIC.nii'
hm_name='/home/maryana/storage2/Posdoc/AVID/AV23/blockface/nii/AV2_AT8_heatmap2blockface_051421.nii'
nii2 = nib.Nifti1Image(vol2,nii.affine)
nib.save(nii2,hm_name)
print('Files {} sucessfully saved.'.format(hm_name))