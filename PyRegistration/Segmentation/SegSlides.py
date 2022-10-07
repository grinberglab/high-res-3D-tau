'''
@author: Maryana Alegro

Segments slides using the LAB distance trick (deltaE)
'''

from matplotlib import pyplot as plt
from skimage import io
import numpy as np
from skimage import color
from skimage import img_as_float, img_as_ubyte
from skimage import morphology as morph
from skimage import filters as filters
from skimage import measure as meas
from numpy import nonzero
from skimage.measure import find_contours
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import scipy.ndimage.morphology as morph2
from sklearn.cluster import KMeans
import NCut as ncut

class SegSlides(object):
    '''
    classdocs
    '''


    def __init__(self, sF=[], sB=[], idx_sF=[], idx_sB=[], ref_hist=np.array([])):
        self.sF = sF
        self.sB = sB
        self.idx_sF = idx_sF
        self.idx_sB = idx_sB
        self.ref_hist = ref_hist #ref_hist = [R_hist,G_hist,B_hist]

    '''
    Performs initial segmentation using the deltaE trick.
    '''
    def doLABSegmentation(self, img):
        imsize = img.shape[0:2]
        lab = color.rgb2lab(img)
        L = lab[...,0]
        A = lab[...,1]
        B = lab[...,2]
        
        mLf = self.sF[0]; mAf = self.sF[1]; mBf = self.sF[2]
        mLb = self.sB[0]; mAb = self.sB[1]; mBb = self.sB[2]
        
        #compute LAB delta E for foreground
        meanL = mLf*np.ones(imsize)
        meanA = mAf*np.ones(imsize)
        meanB = mBf*np.ones(imsize)
        dL = L - meanL
        dA = A - meanA
        dB = B - meanB
        dEf = np.sqrt(dL**2 + dA**2 + dB**2)
        dEf = (dEf - dEf.min())/(dEf.max() - dEf.min())
        
        #compute LAB delta E for background
        meanL = mLb*np.ones(imsize)
        meanA = mAb*np.ones(imsize)
        meanB = mBb*np.ones(imsize)
        dL = L - meanL
        dA = A - meanA
        dB = B - meanB
        dEb = np.sqrt(dL**2 + dA**2 + dB**2)
        dEb = (dEb - dEb.min())/(dEb.max() - dEb.min())

        levelF = filters.threshold_otsu(img_as_ubyte(dEf))
        maskF = img_as_ubyte(dEf) >= levelF
        levelB = filters.threshold_otsu(img_as_ubyte(dEb))
        maskB = img_as_ubyte(dEb) >= levelB
        
        maskF[maskB == True] = False
        
        se = morph.disk(4)
        maskF = morph.binary_closing(morph.binary_opening(maskF, se))
        se2 = morph.disk(6)
        maskF = morph.binary_closing(maskF, se2)

        labels = meas.label(maskF, connectivity=maskF.ndim)
        props = meas.regionprops(labels)
        nL = len(props)

        #remove very small areas
        # areas = np.array([props[a].area for a in range(nL)])
        # perc = np.percentile(areas,25)
        # toDel = nonzero(areas <= perc)[0]
        # ntD = len(toDel)
        # for td in range(ntD):
        #     l = toDel[td]
        #     idx = nonzero(labels == l+1)
        #     maskF[idx] = 0

        centerC = imsize[1]/2
        centerR = imsize[0]/2
        #create a list of tuples
        closest = []
        for l in range(0,nL):
            center = props[l].centroid #(row,col)
            R = center[0]
            C = center[1]
            d = np.sqrt((centerC-C)**2 + (centerR-R)**2)
            nPix = len(nonzero(labels == l+1)[0])
            
            closest.append((l+1,d,nPix,(R,C)))
  
        closest2 = sorted(closest, key=lambda x: (x[1], -x[2]))
        label = 1
        int_ctr = []
        int_dist = []
        for c in closest2:
            ratio = c[2]/float(imsize[0]*imsize[1])
            if ratio > 0.01:
                label = c[0]
                int_ctr = c[3] #(r,c)
                int_dist = c[1]
                break

        img_dic = {'img': img,
                   'L': L,
                   'A': A,
                   'B': B,
                   'mask': maskF,
                   'img_center': (centerR,centerC),
                   'obj_center': int_ctr,
                   'dist':int_dist}
        # np.save('struct2.npy', struct)
        #
        # mask = np.zeros(imsize)
        # mask[labels == label] = 1
        #
        # img2 = img.copy()
        # img2[mask == False] = 0
        
        return img_dic

    def run_Ncut(self, img_dic):
        return ncut.run_ncut(img_dic)

    '''
    Refines segmentation using snakes
    '''
    def doSnakesSegmentation(self,img,mask):

        if img.ndim > 1:
            img = color.rgb2gray(img)

        labels = meas.label(mask)
        props = meas.regionprops(labels)
        conv_hull = props[0].convex_image # there should be only one object
        bbox = props[0].bbox
        mask_tmp = np.zeros(mask.shape,dtype=bool)
        mask_tmp[bbox[0]:bbox[2],bbox[1]:bbox[3]] = conv_hull # copy convex hull to full size image
        se = np.ones([6, 6])
        mask_tmp = morph2.binary_dilation(mask_tmp,se)
        mask_tmp = mask_tmp * 255 # convert to uint
        contour = find_contours(mask_tmp, 1)  # arranges the coordinates in CCW order
        init = np.array([contour[0][:, 1], contour[0][:, 0]]).T
        img2 = gaussian(img, 3)
        snake = active_contour(img2, init, alpha=0.010, beta=8, gamma=0.001)
        perim = np.round(snake).astype(int)
        mask2 = np.zeros(mask.shape, dtype=bool)
        mask2[perim[:, 1], perim[:, 0]] = True
        #se = morph2.generate_binary_structure(2, 4)
        se = np.ones([4,4])
        mask2 = morph2.binary_dilation(mask2,se)
        mask2 = morph2.binary_fill_holes(mask2)

        img3 = img.copy()
        img3[mask2 == False] = 0
        return mask2,img3


    '''
        Imposes a reference histogram on an image
        :param self:
        :param img: image to be enhanced
        :param args: cant be the reference image or its histogram
        :return: enhanced image
    '''
    def shape_histogram(self, img, ref_hist):

        nTotalPix = img.shape[0] * img.shape[1]
        linImg = img.reshape([nTotalPix])
        sort_idx = np.argsort(linImg)

        newImg = -1 * np.ones(linImg.shape)  # empty image vector
        maxVal = ref_hist.shape[0]  # max gray value
        currPos = 0
        for currBin in range(0, maxVal):
            nPixInBin = ref_hist[currBin]  # reference histogram
            for p in range(0, nPixInBin):
                if currPos > nTotalPix-1:
                    print ('Warning: Index larger than num. image pixels')
                    break
                origIdx = sort_idx[currPos]
                newImg[origIdx] = currBin  # final image receives histogram bin value
                currPos += 1

        newImg[newImg < 0] = maxVal
        final_img = newImg.reshape(img.shape)
        return final_img

    def impose_histogram(self,img):
        if img.ndim > 1:
            c1 = img[...,0]
            c2 = img[...,1]
            c3 = img[...,2]
            c1f = self.shape_histogram(c1, self.ref_hist[:, 0])
            c2f = self.shape_histogram(c2, self.ref_hist[:, 1])
            c3f = self.shape_histogram(c3, self.ref_hist[:, 2])
            c1f = c1f.astype('uint8')
            c2f = c2f.astype('uint8')
            c3f = c3f.astype('uint8')
            s = c1.shape
            img2 = np.concatenate((c1f.reshape([s[0], s[1], 1]), c2f.reshape([s[0], s[1], 1]), c3f.reshape([s[0], s[1], 1])), axis=2)
        else:
            img2 = self.shape_histogram(img, self.ref_hist)
            img2 = img2.astype('uint8')

        return img2

    '''
        Performs all segmentation steps
    '''
    def doSegmentation(self, img, run_ncut=True):
        img_orig = img.copy()
        #eh_img = self.impose_histogram(img)
        # run deltaE seg
        #mask,img = self.doLABSegmentation(img)
        #mask,img = self.doKMeansSegmentation(img)
        img_dic = self.doLABSegmentation(img)

        if run_ncut:
            img,mask = self.run_Ncut(img_dic)
        else:
            img = img_dic['img']
            mask = img_dic['mask']
            img[mask == 0] = 0

        # run snakes refinement
        #mask,img = self.doSnakesSegmentation(img,mask)
        #img_orig[mask == 0] = 0

        return mask,img,img_orig

    def doKMeansSegmentation(self,img):

        lab = color.rgb2lab(img)
        back = lab[self.idx_sB[1]:self.idx_sB[3], self.idx_sB[0]:self.idx_sB[2]]
        fore = lab[self.idx_sF[1]:self.idx_sF[3], self.idx_sF[0]:self.idx_sF[2]]
        mLf = np.mean(np.ravel(fore[..., 0]))
        mAf = np.mean(np.ravel(fore[..., 1]))
        mBf = np.mean(np.ravel(fore[..., 2]))
        mLb = np.mean(np.ravel(back[..., 0]))
        mAb = np.mean(np.ravel(back[..., 1]))
        mBb = np.mean(np.ravel(back[..., 2]))

        nPix = lab.shape[0] * lab.shape[1]
        L = lab[..., 0]; A = lab[..., 1]; B = lab[..., 2]

        data = np.concatenate((L.reshape([nPix, 1]), A.reshape([nPix, 1]), B.reshape([nPix, 1])), axis=1)
        init = np.array([[mLf, mAf, mBf], [mLb, mAb, mBb]])
        kmeans = KMeans(n_clusters=2, random_state=0, init=init).fit(data)
        clusters = kmeans.predict(data)
        mask = clusters.reshape([lab.shape[0], lab.shape[1]])
        img[mask == 0] = 0

        return mask,img
