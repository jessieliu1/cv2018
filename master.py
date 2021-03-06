import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
from skimage import color
from skimage.transform import rescale, resize
import scipy as scp
from skimage.segmentation import active_contour
from scipy import signal
#import matlab
#import matlab.engine
from irls import irls
from nearest_neighbor import nearest_n

import numpy.matlib as npm

from skimage.exposure import cumulative_distribution
import os

def write_output_img(filename, img):
    if not os.path.isdir("output"):
        os.mkdir("output")

    cv2.imwrite("output/" + filename, img)


#source: https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x
def cdf(im):
    im = im.astype(int)
    c, b = cumulative_distribution(im) 
    # pad the beginning and ending pixels and their CDF values
    c = np.insert(c, 0, [0]*b[0])
    c = np.append(c, [1]*(255-b[-1]))
    return c

def hist_matching(c, c_t, im):
    im = im.astype(int)
    pixels = np.arange(256)
    # find closest pixel-matches corresponding to the CDF of the input image, given the value of the CDF H of   
    # the template image at the corresponding pixels, s.t. c_t = H(pixels) <=> pixels = H-1(c_t)
    new_pixels = np.interp(c, c_t, pixels) 
    im = (np.reshape(new_pixels[im.ravel()], im.shape)).astype('uint8')
    return im



# add noise
def add_noise(image,sigma=50):
    row,col,ch= image.shape
    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.astype('uint8')
    gauss = gauss.reshape(row,col,ch)
    return (image + gauss).astype('uint8')

def imhistmatch(img1, img2):
    red_output = hist_matching(cdf(img1[:,:,0]),cdf(img2[:,:,0]),img1[:,:,0])
    green_output = hist_matching(cdf(img1[:,:,1]),cdf(img2[:,:,1]),img1[:,:,1])
    blue_output = hist_matching(cdf(img1[:,:,2]),cdf(img2[:,:,2]),img1[:,:,2])
    output = np.dstack((red_output,green_output,blue_output))
    return output


# output = imhistmatch(pikachu, van_gogh)
# plt.imshow(output)

# noisy_output = add_noise(output)
# plt.imshow(noisy_output)

# In[6]:

def gaussian2D(sigma=0.5):
    """
    2D gaussian filter
    """

    size = int(math.ceil(sigma * 6))
    if (size % 2 == 0):
        size += 1
    r, c = np.ogrid[-size / 2: size / 2 + 1, -size / 2: size / 2 + 1]
    g = np.exp(-(c * c + r * r) / (2. * sigma ** 2))
    g = g / (g.sum() + 0.000001)
    
    return g


# In[7]:

def laplace_of_gaussian(gray_img, sigma=1., kappa=0.75, pad=False):
    """
    Applies Laplacian of Gaussians to grayscale image.

    :param gray_img: image to apply LoG to
    :param sigma:    Gauss sigma of Gaussian applied to image, <= 0. for none
    :param kappa:    difference threshold as factor to mean of image values, <= 0 for none
    :param pad:      flag to pad output w/ zero border, keeping input image size
    """
    assert len(gray_img.shape) == 2
    img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape[:2]
    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows-1, 1:cols-1]
    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    # sign change at pixel?
    zero_cross = neg_min + pos_max
    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)
    return log_img


# In[8]:

# gray_van_gogh = color.rgb2gray(pikachu)
# output = segment(gray_van_gogh,1.05)
# plt.imshow(output, cmap='gray')

# In[9]:

def style_transfer(content, style, hall0, mask_temp, hallcoeff, Wcoeff, patch_sizes, scales, imsize):
    print("--------------Starting Style Transfer---------------")
    print("Content shape: ", content.shape)
    print("Mask shape: ", mask_temp.shape)

    output_shape = (imsize, imsize, 3)
    #content image now has imhist applied to it, has the colors of style
    C0 = imhistmatch(resize(content, output_shape),resize(style, output_shape))
    #gaussian noise function is added to the re-colored content image
    X_temp = add_noise(C0)

    h0 = imsize
    w0 = imsize
    gap_sizes = [28, 18, 9, 6]
    
    #iterate through resolutions, like a gaussian pyramid.
    for L in scales:
        print("----------------Resolution: ",L, "------------------")

        content_scaled = rescale(content, 1/L)
        style_scaled = rescale(style, 1/L)

        mask = np.copy(mask_temp)

        mask = rescale(mask, 1/L)

        print("Content scaled shape: ", content_scaled.shape)
        print("Mask scaled shape: ", mask.shape)

        C = np.copy(content_scaled)
        S = np.copy(style_scaled)
        h = math.ceil(h0/L)
        w = math.ceil(w0/L)

        X = np.copy(X_temp)
        #X = np.reshape(X, output_shape)

        X = rescale(X, 1/L)

        #X = x+

        hall = rescale(hall0,1/L)


        # print("content_scaled: ", content_scaled.shape)
        # print("style_scaled: ", style_scaled.shape)
        # print("mask: ", mask.shape)
        # print("C: ", C.shape)
        # print("S: ", S.shape)
        # print("h: ", h)
        # print("w: ", w)
        # print("X: ", X.shape)
        # print("hall: ", hall.shape)

        
        #iterate through patch sizes, like in the algorithm.
        for N in patch_sizes:
            print("On this patch size: ", N)            
            p_str = 4 #Weird constant see line 39 of source
            
            current_patch = N #Conversion from source; current_patch = Q_size
            
            S = np.reshape(S, (h,w,3))
            
            #Check this shape
            P = np.zeros( (N*N*3, (math.floor((h-current_patch)/p_str)+1) * (math.floor((w-current_patch)/p_str)+1)*4))
            
            for k in range (0,(h-current_patch+1), p_str): #+2 bc python not inclusive but matlab is
                for j in range (0, (w-current_patch+1), p_str):
                    patch = S[k:(k+current_patch), j:(j+current_patch), :]

                    # if k == 40 and j == 64:
                    #     plt.imshow(patch)
                    #     print("Patch sum: ", np.sum(patch))

                    for i in range (0,4):
                        temp = scp.misc.imrotate(patch,i*90,'bilinear')
                        temp = temp.flatten()
                        
                        #Check this shape
                        P[:, ( (math.ceil(k/p_str)-1)  * (math.floor((w-current_patch)/p_str)+1)*4 +     (math.ceil(j/p_str)-1)*4 + i + 1 )
] = temp
            
            #S = np.copy(S)
            
            #Remove mean
            # print("Pre mean sum: ", sum(P.sum(axis= 0)))
            # print("P shape:", P.shape)
            #exit()
            mp = np.average(P, axis=1)
            mp = np.reshape(mp, (mp.shape[0],1))
            # print('mp shape:', mp.shape)
            P = np.subtract(P,mp)


            #print("Post mean sum of p: ", sum(P.sum(axis= 0)))
            #'''

            #Compute PCA of P
            print("Doing eig")
            #V,D = np.linalg.eig(P @ P.T)

            temp = P @ P.T
            # print("Temp: ", temp.shape)


            #V,D = scp.linalg.eig(temp)
            # eng = matlab.engine.start_matlab()
            # print("engine started")
            # temp = matlab.double(temp.tolist())

            # print("eigening")
            # V, D = eng.eig(temp)
            D, V = np.linalg.eig(temp)

            D = np.array(D)
            # print("D shape: ", D.shape)
            V = np.array(V)
            # print("V shape: ", V.shape)

            
            #D = D.real
            print("eig gotten")
            d = np.sort(D)
            d = d[::-1]
            I = np.argsort(D)
            D = d
            
            V = V[I]

            # print("D shape: ", D.shape)
            # print("I shape: ", I.shape)
            # print("V shape: ", V.shape)
            
            # print("Sum: ",np.sum(D))
            # print(type(D[0]))
            #Find Top eig values
            eig_index = 0
            energy_cutoff = 0.95*np.sum(D)
            energy = 0
            for i in range(0,D.shape[0]):
                #print(D[i])
                energy += D[i]
                
                if energy >= energy_cutoff:
                    print("We out here")
                    eig_index = i
                    break
                    
            print("Eig index:", eig_index)
            Vp = V[:, :eig_index]
            # print("Vp:", Vp.shape)
            # print("P:", P.shape)
            Pp = np.dot(Vp.T,P) #No transpose because of weird shape mismatch P has shape (3888,...) and Vp has 1x3888
            
            # print(Vp.shape)
            # print(Pp.shape)

            #'''
            
            #Vp = np.zeros((1, 3888))
            #Pp = np.zeros((1, 1156))

            
            # Vp = np.zeros((1, 3888))
            # Pp = np.zeros((1, 1156))
                    
            for i in range (0,3):
                print("----------------------STARTING ANOTHER RUN THROUGH OF SYNTHESIS-----------------------")
                print("Run number: ", i) 
                #1. Style fusion
                print("--------------Style Fusion---------------")
                X = hallcoeff*hall+(1-hallcoeff)*hall
                X = X.flatten()
                #print("X post hall coeff: ", X.shape)


                #2. Patch Matching
                print("--------------Patch Matching---------------")
                index = np.argwhere(np.array(patch_sizes) == current_patch)[0][0] #CAN'T LIST SAME PATCH SIZE TWICE!
                
                gap = gap_sizes[index]
                
                #print(gap)
                
                rows = h*w*3
                columns = (math.floor((h-current_patch)/gap)+1) * (math.floor((w-current_patch)/gap)+1)
                #print(rows,columns)
                
                Rall = np.zeros( (rows, columns) )
                z = np.zeros( ( 3*(current_patch**2), ( math.floor( (h-current_patch)/gap ) +1 ) * ( math.floor ( (w-current_patch)/gap) +1 ) ) )
                #print('z_original', z.shape)

                #print("patch thing:", h-current_patch+1)
                #print("gap", gap)


                counter = 0
                for k in range (0, (h-current_patch+1), gap): #DOUBLE CHECK THIS WHEN YOU GET HERE
                    for j in range (0, (h-current_patch+1), gap):
                        counter +=1
                        print("Counter: ", counter)
                        R = np.zeros((h,w,3))
                        R[k:k+current_patch, j:j+current_patch,:] = 1

                        
                        Rall[:,(math.ceil(k/gap)-1)*(math.floor( (w-current_patch)/gap )+ 1) + math.ceil(j/gap)]=R.flatten() #This line is sketchie AF
                        #print("Rall sum: ", np.sum(Rall))

                        ks, ls, zij, ang = nearest_n(R, X, current_patch, S, h, w, 3, Pp,Vp,p_str,mp,L,gap)
                        temp = scp.misc.imrotate(np.reshape(zij,(current_patch,current_patch,3)),ang*90,'bilinear')
                        # print('temp', temp.shape)
                        # print('k', k)
                        # print('j', j)
                        # print('gap', gap)
                        # print('w', w)
                        # print('current_patch', current_patch)
                        # print('value thing', (math.ceil(k/gap)-1)*(math.floor( (w-current_patch)/gap )+ 1) + math.ceil(j/gap))
                        # print('z', z.shape)
                        z[:,(math.ceil((k)/gap)-1)*(math.floor( (w-current_patch)/gap )+ 1) + math.ceil((j)/gap)]=temp.flatten()
                
                #3. Style Synthesis        
                print("------------Robust Aggregation-------------")
                X_tilde=irls(Rall,X,z)
                print(np.sum(X_tilde))

                             
                #4. Content Fusion
                print("----------------------Content Fusion-------------------")
                #print("Wcoeff: ", Wcoeff)
                print("mask: ", mask.shape)

                mask_max = np.amax(mask.flatten() )
                #print("mask_max: ", mask_max)

                W = npm.repmat(Wcoeff* mask.flatten()/mask_max, 1,3 ).T

                #print("W sum: ", np.sum(W))

                W_temp = W + np.ones((W.shape))
                W_temp = np.reshape(W_temp, (W_temp.shape[0],1))
                
                W_temp = np.divide(np.ones(W_temp.shape), W_temp)

                #print("W_temp sum: ", np.sum(W_temp))

                C = C.flatten()
                C = np.reshape(C, (C.shape[0], 1))

                #print("C shape: ", C.shape)

                W_C_temp = np.multiply(W,C)
                #print("W_C sum: ", np.sum(W_C_temp))


                X_tilde = np.reshape(X_tilde, (X_tilde.shape[0],1))
                #print("X_tilde shape: ", X_tilde.shape)

                X_tilde_temp = X_tilde + W_C_temp
                #print("X_tilde_temp sum: ", np.sum(X_tilde_temp))


                X_hat = np.multiply(W_temp, X_tilde_temp) #DOUBLE CHECK THIS #W is (3*Nc/L x 1)) 
                X_hat = np.reshape(X_hat, (h,w,3))

                #print("X_hat_shape: ", X_hat.shape)
                print("X_hat_sum: ", np.sum(X_hat))

                #write_output_img('test.png', X_hat)

                S = np.reshape(S,(h,w,3))
                #print("S shape: ", S.shape)

                



                #5. Color Transfer
                print("------------Color Transfer---------------")

                cX = imhistmatch(X_hat,S ) 
                #write_output_img('test.png', cX)
                
                #6. Denoise
                print("Denoise")
                #COME BACK TO THIS, might be in cv2 https://docs.opencv.org/3.0-beta/modules/ximgproc/doc/edge_aware_filters.html
                #Might be in existing files
        if (L>1):
            X=np.resize(np.reshape(X, (h,w,3)),L)
    return np.reshape(X,(imsize,imsize,3))


# In[10]:

#master routine, calls style transfer. 
def master_routine(pikachu,van_gogh,segment):
    max_resolution = 400 
    #convert content and style to floats
    content = cv2.normalize(pikachu.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    style = cv2.normalize(van_gogh.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    pikachu = pikachu.astype('uint8')
    pikachu = color.rgb2gray(pikachu)
    #crop content and style into max_resolution 
    #content = content[0:max_resolution-1,0:max_resolution-1]
    #style = style[0:max_resolution-1,0:max_resolution-1]
    '''
    first_iteration = style_transfer(content,
                                  style, 
                                  np.ones((max_resolution,max_resolution,3)),
                                  np.ones((max_resolution,max_resolution)), 
                                  0, 
                                  0,
                                  [36, 22],
                                  [4, 2, 1], 
                                  max_resolution
                                  )
    '''
    print("\n\n\n\n ===========FIRST ITERATION COMPLETE!!!!!=============== \n\n\n\n")

    output = style_transfer(content, 
                            style,
                            np.ones((max_resolution,max_resolution,3)),
                            #first_iteration,
                            segment,
                            0.25, 
                            1.5, 
                            [36, 22, 13], 
                            [4, 2, 1], 
                            max_resolution)

    write_output_img('final_output.png', output)
    print("\n\n\n\n ===========DONEZO!!!=============== \n\n\n\n")
    #plt.imshow(output)
    

if __name__ == '__main__':
    I= np.asarray(Image.open('test.jpg').convert("L"), dtype=float)
    pikachu = np.asarray(Image.open('cu.jpg').convert("RGB"),dtype=int) #REPLACED PIKACHU!
    van_gogh = np.asarray(Image.open('van_gogh.jpg').convert("RGB"),dtype=int)
    segment = np.asarray(Image.open('cu_segment.jpg').convert("RGB"),dtype=float)
    master_routine(pikachu,van_gogh,segment)


# In[ ]:

#get_ipython().run_cell_magic('time', '', 'master_routine(pikachu,van_gogh)')


# current_patch = 36
# h = 100
# w = 100
# k = 40
# j = 64
# i= 0
# p_str = 4

# print((math.ceil(k/p_str)-1)  * (math.floor((w-current_patch)/p_str)+1)*4 +     (math.ceil(j/p_str)-1)*4 + i + 1 )
