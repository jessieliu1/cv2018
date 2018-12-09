
# coding: utf-8

# In[9]:

import numpy as np
import numpy.matlib as npm
import math
#import matlab
#import matlab.engine

# In[10]:

def ind2sub(s, IND):
    print('type IND: ', type(IND))
    row_size = s.shape[0]
    
    I = []
    J = []
    
    for i in IND:
        i = i-1
        J.append( (i//row_size) )
        I.append( (i%row_size) )
    
    I = np.array(I).astype(int)
    J = np.array(J).astype(int)
    return I,J 
      
def add_noise(image,sigma=50):
    print(image.shape)
    row,col,ch= image.shape


    mean = 0
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.astype('uint8')
    gauss = gauss.reshape(row,col,ch)
    return (image + gauss).astype('uint8')

def nearest_n(R, X, Q_size, S, h, w, c, Pp,Vp,Pstride,mp,L, gap):
    print("R sum: ", np.sum(np.sum(R)))
    R = R.flatten()
    X = X.flatten()
    print("X: ", X.shape)
    print("R: ", R.shape)
    S = np.reshape(S, (h,w,3))
    print("S: ", S.shape)
    
    #temp = np.argwhere(R != 0)
    #print(temp.shape)
    #print(temp)
    
    RX = X[np.array(np.where(R!=0.0)[0])]
    RX = np.reshape(RX, (RX.shape[0], 1))
    print("RX:", RX.shape)
    
    #print("test")
    min_l2 = float('inf')

    #print("test2")

    print("Vp: ", Vp.shape)
    RXp = Vp.T @ (np.subtract(RX,mp))
    
    print("RXp: ", RXp.shape)
    
    #eng = matlab.engine.start_matlab()
    print("engine started")

    temp = int(Pp.shape[1])
    #RXp = matlab.double(RXp.tolist())

    print('temp: ', temp)

    print("Heading into tiling")
    dif = npm.repmat(RXp, 1, temp)
    #dif = eng.repmat(RXp, [1, temp])

    print('dif shape: ', dif.shape)

    #dif = np.tile(RXp, (1, Pp.shape[1])) - Pp
    
    #dif = RXp

    print("Out of tiling")
    
    sqr = np.sum(np.square(dif), axis = 0)
    
   # sqr = add_noise(sqr, 2) #Play with STD, also applying gaussian noise 
    sqr = sqr + npm.randn(sqr.shape)
    sqr = sqr.flatten()

    idx = np.argmin(sqr) #find minimum index in each column
    print('idx: ', idx)
    
    temp = np.zeros( (math.floor( (h-Q_size)/Pstride), math.floor( (w-Q_size)/Pstride)) )

    print("ind2sub starting")
    
    inter = np.array(np.ceil(idx/4)).astype(int)
    print('inter ', inter)
    # inter = np.reshape(inter, (inter.shape[0], 1))
    print('type inter ', type(inter))

    ls,ks =ind2sub(temp, [inter])
    # maybe don't subtract?
    ks=ks*Pstride+1
    ls=ls*Pstride+1
    
    ang=np.fmod(np.add(idx, 3),4)
    
    print("ks,ls: ", ks[0], ls[0])
    print("S size: ", S.size)

    z = S[ks[0]-1:(ks[0]+Q_size-1),ls[0]-1:(ls[0]+Q_size-1),:]
    print('z', z.shape)
    z = z.flatten()

    return ks,ls,z,ang
    


# In[ ]:



