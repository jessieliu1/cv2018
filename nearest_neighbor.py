
# coding: utf-8

# In[9]:

import numpy as np
import numpy.matlib as npm
import matlab
import matlab.engine

# In[10]:

def ind2sub(s, IND):
    row_size = s.shape[0]
    
    I = []
    J = []
    
    for i in IND:
        i = i-1
        
        J.append( (i//row_size) )
        I.append( (i%row_size) )
    
    I = np.array(I)
    J = np.array(J)
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

    S = np.reshape(S, (h,w,3))
    
    temp = np.argwhere(R != 0)
    #print(temp.shape)
    #print(temp)
    
    RX = np.where(np.argwhere(R != 0))[0]
    print("RX:", RX.shape)
    
    #print("test")
    min_l2 = float('inf')

    #print("test2")
    
    RXp = Vp.T * (np.subtract(RX,mp))
    
    print("RXp: ", RXp.shape)
    
    #eng = matlab.engine.start_matlab()
    print("engine started")

    temp = int(Pp.shape[1])
    #RXp = matlab.double(RXp.tolist())



    print("Heading into tiling")
    RXp = npm.repmat(RXp, 1, temp)
    #dif = eng.repmat(RXp, [1, temp])

    print(dif)



    #dif = np.tile(RXp, (1, Pp.shape[1])) - Pp
    
    #dif = RXp

    print("Out of tiling")
    
    sqr = np.sum(np.square(dif), axis = 0)
    
    sqr = add_noise(sqr, 2) #Play with STD, also applying gaussian noise 
    
    idx = np.argmin(sqr, axis = 0) #find minimum index in each column
    
    
    temp = np.zeros( (math.floor( (h-Q_size)/Pstride), math.floor( (w-Q_size)/Pstride)) )

    print("ind2sub starting")
    
    ls,ks =ind2sub(temp,math.ceil(idx/4)) 
    ks=(np.subtract(ks-1))*Pstride+1
    ls=(np.subtract(ls-1))*Pstride+1
    
    ang=np.fmod(np.add(idx+3),4)
    
    z = S[ks:(ks+Q_size),ls:(ls+Q_size),:]
    
    return ks,ls,z,ang
    


# In[ ]:



