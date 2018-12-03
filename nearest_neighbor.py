
# coding: utf-8

# In[9]:

import numpy as np


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
      

def nearest_n(R, X, Q_size, S, h, w, c, Pp,Vp,Pstride,mp,L, gap):

    S = np.reshape(S, (h,w,3))
    RX = np.matmul(X, np.where(R != 0, 1, 0))
    
    min_l2 = float('inf')

    RXp = Vp.T * (np.subtract(RX,mp))
    dif = np.tile(RXp, (1, Pp.shape[1])) - Pp
    
    
    sqr = np.sum(np.square(dif), axis = 0)
    
    sqr = add_noise(sqr, 2) #Play with STD, also applying gaussian noise 
    
    idx = np.argmin(sqr, axis = 0) #find minimum index in each column
    
    
    temp = np.zeros( (math.floor( (h-Q_size)/Pstride), math.floor( (w-Q_size)/Pstride)) )

    ls,ks =ind2sub(temp,math.ceil(idx/4)) 
    ks=(np.subtract(ks-1))*Pstride+1
    ls=(np.subtract(ls-1))*Pstride+1
    
    ang=np.fmod(np.add(idx+3),4)
    
    z = S[ks:(ks+Q_size),ls:(ls+Q_size),:]
    
    return ks,ls,z,ang
    


# In[ ]:



