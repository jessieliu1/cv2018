import numpy as np
import numpy.matlib as npm

def irls(R, X, z):
    print("Full R: ", R.shape)
    print("Zeros in R: ", len( ( np.where(R< 1)[0] )  )   )

    tNc, Nij = R.shape
    I = 5 #Changed from 5!
    Xk = X
    r = 0.8
    #unsampled_pix = (~(R.sum(axis=1) > 0)).astype(float)
    unsampled_pix = np.zeros((tNc))
    R_copy = np.copy(R)

    for k in range(0, I):
        print("----------new k iteration-------")
        print("k iteration number: ",k)
        
        R_copy = np.copy(R)

        print("Zeros in R: ", len( ( np.where(R_copy< 1)[0] )  )   )
        A = unsampled_pix
        B = np.multiply(Xk, unsampled_pix)

        for i in range(0, Nij):
            #print("----------new Nij iteration-------")
            #print("Nij iteration number: ", i)

            xk_temp = np.array(np.where(R_copy[:,i]!=0.0))

            #print("xk_temp shape: ", xk_temp.shape)

            logical = Xk[xk_temp[0]]
            #print("Xk Logical: ", logical.shape)

            subtract = np.subtract(logical, z[:,i])
            square = np.square(subtract)
            add = np.add(square, 1e-10)
            summation = np.sum(add)

            w = np.power(summation, ((r-2)/2))

            #print("W: ", w)

            A = A + w * R_copy[:,i]

            #print("A shape: ", A.shape)

            B_temp=R_copy[:,i]

            #print("Pre logical B temp: ", B_temp.shape)
            #print("B Nonzeros: ", len(np.nonzero(B_temp)[0]))
            #print("Z :", z[:,i].shape)


            B_temp[np.nonzero(B_temp)] = z[:,i]
            #print("B_Temp shape: ", B_temp.shape)

            B=B+w*B_temp
            #print("B shape: ", B.shape)

        Xk = np.multiply (np.divide(1,(A + 1e-5)), B)


    return Xk







