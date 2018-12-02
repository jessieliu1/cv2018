def irls(R, X, z):
    tNc, Nij = R.shape
    I = 5
    Xk = X
    r = 0.8
    unsampled_pix = double(~(R.sum(axis=1) > 0))

    for k in range(0, I):
        A = unsampled_pix
        B = np.matmul(Xk, unsampled_pix)

        for i in range(0, Nij):
            w = np.power(sum(np.power(np.matmul(Xk, np.where(R[:,i] != 0, 1, 0))-z[:,i], 2) + 1e-10), ((r-2)/2))
            A = A + w * R[:,i]
            temp=R[:,i]
            temp[np.where(temp!= 0, 1, 0)]=z[:,i]
            B=B+w*temp
        Xk = np.divide(1,(A + 1e-10)) * B
    return Xk

