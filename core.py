import numpy as np 

def Norm(X):
    X[0] = X[0]/X[3]
    X[1] = X[1]/X[3]
    X[2] = X[2]/X[3]
    X[3] = X[3]/X[3]
    return X

def Norm2(X):
    X[0] = X[0]/X[2]
    X[1] = X[1]/X[2]
    X[2] = X[2]/X[2]
    return X

def RT2P(R, T):
		T = T.reshape(3)
		return np.array(
              [[R[0][0],  R[0][1],    R[0][2],  T[0]],
               [R[1][0],  R[1][1],    R[1][2],  T[1]],
               [R[2][0],  R[2][1],    R[2][2],  T[2]],
               [      0, 	    0, 			0,     1]])

def P2RT(P):
    R = np.array(
            [[P[0][0],  P[0][1],    P[0][2]],
            [P[1][0],  P[1][1],    P[1][2]],
            [P[2][0],  P[2][1],    P[2][2]]])

    T = np.array([P[0][3], P[1][3], P[2][3]])
            
    return (R, T)

def GetInvP(R, T):
    return RT2P(R.transpose(), -1*R.transpose().dot(T))

def LinearLSTriangulation(x, P, x1, P1):
    A  = np.zeros((4,4))
    A [0] = (x[0] * P[2] - P[0])
    A [1] = (x[1] * P[2] - P[1])
    A [2] = (x1[0] * P1[2] - P1[0])
    A [3] = (x1[1] * P1[2] - P1[1])

    B = np.zeros((4,1))
    B[0,0] = 0.0001
    B[1,0] = 0.0001 
    B[2,0] = 0.0001 
    B[3,0] = 0.0001
    u, s, vh = np.linalg.svd(A)
    
    #X = np.linalg.solve(A, B).reshape(4)
    X = vh[3].transpose()
    
    Norm(X)
    #Norm(X1)
    #print(X-X1, '\n')
    X_ = np.zeros((3), dtype = 'float32')
    X_[0] = X[0]
    X_[1] = X[1]
    X_[2] = X[2]
    return X_