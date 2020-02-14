import numpy as np
import time
import h5py
from scipy.io import loadmat
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import cg

tic = time.clock()

#Hydraulic Conductivity <<<<<<<<<<<<<[Input!] 
filepath= 'C:/UHM/Research/Upscaling/WD/relz.mat'
f = h5py.File(filepath)
mat = list(f['relz'])
data = np.asarray(mat)
Kdata = np.exp(data[1,:,:]).T

#Dimesion
d_up = 10        #<<<<<<<<<<<<<[Input!]     #Upscaled grid = d_up by d_up
d_fine = Kdata.shape[1]
N = int(d_fine/d_up)

#Iteration
#maxit = 100
####################[Upscaling]###############################
Kxx = np.zeros((d_up,d_up))
Kyy = np.zeros((d_up,d_up))
Kxy = np.zeros((d_up,d_up))

#Function
def spec2D(xx):
    xx = xx.reshape(-1,1)
    Ax =-xk*np.fft.fftn(K*np.fft.ifftn(np.reshape(xk.reshape(-1,1)*xx,(N,N))))-yk*np.fft.fftn(K*np.fft.ifftn(np.reshape(yk.reshape(-1,1)*xx,(N,N))))
    return Ax.reshape(-1)

#Upscaling
proc = 0
for i in range(d_up):
    for j in range(d_up):
        proc = proc + 1
        K = Kdata[N*i:N*(i+1),N*j:N*(j+1)]
        #Grid
        n = int(N/2)
        kk = np.asarray(list(range(0,n+1))+list(range(-n+1,0)))  #kk = [0:N/2-1 N/2 -N/2+1:-1] in matlab
        xk, yk = np.meshgrid(kk,kk)

        #Solve g1 & g2
        b1 = 1j*xk*np.fft.fftn(K)
        b2 = 1j*yk*np.fft.fftn(K)
       
        A = LinearOperator((N**2,N**2), matvec = spec2D)
         
        g1, exitcode1 = cg(A,b1.reshape(-1))
        if exitcode1 != 0:
            print("cg not converged: %d, %d, %s" % (exitcode1,proc,'g1'))
            g1, exitcode1 = gmres(A,b1.reshape(-1),x0=g1)
       # print('g1 convergence:', exitcode1)
        
        g2, exitcode2 = cg(A,b2.reshape(-1))
        if exitcode2 != 0:
            print("cg not converged: %d, %d, g2" % (exitcode1,proc))
            g2, exitcode2 = gmres(A,b2.reshape(-1),x0=g2)
        if proc % (2*d_up) == 0:
            print("Processing: [%d/%d]" % (proc,d_up**2))
        
        gm1 = g1.reshape(N,N)
        gm2 = g2.reshape(N,N)

        Kij = -0.5*np.mean(K*np.fft.ifftn(1j*yk*gm1+1j*xk*gm2))
        Kaa = -0.5*np.mean(K*np.fft.ifftn(2j*xk*gm1))
        Kbb = -0.5*np.mean(K*np.fft.ifftn(2j*yk*gm2))

        Kxx[i,j]= np.mean(K)+Kaa
        Kyy[i,j]= np.mean(K)+Kbb
        Kxy[i,j]= Kij
        

toc = time.clock()
print('Upscaling time:',toc-tic)