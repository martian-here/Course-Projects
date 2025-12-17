import numpy as np
import matplotlib.pyplot as plt

# Lagrange poynomial L_i
def L(i,a,x):
	b = 1.0
	for j in range(np.size(N)):
		if (j != i):
			b = b*(a-x[j])/(x[i]-x[j])
	return b

def Legendre(x,N):  # return P_N(x)
    P = np.zeros(N+1)
    P[0] = 1
    if (N>0):
    	P[1] = x
    if (N>1):
    	for i in range(1,N):
        	P[i+1] = ((2*i+1)*x*P[i] - i*P[i-1]) / (i+1)
    return P[N]

def Legendre_roots(N):          # return N Legendre roots
	epsilon = 0.1
	roots = np.cos(np.pi - np.linspace(epsilon,np.pi-epsilon,N))
	for i in range(N):
		for itr in range(40):
			y1 = Legendre(roots[i],N)
			dy1 = N*(roots[i]*Legendre(roots[i],N)-Legendre(roots[i],N-1)) / (roots[i]**2-1)
			roots[i] = roots[i] - y1/dy1
	return roots

def Legendre_weights(N):       # return N Legendre weights
	weights = np.zeros(N)
	roots = Legendre_roots(N)
	for i in range(N):
		aa = N*(roots[i]*Legendre(roots[i],N) - Legendre(roots[i],N-1)) / (roots[i]**2-1)
		weights[i] = 2 / ((1-roots[i]**2)*aa**2)
	return weights
#.........................................................................................................#


N_values = np.array([1,2,4,8,16,32,64,128])
I_n = 0.0*N_values
I_exact = 4/np.pi
l1Error = 0.0*N_values
l2Error = 0.0*N_values

for itr in range(np.size(N_values)):
	N = N_values[itr] + 1 
	x = Legendre_roots(N)
	f = np.cos(np.pi*x/2)
	weights = Legendre_weights(N)
	I_n[itr] = np.sum(weights*f)
	l1Error[itr] = np.log10( np.abs(I_n[itr]-I_exact) / np.abs(I_exact) )
	l2Error[itr] = np.log10( np.sqrt((I_n[itr]-I_exact)**2) / np.sqrt((I_exact)**2) )
	
	

plt.figure()
plt.plot(N_values, l1Error, 'k', label="$L_1$")
plt.plot(N_values, l2Error,'--r', label="$L_2$")
plt.xlabel("N",fontsize=13)
plt.ylabel("log$_{10}$ Error",fontsize=13)
plt.legend()
plt.grid()

plt.show()
