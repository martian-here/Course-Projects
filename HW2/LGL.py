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

def Lobatto(x,N):    # return Lo_N(x)
    return (1-x**2)*(N-1)*(x*Legendre(x,N-1) - Legendre(x,N-2)) / (x**2-1)

def Lobatto_roots(N):
    epsilon = 0.1
    roots = np.cos(np.pi - np.linspace(epsilon,np.pi-epsilon,N))
    roots[0] = -1
    roots[-1] = 1
    for i in range(1,N-1):
        for itr in range(200):
            n = N-1
            y1 = n*(roots[i]*Legendre(roots[i],n)-Legendre(roots[i],n-1)) / (roots[i]**2-1)
            dy1 = (n/(roots[i]**2-1)**2)*( (roots[i]**2-1)*Legendre(roots[i],n) + (n-2)*roots[i]*(roots[i]*Legendre(roots[i],n)-Legendre(roots[i],n-1)) - (n-1)*(roots[i]*Legendre(roots[i],n-1)-Legendre(roots[i],n-2)) )
            roots[i] = roots[i] - y1/dy1
    return roots


def Lobatto_weights(N):
    weights = np.zeros(N)
    roots = Lobatto_roots(N)
    for i in range(1,N-1):
        weights[i] = 2 / (N*(N-1)*Legendre(roots[i],N-1)**2)
    weights[0] = 2/(N*(N-1))
    weights[-1] = 2/(N*(N-1))
    return weights
#.........................................................................................................#


N_values = np.array([2,4,8,16,32,64,128]) + 1
I_n = 0.0*N_values
I_exact = 4/np.pi
l1Error = 0.0*N_values
l2Error = 0.0*N_values

for itr in range(np.size(N_values)):
	N = N_values[itr]
	x = Lobatto_roots(N)
	f = np.cos(np.pi*x/2)
	weights = Lobatto_weights(N)
	I_n[itr] = np.sum(weights*f)
	l1Error[itr] = np.log10( np.abs(I_n[itr]-I_exact) / np.abs(I_exact) )
	l2Error[itr] = np.log10( np.sqrt((I_n[itr]-I_exact)**2) / np.sqrt((I_exact)**2) )
	
	

plt.figure()
plt.plot(N_values, l1Error,'k', label="$L_1$")
plt.plot(N_values, l2Error,'--r', label="$L_2$")
plt.xlabel("N",fontsize=13)
plt.ylabel("log$_{10}$ Error",fontsize=13)
plt.legend()
plt.grid()

plt.show()
