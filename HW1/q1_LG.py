import numpy as np
import matplotlib.pyplot as plt


def legendre(x,N):  # return P_N(x)
    P = np.zeros(N+1)
    P[0] = 1
    P[1] = x
    for i in range(1,N):
        P[i+1] = ((2*i+1)*x*P[i] - i*P[i-1]) / (i+1)
    return P[N]

def Legendre_roots(N):
	epsilon = 0.1
	roots = np.cos(np.pi - np.linspace(epsilon,np.pi-epsilon,N))
	for i in range(N):
		for itr in range(100):
			y1 = legendre(roots[i],N)
			dy1 = N*(roots[i]*legendre(roots[i],N)-legendre(roots[i],N-1)) / (roots[i]**2-1)
			roots[i] = roots[i] - y1/dy1
	return roots

def L(i,a,x,N):             # Lagrange poynomial L_i
	b = 1.0
	for j in range(N):
		if (j != i):
			b = b*(a-x[j])/(x[i]-x[j])
	return b

##########################################################################################################

xx = np.linspace(-1,1,50)
ff = 0*xx
ff_true = np.sin(np.pi*xx/2)


N_values = np.array([2,4,8,16,32])
l1Error = 0*N_values
l2Error = 0*N_values
lInftyError = 0*N_values

for itr in range(np.size(N_values)):
	N = N_values[itr]
	x = Legendre_roots(N)
	f = np.sin(np.pi*x/2)
	ff = 0*xx
	for i in range(50):
		for j in range(N):
			ff[i] = ff[i] + L(j,xx[i],x,N)*f[j]
	
	l1Error[itr] = np.log10( np.sum(np.abs(ff-ff_true))/np.sum(np.abs(ff_true)) )
	l2Error[itr] = np.log10( np.sqrt(np.sum((ff-ff_true)**2)/np.sum((ff_true)**2)) )
	lInftyError[itr] = np.log10( np.max(np.abs(ff-ff_true))/np.max(np.abs(ff_true)) )


plt.figure()
plt.plot(N_values, l1Error, label="$L_1$ error")
plt.plot(N_values, l2Error,'--', label="$L_2$ error")
plt.plot(N_values, lInftyError,'-.', label="$L_{\infty}$ error")
plt.xlabel("N")
plt.ylabel("log$_{10}$ Error")
plt.legend()
plt.grid()

plt.show()

