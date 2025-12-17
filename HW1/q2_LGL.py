import numpy as np
import matplotlib.pyplot as plt


def legendre(x,N):  # return P_N(x)
    P = np.zeros(N+1)
    P[0] = 1
    P[1] = x
    for i in range(1,N):
        P[i+1] = ((2*i+1)*x*P[i] - i*P[i-1]) / (i+1)
    return P[N]

def Lobatto_roots(N):  # returns N L-G_L points
	epsilon = 0.1
	roots = np.cos(np.pi - np.linspace(epsilon,np.pi-epsilon,N))
	roots[0] = -1
	roots[-1] = 1
	for i in range(1,N-1):
		for itr in range(100):
			n = N-1
			y1 = n*(roots[i]*legendre(roots[i],n)-legendre(roots[i],n-1)) / (roots[i]**2-1)
			dy1 = (n/(roots[i]**2-1)**2)*( (roots[i]**2-1)*legendre(roots[i],n) + (n-2)*roots[i]*(roots[i]*legendre(roots[i],n)-legendre(roots[i],n-1)) - (n-1)*(roots[i]*legendre(roots[i],n-1)-legendre(roots[i],n-2)) )
			roots[i] = roots[i] - y1/dy1
	return roots


def L(i,a,x,N):             # Lagrange poynomial L_i
	b = 1.0
	for j in range(N):
		if (j != i):
			b = b*(a-x[j])/(x[i]-x[j])
	return b

def dL(i,a,x):   #Lagrange poynomial derivative (d/dx)L_i(a)
	c = 0.0
	for k in range(np.size(x)):
		b = 1.0
		if(k != i):
			for j in range(np.size(x)):
				if (j != i and j != k):
					b = b*(a-x[j])/(x[i]-x[j])
			c = c + b/(x[i]-x[k])
	return c
####################################################################

xx = np.linspace(-1,1,50)
df_true =0.5*np.pi*np.cos(np.pi*xx/2)


N_values = np.array([2,4,8,16,20])
l1Error = 0*N_values
l2Error = 0*N_values
lInftyError = 0*N_values

for itr in range(np.size(N_values)):
	N = N_values[itr]
	x = Lobatto_roots(N)
	f = np.sin(np.pi*x/2)
	df = 0*xx
	for i in range(50):
		for j in range(N):
			df[i] = df[i] + dL(j,xx[i],x)*f[j]
			
	l1Error[itr] = np.log10( np.sum(np.abs(df-df_true))/np.sum(np.abs(df_true)) )
	l2Error[itr] = np.log10( np.sqrt(np.sum((df-df_true)**2)/np.sum((df_true)**2)) )
	lInftyError[itr] = np.log10( np.max(np.abs(df-df_true))/np.max(np.abs(df_true)) )




plt.figure()
plt.plot(N_values, l1Error, label="$L_1$ error")
plt.plot(N_values, l2Error,'--', label="$L_2$ error")
plt.plot(N_values, lInftyError,'-.', label="$L_{\infty}$ error")
plt.xlabel("N")
plt.ylabel("log$_{10}$ Error")
plt.legend()
plt.grid()



plt.show()

