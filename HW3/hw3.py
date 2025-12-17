import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def legendre(x,N):  # return P_N(x)
    P = np.zeros(N+1)
    P[0] = 1
    if (N>0):
    	P[1] = x
    if (N>1):
    	for i in range(1,N):
        	P[i+1] = ((2*i+1)*x*P[i] - i*P[i-1]) / (i+1)
    return P[N]

def Lobatto(x,N):    # return Lo_N(x)  N>1
    return (1-x**2)*(N-1)*(x*legendre(x,N-1) - legendre(x,N-2)) / (x**2-1)

def Lobatto_roots(N):
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


def Lobatto_weights(N):
    weights = np.zeros(N)
    roots = Lobatto_roots(N)
    for i in range(1,N-1):
        weights[i] = 2 / (N*(N-1)*legendre(roots[i],N-1)**2)
    weights[0] = 2/(N*(N-1))
    weights[-1] = 2/(N*(N-1))
    return weights


def L(i,a,x):             # Lagrange poynomial L_i(a)
	b = 1.0
	for j in range(np.size(x)):
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
	
def rk4(q,Rp,dt):
	k1 = dt * np.dot(Rp,q)
	k2 = dt * np.dot(Rp,q+k1/2)
	k3 = dt * np.dot(Rp,q+k2/2)
	k4 = dt * np.dot(Rp,q+k3)
	return q + dt * (k1+2*k2+2*k3+k4)/6

def R_dg(N,Ne,No,dx):
	xi, w_k, x_k = Lobatto_roots(N+1), Lobatto_weights(No+1), Lobatto_roots(No+1)	
	M, D, F = np.zeros((N+1,N+1)), np.zeros((N+1,N+1)), np.zeros((N+1,N+3))
	for i in range (N+1):
		for j in range (N+1):
			for k in range (No+1):
				M[i,j] = M[i,j] + w_k[k]*L(i,x_k[k],xi)*L(j,x_k[k],xi)  # \Delta x/2 not included
				D[i,j] = D[i,j] + w_k[k]*dL(i,x_k[k],xi)*L(j,x_k[k],xi)	
				
	F[0,0] = -1/2; F[0,1] = -1/2; F[-1,-1] = 1/2; F[-1,-2] = 1/2
	A= np.zeros((Ne*(N+1),Ne*(N+1)))
	B = np.zeros((Ne*(N+1),Ne*(N+1)))
	C = np.zeros((Ne*(N+1),Ne*(N+1)))	
	for i in range(Ne):
		A[i*(N+1):(i+1)*(N+1),i*(N+1):(i+1)*(N+1)] = A[i*(N+1):(i+1)*(N+1),i*(N+1):(i+1)*(N+1)] + M
		B[i*(N+1):(i+1)*(N+1),i*(N+1):(i+1)*(N+1)] = B[i*(N+1):(i+1)*(N+1),i*(N+1):(i+1)*(N+1)] + D
		if (i>0 and i < Ne-1):
			C[i*(N+1):(i+1)*(N+1),i*(N+1)-1:(i+1)*(N+1)+1] = C[i*(N+1):(i+1)*(N+1),i*(N+1)-1:(i+1)*(N+1)+1] + F
	if (Ne>1):
		C[:N+1,:N+2] = C[:N+1,:N+2] + F[:,1:]
		C[(Ne-1)*(N+1):,(Ne-1)*(N+1)-1:] = C[(Ne-1)*(N+1):,(Ne-1)*(N+1)-1:] + F[:,:-1]
	
	# Periodic boundary
	Mg, Dg, Fg = A, B, C
	Fg[0,-1] = F[0,0]
	Fg[-1,0] = F[-1,-1]
	
	R = np.matmul(np.linalg.inv(Mg),(Fg-Dg))	
	return -2*R /dx

def R_cg(N,Ne,No,dx):
	xi, w_k, x_k = Lobatto_roots(N+1), Lobatto_weights(No+1), Lobatto_roots(No+1)
	M, D = np.zeros((N+1,N+1)), np.zeros((N+1,N+1))
	for i in range (N+1):
		for j in range (N+1):
			for k in range (No+1):
				M[i,j] = M[i,j] + w_k[k]*L(i,x_k[k],xi)*L(j,x_k[k],xi)  # \Delta x/2 not included
				D[i,j] = D[i,j] + w_k[k]*L(i,x_k[k],xi)*dL(j,x_k[k],xi)

	A, B = np.zeros((Ne*N+1,Ne*N+1)), np.zeros((Ne*N+1,Ne*N+1))
	for i in range(Ne):
		A[i*N:(i+1)*N+1,i*N:(i+1)*N+1] = A[i*N:(i+1)*N+1,i*N:(i+1)*N+1] + M
		B[i*N:(i+1)*N+1,i*N:(i+1)*N+1] = B[i*N:(i+1)*N+1,i*N:(i+1)*N+1] + D

	Mg = A[:-1,:-1]	
	Dg = B[:-1,:-1]
	# Periodic boundary
	if(Ne>1):
		Mg[0,0] = Mg[N,N]
		Mg[0,Ne*N-N:] = Mg[N,:N]
		Mg[-1,-1] = Mg[-N-1,-N-1]
		Mg[-N:,0] = Mg[-2*N:-N,-N]

		Dg[0,0] = Dg[N,N]
		Dg[0,Ne*N-N:] = Dg[N,:N]
		Dg[-1,-1] = Dg[-N-1,-N-1]
		Dg[-N:,0] = Dg[-2*N:-N,-N]
	R = np.matmul(np.linalg.inv(Mg),Dg)	
	return -2*R /dx
	
# ++++++++++++++++++++++++ Advection Eq+++++++++++++++++++++++++++++
	
	
NN = np.array([1,4,8,16])#int(input("Enter polynomial order (N): "))

while (method := int(input("Enter Method 0 - CG  or  1 - DG: "))) not in [0, 1]: print("Error: Enter only 0 or 1!")

while (ab := int(input("Enter order of Integration: InExact - 0 or  Exact integration - 1: "))) not in [0, 1]: print("Error: Enter only 0 or 1!")
No = ab + NN

u = 2

error = np.zeros((4,3))  #(N,Ne)

Np = np.array([16,32,64])

if (method == 0):
	for kk in range(4):
		N = NN[kk]
		Q = No[kk]
		no_of_elements = Np // N
		xi = Lobatto_roots(N+1)
		for itr in range(np.size(no_of_elements)):
			Ne = no_of_elements[itr]
			Nt = N*Ne+1
			dx = 2/Ne
			x = np.zeros(Nt)
			for i in range (Ne):
				x[i*N:i*N+N]  = (dx/2) * (1+xi[:-1]) + i*dx -1
			q = np.exp(-16*x**2)
			q_true = np.exp(-16*x**2)
			Rp = R_cg(N,Ne,Q,dx)
			dt = dx/1000
			t = np.arange(0,1+dt,dt)
			for k in range (np.size(t)):
				q[:-1] = rk4(q[:-1],u*Rp,dt)
				q[-1] = q[0]
			error[kk,itr] = np.log10 (np.sqrt( np.sum((q_true-q)**2) / np.sum((q_true)**2) ) )
		
		
elif (method == 1):
	for kk in range(4):
		N = NN[kk]
		Q = No[kk]
		no_of_elements = Np // N
		xi = Lobatto_roots(N+1)
		for itr in range(np.size(no_of_elements)):
			Ne = no_of_elements[itr]
			Nt = Ne*(N+1)
			dx = 2/Ne
			x = np.zeros(Nt)
			for i in range (Ne):
				x[i*(N+1):i*(N+1)+N+1]  = (dx/2) * (1+xi) + i*dx -1
			q = np.exp(-16*x**2)
			q_true = q = np.exp(-16*x**2)
			Rp = R_dg(N,Ne,Q,dx)
			dt = dx/5000
			t = np.arange(0,1+dt,dt)
			for k in range (np.size(t)):
				q = rk4(q,u*Rp,dt)
			error[kk,itr] = np.log10 (np.sqrt( np.sum((q_true-q)**2) / np.sum((q_true)**2) ) )


print(error)
plt.plot(Np+1, error[0,:],'-sk',label='N=1')
plt.plot(Np+1, error[1,:],'-sb',label='N=4')
plt.plot(Np+1, error[2,:],'-sr',label='N=8')
plt.plot(Np+1, error[3,:],'-sg',label='N=16')
plt.xlabel("$N_p$",fontsize=12)
plt.ylabel("log$_{10}(L_2$ error)",fontsize=12)
plt.title(f"{'CG' if method == 0 else 'DG'}, {'Exact Integration' if ab == 1 else 'Inexact Integration'}")
plt.legend()
plt.grid()

plt.show()
