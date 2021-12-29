
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
#%matplotlib inline

def methodgradientdesc(nmax,theta0,theta1):
    
    X = (np.random.randn(100) + 1) * 50
    jitter = 50 * np.random.randn(100)
    y = theta0 + theta1 * X + jitter
    
    def cost_function(X, y):
        return lambda thetas: sum((thetas[0] + thetas[1] * X - y) ** 2) / len(X)

    J = cost_function(X,y)

    def derivative_theta_0(X, y):
        return lambda theta_0, theta_1: 2*np.sum(theta_0 + theta_1 * X - y) / len(X)

    J_prime_0 = derivative_theta_0(X,y)

    def derivative_theta_1(X, y):
        return lambda theta_0, theta_1: 2*np.sum((theta_0 + theta_1 * X - y) * X) / len(X)

    J_prime_1 = derivative_theta_1(X,y)

    alpha=0.0001
    eps = 0.05
    cost = J([theta0,theta1])

    theta = np.zeros((nmax,2))
    costes = np.zeros((nmax,3))
    
    for i in range(0,nmax):
    
        theta0_new = theta0 - alpha*J_prime_0(theta0,theta1)
        theta1_new = theta0 - alpha*J_prime_1(theta0,theta1)
 
        theta0 = theta0_new
        theta1 = theta1_new
    
        theta[i,0] = theta0
        theta[i,1] = theta1

        costes[i,0] = i+1
        costes[i,1] = cost
    
        cost_new = J([theta0,theta1])
        costes[i,2] = cost_new
    
        convergence = np.abs(cost_new-cost) < eps
        cost = cost_new
    
        if convergence == True:
            print('Convergence FOUND!')
            print('The maximum number of iterations have been',i)
            print(f'The value of the new theta0 = {theta[i,0]:.2f} and the new theta1 = {theta[i,1]:.2f}')
            print(f'Y = {theta[i,0]:.2f} + {theta[i,1]:.2f} * x')
            print(f'with the minimum cost value is {costes[i,2]:.2f}')
            costes_new = costes[0:i,:]
            theta_new = theta[0:i,:]
            nmax_new = i
            fig, ax = plt.subplots(figsize=(12,8))
            ax.plot(X,theta_new[nmax_new-1,1]*X+theta_new[nmax_new-1,0],'g-')    
            ax.scatter(X, y)
            ax.grid(True)
            fig, bx = plt.subplots(figsize=(12,8))
            bx.plot(costes_new[:,0],costes_new[:,1],'r.')
            bx.grid(True)
            break
    if convergence == False:
        print('Convergence NOT FOUND! \nincludes a higher maximum number of iterations\nor other thetas')    
