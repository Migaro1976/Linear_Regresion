
def linearegression2(theta0,theta1,theta2):

    '''
    This function creates scatter points by entering 3 parameters: theta0, theta1 and theta2 
    We will obtain data for X and Y and so on fit to a quadratic regression line by 
    the algebraic method of normal equations
    
    In matrix notation, we can implement our hypothesis as: ℎ𝜃(𝑥(𝑖))=(𝑥(𝑖))𝑇𝜃
    
    Note that, in this case, if we want to consider our hypothesis such  ℎ(𝜃)=𝜃0+𝜃𝑖𝑥(𝑖)  
    where  𝑥  is a vector, for consistency, we need to introduce an additional "constant feature" in our data.
    
    In other words, we need to map our input data as follows: 𝑥𝑖→[1,𝑥𝑖]
    
    We can express gradient of J as follows: ∇𝜃𝐽(𝜃)=𝑋𝑇𝑋𝜃−𝑋𝑇𝑦⃗ 
    
    To minimize J, we set its derivatives to zero, therefore obtaining the normal equations: 𝑋𝑇𝑋𝜃=𝑋𝑇𝑦⃗ 
    We can solve this equation for theta. 
    
    As a final remark, we can extend this method to non linear hypothesis by extending our 
    input data 𝑥 to the features we need. For example, for a parabolic fit: 𝑥𝑖→[1,𝑥𝑖,𝑥2𝑖]
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    X = (np.random.randn(100) + 1) * 50
    X = np.linspace(-100,150,100)
    jitter = 3000 * np.random.randn(100)
    y = theta0 + theta1 * X + theta2 *(X**2) + jitter
    
#Se construye matriz [1,X,X**2]
# |1   X01  X02|
# |1   X11  X22|
# |...         |
# |1   Xn1  Xm2|

    Vector1 = np.ones((len(X)))
    VectorX = np.array(X)
    VectorX2 = np.array(X**2)
    Union_1XX2 = np.array([Vector1,VectorX,VectorX2])
    Matriz_1XX2 = Union_1XX2.T

# para despejar 𝜃 de la ecuacion 𝑋𝑇𝑋𝜃=𝑋𝑇𝑦⃗  hay que multiplicar por la inversa de 𝑋𝑇𝑋 en los dos lados
# se crea una matriz XTX y su inversa
    XTX = np.dot(Matriz_1XX2.T,Matriz_1XX2)
    InvXTX = np.linalg.inv(XTX)

# primero multiplicamos la inversa de XTX por la traspuesta de X
    InvXTX_XmatrizT = np.dot(InvXTX,Matriz_1XX2.T)

# por último multiplicamos el resultado anterior por el vector Y
    VectorY = np.array(y)
    InvXTX_XT_Y = np.dot(InvXTX_XmatrizT,VectorY.T)
    theta = InvXTX_XT_Y

    #print(f'The value of theta0 = {theta[0]:.2f} , theta1 = {theta[1]:.2f} and theta2 = {theta[2]:.2f}')
    print(f'Y = {theta[0]:.2f} {theta[1]:+.2f}*x {theta[2]:+.2f}*x\u00B2')

    fig, bx = plt.subplots(figsize=(12,8))
    bx.plot(X,theta[0]+theta[1]*X+theta[2]*(X**2),'g-')
    bx.scatter(X, y)
    bx.grid(True)
