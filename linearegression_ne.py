
def linearegression(theta0,theta1):

    '''
    This function creates scatter points by entering 2 parameters: theta0, theta1
    We will obtain data for X and Y and so on fit to a quadratic regression line by 
    the algebraic method of normal equations
    
    In matrix notation, we can implement our hypothesis as: βπ(π₯(π))=(π₯(π))ππ
    
    Note that, in this case, if we want to consider our hypothesis such  β(π)=π0+πππ₯(π)  
    where  π₯  is a vector, for consistency, we need to introduce an additional "constant feature" in our data.
    
    In other words, we need to map our input data as follows: π₯πβ[1,π₯π]
    
    We can express gradient of J as follows: βππ½(π)=ππππβπππ¦β 
    
    To minimize J, we set its derivatives to zero, therefore obtaining the normal equations: ππππ=πππ¦β 
    We can solve this equation for theta. 
    
    As a final remark, we can extend this method to non linear hypothesis by extending our 
    input data π₯ to the features we need. For example, for a parabolic fit: π₯πβ[1,π₯π,π₯2π]
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    X = (np.random.randn(100) + 1) * 50
    X = np.linspace(-100,150,100)
    jitter = 50 * np.random.randn(100)
    y = theta0 + theta1 * X + jitter
    
#Se construye matriz [1,X]
# |1   X01 |
# |1   X11 |
# |...     |
# |1   Xn1 |

    Vector1 = np.ones((len(X)))
    VectorX = np.array(X)
    
    Union_1X = np.array([Vector1,VectorX])
    Matriz_1X = Union_1X.T

# para despejar π de la ecuacion ππππ=πππ¦β  hay que multiplicar por la inversa de πππ en los dos lados
# se crea una matriz XTX y su inversa
    XTX = np.dot(Matriz_1X.T,Matriz_1X)
    InvXTX = np.linalg.inv(XTX)

# primero multiplicamos la inversa de XTX por la traspuesta de X
    InvXTX_XmatrizT = np.dot(InvXTX,Matriz_1X.T)

# por ΓΊltimo multiplicamos el resultado anterior por el vector Y
    VectorY = np.array(y)
    InvXTX_XT_Y = np.dot(InvXTX_XmatrizT,VectorY.T)
    theta = InvXTX_XT_Y

    #print(f'The value of theta0 = {theta[0]:.2f} , theta1 = {theta[1]:.2f} and theta2 = {theta[2]:.2f}')
    print(f'Y = {theta[0]:.2f} {theta[1]:+.2f}*x')

    fig, bx = plt.subplots(figsize=(12,8))
    bx.plot(X,theta[0]+theta[1]*X,'g-')
    bx.scatter(X, y)
    bx.grid(True)
