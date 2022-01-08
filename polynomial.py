
#CREAMOS UN GRÁFICO CON PUNTOS DE DISPERSIÓN.
#DESPUÉS CON DIFERENTES FUNCIONES POLINÓMICAS VEMOS COMO SE AJUSTAN
#A LA NUBE DE PUNTOS

import numpy as np

theta_0 = 10
theta_1 = 1
theta_2 = 5
theta_3 = 4

X = (np.random.randn(100) + 1) * 50
X = np.linspace(-100,150,100)
jitter = 1000000 * np.random.randn(100)
y = theta_0 + theta_1 * X + theta_2 *(X**2) + theta_3 *(X**3) + jitter
  
def thetas(m):
    '''
    This function implement a Linear Regression problem with Normal equations (Algebra)
    
    We must introduce the degree (m) of the polynomial to obtain the thetas of the polynomial function
    
    In matrix notation, we can implement our hypothesis as: ℎ𝜃(𝑥(𝑖))=(𝑥(𝑖))𝑇𝜃
    
    Note that, in this case, if we want to consider our hypothesis such  ℎ(𝜃)=𝜃0+𝜃𝑖𝑥(𝑖) where  𝑥  is a vector,
    for consistency, we need to introduce an additional "constant feature" in our data. In other words, 
    we need to map our input data as follows: 𝑥𝑖→[1,𝑥𝑖]
    
    We can express gradient of J as follows: ∇𝜃𝐽(𝜃)=𝑋𝑇𝑋𝜃−𝑋𝑇𝑦⃗ To minimize J, we set its derivatives to zero,
    therefore obtaining the normal equations: 𝑋𝑇𝑋𝜃=𝑋𝑇𝑦⃗ 
    We can solve this equation for theta. As a final remark, we can extend this method to non linear hypothesis
    by extending our input data 𝑥 to the features we need. 
    '''
    import numpy as np
    def matriz_polinomio(n):
        #Se construye matriz [1,X]
        # |1   X01  ... X0m|
        # |1   X11  ... X1m|
        # |...             |
        # |1   Xn1  ... Xnm|
        Vector1 = np.ones((len(X)))
        VectorX = np.array(X)
        Union_X = np.array([Vector1,VectorX])
        for i in range(1,n):
            Vector1X = np.array(X**(i+1))
            Union_X = np.append(Union_X, [Vector1X],axis=0)
        return Union_X
   
    Matriz_1X = matriz_polinomio(m).T
    #para despejar 𝜃 de la ecuacion 𝑋𝑇𝑋𝜃=𝑋𝑇𝑦⃗  hay que multiplicar por la inversa de 𝑋𝑇𝑋 en los dos lados
    #se crea una matriz XTX y su inversa
    XTX = np.dot(Matriz_1X.T,Matriz_1X)
    InvXTX = np.linalg.inv(XTX)

    #primero multiplicamos la inversa de XTX por la traspuesta de X
    InvXTX_XmatrizT = np.dot(InvXTX,Matriz_1X.T)

    #por último multiplicamos el resultado anterior por el vector Y
    VectorY = np.array(y)
    InvXTX_XT_Y = np.dot(InvXTX_XmatrizT,VectorY.T)
    theta = InvXTX_XT_Y
    return theta
