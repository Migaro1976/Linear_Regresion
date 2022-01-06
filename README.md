# Implementing a Linear Regression problem with Python 

In linear regression, our hypothesis function ℎ𝜃 is:

  ℎ𝜃(𝑥)=𝜃0+𝜃1𝑥

And, as we are doing regression, our cost function is: 𝐽(𝜃0,𝜃1)=1𝑚∑𝑖=1𝑚(𝑦̂ 𝑖−𝑦𝑖)2=1𝑚∑𝑖=1𝑚(ℎ𝜃(𝑥𝑖)−𝑦𝑖)2
Nota that, the cost funtion is just the sum of all the square errors from our hypothesis (𝑦̂ 𝑖) versus the data (𝑦𝑖).

The best parameters for our hypothesis will give us the minimum cost function.

Finding a minimum for J
Finding a minimum of a function is equivalent to finding the parameters that make the gradient of that function to vanish. In other words:

∇𝜃𝐽(𝜃)=0
We will implement two ways of solving this problem.

# A) Gradient descent (Numerical method)
From a starting point (𝜃), we will try to move to a new point 𝜃′, decreasing the cost funtion (𝐽(𝜃)). We will do this many times, up to we find a minimum (or close enough to it).

Partial differentials of the cost function (using chain rule)
∂𝐽∂𝜃0=2𝑚∑𝑖=1𝑚(ℎ𝜃(𝑥𝑖)−𝑦𝑖)
∂𝐽∂𝜃1=2𝑚∑𝑖=1𝑚(ℎ𝜃(𝑥𝑖)−𝑦𝑖)⋅𝑥𝑖
Finally, we need to update iteratively the values for 𝜃0 and 𝜃1. Using Gradient Descent algorithm with learning rate (𝛼) until convergence criterion (𝜖) is achieved:

     while (convergence==False):
𝜃′0=𝜃0−𝛼∂𝐽∂𝜃0
𝜃′1=𝜃1−𝛼∂𝐽∂𝜃1
𝐽′=𝐽(𝜃′0,𝜃′1)
Δ𝐽=𝑎𝑏𝑠(𝐽′−𝐽)
𝑐𝑜𝑛𝑣𝑒𝑟𝑔𝑒𝑛𝑐𝑒=(Δ𝐽<𝜖)

# B) Normal equations (Algebra)
In matrix notation, we can implement our hypothesis as:

ℎ𝜃(𝑥(𝑖))=(𝑥(𝑖))𝑇𝜃
Note that, in this case, if we want to consider our hypothesis such ℎ(𝜃)=𝜃0+𝜃𝑖𝑥(𝑖) where 𝑥 is a vector, for consistency, we need to introduce an additional "constant feature" in our data. In other words, we need to map our input data as follows:

𝑥𝑖→[1,𝑥𝑖]
we can express gradient of J as follows:

∇𝜃𝐽(𝜃)=𝑋𝑇𝑋𝜃−𝑋𝑇𝑦⃗ 
To minimize J, we set its derivatives to zero, therefore obtaining the normal equations:

𝑋𝑇𝑋𝜃=𝑋𝑇𝑦⃗ 
We can solve this equation for theta.

As a final remark, we can extend this method to non linear hypothesis by extending our input data 𝑥 to the features we need. For example, for a parabolic fit:

𝑥𝑖→[1,𝑥𝑖,𝑥2𝑖]

# Problem
Giving the data provided below (x->y), find the best equation fit, using:
-Gradient Descent
-Normal Equations
Using your own python implementation, using numpy and scipy tools (not scipy!).
