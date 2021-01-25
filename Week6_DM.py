#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =================================================================
# Class_Ex1:
# We will do some  manipulations on numpy arrays by importing some
# images of a racoon.
# scipy provides a 2D array of this image
# Plot the grey scale image of the racoon by using matplotlib
# ----------------------------------------------------------------
from scipy import misc
#import imageio
import numpy as np
import scipy
import matplotlib.pyplot as plt
face = misc.face(gray=True) 
plt.imshow(face)
plt.show()
print('#',50*"-")


# In[2]:


# =================================================================
# Class_Ex2:
# If still the face is gray choose the color map function and make it
# gray
# ----------------------------------------------------------------
plt.imshow(face, cmap=plt.cm.gray)
print('#',50*"-")


# In[3]:


# =================================================================
# Class_Ex3:
# Crop the image (an array of the image) with a narrower centering
# Plot the crop image again.
# ----------------------------------------------------------------
crop_face = face[100:-100, 100:-100]
plt.imshow(crop_face,cmap=plt.cm.gray)
plt.show()
print('#',50*"-")


# In[4]:


# =================================================================
# Class_Ex4:
# Take the racoon face out and mask everything with black color.
# ----------------------------------------------------------------
sy, sx = face.shape
y, x = np.ogrid[0:sy, 0:sx] # x and y indices of pixels
centerx, centery = (660, 300) # center of the image
mask = ((y - centery)**2 + (x - centerx)**2) > 230**2 # circle
face[mask] = 0
plt.imshow(face)    
print('#',50*"-")


# In[5]:


# =================================================================
# Class_Ex5:
# For linear equation systems on the matrix form Ax=b where A is
# a matrix and x,b are vectors use scipy to solve the for x.
# Create any matrix A and B (Size matters)
# ----------------------------------------------------------------
A = np.mat('[1 2;3 4]')
b= np.array([5,6])
print(A)
print(b)
inva=np.linalg.inv(A)
print(inva)
result=inva.dot(b)
print("result",result)
print('#',50*"-")


# In[6]:


# =================================================================
# Class_Ex6:
# Calculate eigenvalue of matrix A. (create any matrix and check your
# results.)
# ----------------------------------------------------------------
import numpy as np
A = np.matrix([[1,4],[3,5]])
w,v=np.linalg.eig(A)
print("Eigen values",w)
print('#',50*"-")


# In[7]:


# =================================================================
# Class_Ex7:
# Sparse matrices are often useful in numerical simulations dealing
# with large datasets
# Convert sparse matrix to dense and vice versa
# ----------------------------------------------------------------
from scipy.sparse import csr_matrix
A = np.array([[1,0,4],[0,0,5],[0,0,9]])
sA=csr_matrix(A)
print(sA)
d=sA.todense()
print(d)
print('#',50*"-")


# In[8]:


# =================================================================
# Class_Ex8:
# Create any polynomial to order of 3 and write python function for it
# then use scipy to minimize the function (use Scipy)
# ----------------------------------------------------------------
from scipy.optimize import minimize
func= lambda x: (x[0]-3)**3+ (x[1]-5)**2 +x[2]
vall=minimize(func,(2,0,1)) 
print(vall)


# In[9]:


# =================================================================
# Class_Ex9:
# use the brent or fminbound functions for optimization and try again.
# (use Scipy)
# ----------------------------------------------------------------
from scipy.optimize import fminbound, brentq
def func1(x):
    return x**3
val11=fminbound(func1,-1,3)
print(val11)

def func2(y):
    return (y**4)/5
val22=brentq(func2,0,2)
print(val22)


# In[10]:


# =================================================================
# Class_Ex10:
# Find a solution to a function. f(x)=0 use the fsolve (use Scipy)
# ----------------------------------------------------------------
import cmath
from scipy.optimize import fsolve
def func(x):
    return np.cos(x)*2-2*x-1
val= fsolve(func,[1])
print (val)


# In[11]:


# =================================================================
# Class_Ex11:
# Create a sine or cosine function with a big step size. Use scipy to
# interpolate between each data points. Use different interpolations.
# plot the results (use Scipy)
# ----------------------------------------------------------------


import scipy.interpolate
x1=scipy.linspace(-3,3,10)
x1num=scipy.linspace(-3,3,100)
y1=scipy.sin(x1)
polynomial=scipy.interpolate.lagrange(x1,y1)
plt.plot(x1num,polynomial(x1num),x1,y1,'or')
plt.show()

x2=scipy.linspace(-4,4,10)
x2num=scipy.linspace(-4,4,100)
y2=scipy.cos(x2)
polynomial=scipy.interpolate.KroghInterpolator(x2,y2)
plt.plot(x2num,polynomial(x2num),x2,y2,'or')
plt.show()


# In[12]:


# =================================================================
# Class_Ex12:
# Use scipy statistics methods on randomly created array (use Scipy)
# PDF, CDF (CUMsum), Mean, Std, Histogram
# ----------------------------------------------------------------
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
randnums= np.random.randint(1,101,5)
print(type(randnums))
value1=norm.cdf(randnums)
value2=norm.pdf(randnums)
print(value1)
print(value2)
print(norm.mean(randnums))
print(norm.std(randnums))
#print(norm.var(randnums))
plt.hist(randnums)
plt.show()


# In[15]:


# =================================================================
# Class_Ex13:
# USe hypothesise testing  if two datasets of (independent) random varibales
# comes from the same distribution (use Scipy)
# Calculate p values.
# ----------------------------------------------------------------
from scipy.stats import norm 
from scipy.stats import ks_2samp
from scipy.stats import ttest_1samp
import numpy as np
x= np.random.randint(1,101,20)
y= np.random.randint(1,101,20)
diff=x-y
t,p=ttest_1samp(diff,0.0)
print("the p value is :", p)
print('#',50*"-")


# In[ ]:




