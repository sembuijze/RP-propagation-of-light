#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np 
import matplotlib.pyplot as plt 
import scipy
import scipy.stats as scp
from scipy import special
import math
import itertools
from itertools import product


# In[48]:


#Defining all the constants
N_a = 6.022e23
a = 2.1e-29
p = 1013.23e2 #hPa
T0 = 273.15 + 60
L = 10
N = 5000
z = np.linspace(0,8,N)
T = T0 - L*z
R = 8.314
A = 4/3 * np.pi * N_a * a

n = np.sqrt(1 + (3*A*p)/(R*T))


# In[49]:


t = np.linspace(40,80,5)
h = np.linspace(10,30,5)
for i in range(len(h)):
    l = h[i]
    T = T0 - l*z
    n = np.sqrt(1 + (3*A*p)/(R*T))
    b = A*p/(R*T)
    #n = np.sqrt((1+b)/(1-b))
#calculating the angles the lightray makes between the different air layers
    degrees = np.zeros(N)
    degrees[0] = 0.5*np.pi
    for j in range(0,N-1):
        degrees[j+1] = np.arcsin(n[j]/n[j+1] * np.sin(degrees[j]))
        #construct a vector for the lightray using the angles
    vector = np.zeros(N)
    vector[0] = 0
    for m in range(0,N-1):
        vector[m+1] = vector[m] + (z[-1]/N) * np.tan(degrees[m+1])
    plt.plot(vector,z, label = 'L = %s. K/m'%(h[i]))
plt.xlabel('distance (m)')
plt.ylabel('height (m)')
plt.legend()
plt.title('Mirage with a ground temperature of 60 °C and different temperature gradients')
plt.grid()
plt.show()


# In[50]:


vector = np.zeros(N)
vector[0] = 0
for i in range(0,N-1):
    vector[i+1] = vector[i] + (z[-1]/N) * np.tan(degrees[i+1])


# In[51]:


plt.plot(vector,z)
plt.xlabel('distance (m)')
plt.ylabel('height (m)')
plt.title('Mirage with a temperature gradient of 10 K/m and a ground temperatur of 60°C')
plt.grid()
plt.show()


# In[52]:


j = 0 
height = z[0]
while height <= 1.80:
    height = z[j]
    j += 1
    
#normal person with a height of 1.80 m
# invert the path of the light to the point of a hight of 1.80 m and append it to vector so we get the entire path of the light

zx = z[:j]
z_i = zx[::-1]
vectorx = vector[:j]
vector_i = -vectorx[::-1]


# In[53]:


z_1 = np.append(z_i,z)
vector_1 = np.append(vector_i,vector)


# In[54]:


#now we need to calculate where the person will see the mirrage
dy = z_1[1]-z_1[0]
dx = vector_1[1]-vector_1[0]
theta = np.arctan(dy/dx)


# In[55]:


c = dy/dx

y = -1.93 + vector_1*c


# In[56]:


plt.plot(vector_1,z_1, label = 'ray of light')
plt.plot(vector_1,y, label = 'the path the person thinks the light makes')
plt.xlabel('distance (m)')
plt.ylabel('height (m)')
#plt.title('Mirage with a temperature gradient of 10 K/m and a ground temperatur of 60°C')
plt.grid()
plt.ylim(0,4)
plt.legend()
plt.show()


# In[57]:


def y(x,a,alpha,n_0):
    return (np.sin((x/a)*(n_0*alpha))*np.sqrt(n_0**2 - alpha**2))/(n_0*alpha)


x = np.linspace(0,10,100)
n_0 = 1.1
theta = 0.25*np.pi
a = np.sqrt((n_0**2)/(np.tan(theta)**2 + 1))
R = 1
alpha = np.zeros(10)
c = 0.1
alpha = (1/R)-c


path = y(x,a,alpha,n_0)
plt.plot(x,path)
plt.hlines(R,0,10)
plt.hlines(-R,0,10)
plt.xlabel('x (m)')
plt.ylabel('r (cm)')

plt.grid()
plt.show()


# In[58]:


n_0 = 1.1
theta = 0.25*np.pi
a = np.sqrt((n_0**2)/(np.tan(theta)**2 + 1))
R = 1
c = 0.1
alpha = (1/R)-c
N = 100
r = np.linspace(0,1,N)


# In[59]:


n = n_0*np.sqrt(1-alpha**2*r**2)
degrees = np.zeros(N)
degrees[0] = theta
for j in range(N-1):
    degrees[j+1] = np.arcsin(n[j]/n[j+1] * np.sin(degrees[j]))
vector = np.zeros(N)
vector[0] = 0
for m in range(N-1):
    vector[m+1] = vector[m] + (r[-1]/N) * np.tan(degrees[m+1])
plt.plot(vector,r)
plt.xlabel('Distance (m)')
plt.ylabel('Radius (m)')
plt.grid()
plt.show()


# In[ ]:




