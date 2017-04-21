#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pylab as lab
import subprocess
import math as ma
import timeit

input = sys.argv[1];
g = open(input, 'r');
g.next()
numBodies = [];
cpuTime = [];
gpuTime = [];
lmcGPUTime = []
gpuNoTrans = []
lmcGPUnoTrans = []
for line in g:
 ln = line.split(' ')
 numBodies.append(int(ln[0]))
 cpuTime.append(float(ln[1]))
 gpuTime.append(float(ln[2]))
 lmcGPUTime.append(float(ln[3]))
 gpuNoTrans.append(float(ln[4]))
 lmcGPUnoTrans.append(float(ln[5]))

# for i in range(len(x1)):
# 	x1s[bID1[i]] = x1[i];
# 	y1s[bID1[i]] = y1[i];

plt.plot(numBodies, cpuTime, 'r', label="i5-3360M @ 2.8GHz\n(Python, Single Thread)")
plt.plot(numBodies, gpuTime, 'b', label="NVS 5400M\n(OpenCL, 96 CUDA Cores)")
plt.plot(numBodies, gpuNoTrans, 'b', label="NVS 5400M Non-Transfer\n(OpenCL, 96 CUDA Cores)")
plt.plot(numBodies, lmcGPUTime, 'g', label="GeForce GTX 480\n(OpenCL, 480 CUDA Cores)")
plt.plot(numBodies, lmcGPUnoTrans, 'g', label="GeForce GTX 480 Non-Transfer\n(OpenCL, 480 CUDA Cores)")
# plt.plot(numBodies, lmcGPUTime, 'g', label="GeForce GTX 480\n(OpenCL, 480 CUDA Cores)")
plt.plot(numBodies, cpuTime, 'or')
plt.plot(numBodies, gpuTime, 'ob')
plt.plot(numBodies, gpuNoTrans, 'xb')
plt.plot(numBodies, lmcGPUTime, 'og')
plt.plot(numBodies, lmcGPUnoTrans, 'xg')
plt.ylabel('Execution Time (ms)')
plt.xlabel('Number of Bodies (n)')
plt.title('Execution and Data Transmit Time of GPU-Accelerated Bounding Box \n Compared to Execution Time of Python Min/Max')
legend = plt.legend(loc="upper left")

plt.show();