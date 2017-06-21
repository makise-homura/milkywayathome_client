#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pylab as lab
import subprocess

#./bin/milkyway_nbody -f nbody/sample_workunits/EMD_10k_plummer.lua -o CPUBRUTE.out -z CPU.hist -x -i -e 36912 1 1 .2 12
timesteps = 1

os.system('rm GPUBRUTE.out')
# os.system('rm CPUBRUTE.out')
os.system('rm GPUACCTEST.out')

print "RUNNING GPU SYSTEM:"
os.system("cmake -DNBODY_OPENCL=ON -DDOUBLEPREC=ON -DCMAKE_BUILD_TYPE=RELEASE ")
os.system('make -j 9')
executeString = './bin/milkyway_nbody -f nbody/sample_workunits/for_developers.lua -o GPUBRUTE.out -z GPU.hist -b -i 4 1 .2 1 12 1' #>> GPUACCTEST.out'
os.system(executeString)