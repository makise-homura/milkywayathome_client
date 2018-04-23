#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pylab as lab
import subprocess

#./bin/milkyway_nbody -f nbody/sample_workunits/EMD_10k_plummer.lua -o CPUBRUTE.out -z CPU.hist -x -i -e 36912 1 1 .2 12
timesteps = 50000

os.system('rm GPUBRUTE.out')
os.system('rm CPUBRUTE.out')
os.system('rm GPUACCTEST.out')
os.system('rm CPUACCTEST.out')

print "RUNNING CPU SYSTEM:"
os.system("cmake -DNBODY_OPENCL=OFF -DDOUBLEPREC=ON -DNBODY_OPENMP=ON -DCMAKE_BUILD_TYPE=RELEASE -DBOINC_APPLICATION=OFF")
os.system('make -j 9')
executeString = './bin/milkyway_nbody -f nbody/sample_workunits/for_developers.lua -o CPUBRUTE.out -z CPU.hist -b -i 4 1 .2 .2 12 .2' # >> CPUACCTEST.out'
os.system(executeString)

print "RUNNING GPU SYSTEM:"
os.system("cmake -DNBODY_OPENCL=ON -DDOUBLEPREC=ON -DCMAKE_BUILD_TYPE=RELEASE -DBOINC_APPLICATION=OFF")
os.system('make -j 9')
executeString = './bin/milkyway_nbody -f nbody/sample_workunits/for_developers.lua -o GPUBRUTE.out -z GPU.hist -b -i 4 1 .2 .2 12 .2' # >> GPUACCTEST.out'
os.system(executeString)


# print "==========================FILE DIFF===============================\n"
# os.system('diff CPUACCTEST.out GPUACCTEST.out')
# print "==================================================================\n"
os.system('./bin/milkyway_nbody -h CPU.hist -s GPU.hist')
print "PLOTTING DATA"
os.system('python2 PlotNbody.py GPUBRUTE.out CPUBRUTE.out')