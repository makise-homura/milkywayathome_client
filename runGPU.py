#!/usr/bin/python
import os
import sys
import subprocess

#./bin/milkyway_nbody -f nbody/sample_workunits/EMD_10k_plummer.lua -o CPUBRUTE.out -z CPU.hist -x -i -e 36912 1 1 .2 12
timesteps = 1

if("-b" in sys.argv):
    os.system('rm -r ../build')
    os.system('mkdir ../build')

os.system('rm -r ../results')
os.system('mkdir ../results')
os.system('rm GPUData.out')
os.system('rm GPUACCTEST.out')

# print "RUNNING GPU SYSTEM:"
os.chdir('../build')
if("-b" in sys.argv):
    os.system("cmake ../milkywayathome_client -DBOINC_APPLICATION=OFF -DNBODY_OPENCL=ON -DDOUBLEPREC=ON -DCMAKE_BUILD_TYPE=RELEASE ")
    os.system('make -j 9')
if("-m" in sys.argv):
    os.system('make -j 9')
executeString = './bin/milkyway_nbody -f ../milkywayathome_client/nbody/sample_workunits/for_developers.lua -o GPUData.out -z GPU.hist -b -i 4 1 .2 .2 32 .2' #>> GPUACCTEST.out'
os.system(executeString)
os.chdir('../milkywayathome_client')
executeString = 'python2 PlotSingleNbody.py GPUData.out'
# os.system()