import os
import sys

kernels = ['advancePosition', 'advanceHalfVelocity', 'forceCalculationExact', 'boundingBox', 'encodeTree']

print "Type the numbers of the kernels you wish to run, all on one line, no spaces"
print "Kernels will run in the order of the numbers"

for i in range(len(kernels)):
    print "{}: {}".format(i, kernels[i])

choices = raw_input()

finalkern = ''

for i in choices:
    finalkern += kernels[int(i)]
    finalkern += ' '
print finalkern
os.system('gcc nbody_kernels.c -o kernels -lOpenCL')
os.system('./kernels {}'.format(finalkern))
