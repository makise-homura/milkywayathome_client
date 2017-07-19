#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pylab as lab
import subprocess
import math as ma

def convert_to_Lambda_Beta(x1, x2, x3, cartesian):
    phi   = mt.radians(128.79)
    theta = mt.radians(54.39)
    psi   = mt.radians(90.70)

    if(cartesian):
        x_coor = x1
        y_coor = x2
        z_coor = x3
        x_coor += 8.0 #convert to solar centric
    else:
        l = mt.radians(x1)
        b = mt.radians(x2)
        r = x3

        x_coor = r * mt.cos(l) * mt.cos(b) #this is solar centered x
        y_coor = r * mt.sin(l) * mt.cos(b)
        z_coor = r * mt.sin(b)

    #A = MB
    B = [x_coor, y_coor, z_coor]
    M_row1 = [mt.cos(psi) * mt.cos(phi) - mt.cos(theta) * mt.sin(phi) * mt.sin(psi),
               mt.cos(psi) * mt.sin(phi) + mt.cos(theta) * mt.cos(phi) * mt.sin(psi),
               mt.sin(psi) * mt.sin(theta)]

    M_row2 = [-mt.sin(psi) * mt.cos(phi) - mt.cos(theta) * mt.sin(phi) * mt.cos(psi),
               -mt.sin(psi) * mt.sin(phi) + mt.cos(theta) * mt.cos(phi) * mt.cos(psi),
               mt.cos(psi) * mt.sin(theta)]

    M_row3 = [mt.sin(theta) * mt.sin(phi),
               -mt.sin(theta) * mt.cos(phi),
               mt.cos(theta)]

    A1 = M_row1[0] * B[0] + M_row1[1] * B[1] + M_row1[2] * B[2]
    A2 = M_row2[0] * B[0] + M_row2[1] * B[1] + M_row2[2] * B[2]
    A3 = M_row3[0] * B[0] + M_row3[1] * B[1] + M_row3[2] * B[2]

    beta = mt.asin(-A3 / mt.sqrt(A1 * A1 + A2 * A2 + A3 * A3))
    lamb = mt.atan2(A2, A1)

    beta = mt.degrees(beta)
    lamb = mt.degrees(lamb)

    return lamb, beta



input = sys.argv[1];
g = open(input, 'r');
x1 = [];
y1 = [];
z1 = [];
vx1 = [];
vy1 = [];
vz1 = [];
bID1 = [];
g.next()
g.next()
g.next()
g.next()
g.next()
for line in g:
 ln = line.split(',');
 x1.append(float(ln[1]));
 y1.append(float(ln[2]));
 z1.append(float(ln[3]));
 vx1.append(float(ln[4]));
 vy1.append(float(ln[5]));
 vz1.append(float(ln[6]));
 bID1.append(int(ln[12]));

x1s = [None]*len(bID1);
y1s = [None]*len(bID1);
minB = len(x1)
print str(len(x1)) + ' ' + str(len(y1)) + ' ' + str(len(bID1))
for i in range(len(x1)):
	x1s[bID1[i]] = x1[i];
	y1s[bID1[i]] = y1[i];


plt.subplot(421)
plt.plot(x1s,y1s, 'ob', ms=1, label="GPU Data")
# plt.xlim([-60,40])
# plt.ylim([-100,100])
legend = plt.legend(loc="upper right")



input = sys.argv[2];
f = open(input, 'r');
x2 = [];
y2 = [];
z2 = [];
vx2 = [];
vy2 = [];
vz2 = [];
bID2 = [];

f.next()
f.next()
f.next()
f.next()
f.next()
for line in f:
 ln = line.split(',');
 x2.append(float(ln[1]));
 y2.append(float(ln[2]));
 z2.append(float(ln[3]));
 vx2.append(float(ln[4]));
 vy2.append(float(ln[5]));
 vz2.append(float(ln[6]));
 bID2.append(int(ln[12]));

x2s = [None]*len(bID2);
y2s = [None]*len(bID2);
for i in range(len(x2)):
	x2s[bID2[i]] = x2[i];
	y2s[bID2[i]] = y2[i];

print "Number of bodies in x list: " + str(len(x2))
print "Number of bodies in y list: " + str(len(y2))


plt.subplot(422)
plt.plot(x2,y2, 'ob', ms=1, label="CPU Data");
# plt.xlim([-100,100])
# plt.ylim([-100,100])
legend = plt.legend(loc="upper right")
x3 = []
y3 = []
for i in range(len(x2s)):
	x3.append(x2s[i]-x1s[i])
	y3.append(y2s[i]-y1s[i])

plt.subplot(423)
plt.plot(x3,y3, 'ob', label="Residual");
legend = plt.legend(loc="upper right")

r = []
for i in range(len(x3)):
	r.append(ma.sqrt((x3[i]*x3[i])+(y3[i]*y3[i])))
sum = 0;
for elem in r:
	sum += elem;

averageResidual = sum/(len(r)*1.0)
print "AVERAGE RESIDUAL: " + str(averageResidual) + "\n"
plt.subplot(424)
binwidth = (max(r)-min(r))/50.0
if(binwidth < .001):
	binwidth = .001
plt.hist(r, bins=np.arange(min(r), max(r) + binwidth, binwidth), label="Residual");
legend = plt.legend(loc="upper right")



#Histograms:
input = sys.argv[3]
f = open(input, 'r');


l1 = []
b1 = []
value1 = []

for line in f:
    ln = line.split(' ')
    # print line
    if(ln[0] == '1'):
        l1.append(float(ln[1]))
        b1.append(float(ln[2]))
        value1.append(float(ln[3]))

barwidth = l1[1] - l1[0]
plt.subplot(425)
plt.bar(l1, value1, barwidth, align='center', label='GPU Histogram')
legend = plt.legend(loc="upper right")
f.close()


input = sys.argv[4]
f = open(input, 'r');


l2 = []
b2 = []
value2 = []

for line in f:
    ln = line.split(' ')
    if(ln[0] == '1'):
        l2.append(float(ln[1]))
        b2.append(float(ln[2]))
        value2.append(float(ln[3]))

barwidth = l2[1] - l2[0]
plt.subplot(426)
plt.bar(l2, value2, barwidth, align='center', label='CPU Histogram')
legend = plt.legend(loc="upper right")
f.close()

plt.subplot(427)
plt.bar(l1, value1, barwidth, align='center', color='orange', label='GPU')
plt.bar(l2, value2, barwidth, align='center', color='blue', label='CPU')
legend = plt.legend(loc="upper right")


plt.subplot(428)
plt.bar(l2, value2, barwidth, align='center', color='blue', label='CPU')
plt.bar(l1, value1, barwidth, align='center', color='orange', label='GPU')
legend = plt.legend(loc="upper right")

plt.show();
