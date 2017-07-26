import os
import sys

def isFloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

#os.system('python2 RunAndPlotNbody.py')
timesteps = 50000
#removes old files
os.system('rm GPUBRUTE.out')
os.system('rm CPUBRUTE.out')
os.system('rm GPUACCTEST.out')
os.system('rm CPUACCTEST.out')


#gets the arguments
#'g' for just gpu, 'c' for just cpu, 'h' for historgram printing, and nothing for the full test
choice = sys.argv
gcom, ccom = [], []
#force sets the arguments if none are given
if len(choice) == 1:
    choice.append('g')
    choice.append('c')
    choice.append('h')
    choice.append('b')

#runs the cpu calculations
if 'c' in choice:
    print "RUNNING CPU SYSTEM:"
    os.system("cmake -DNBODY_OPENCL=OFF -DDOUBLEPREC=ON -DNBODY_OPENMP=ON -DCMAKE_BUILD_TYPE=RELEASE")
    os.system('make -j 9')
    executeString = './bin/milkyway_nbody -f nbody/sample_workunits/for_developers.lua -o CPUBRUTE.out -z CPU.hist -b -i 4 1 .2 .2 12 .2' # >> CPUACCTEST.out'
    os.system(executeString)

    #opens and reads through the histogram files
    #gets the lambda and probability values
    if 'h' in choice:
        chist = open('CPU.hist', 'r')
        cpu = []

        for line in chist:
            temp = line.split(" ")
            if temp[0] == "1":
                cpu.append([temp[1], temp[3]])

    if 'b' in choice:
        cbruteout = open('CPUBRUTE.out', 'r')
        cpbrute = []
        next(cbruteout)
        next(cbruteout)
        next(cbruteout)
        cpbrute.append(next(cbruteout).replace(',\n',''))
        temp = cpbrute[0].split(" ")
        for x in range(len(temp)):
            temp[x] = temp[x].replace(',','')
            if isFloat(temp[x]):
                ccom.append(float(temp[x]))
        next(cbruteout)
        #cpbrute.append(next(cbruteout).replace('\n',''))
        for line in cbruteout:
            line = line.replace(' ','').replace('\n','')
            line = line.split(',')
            line = list(map(float, line))
            cpbrute.append(line)
        #print cpbrute

#runs gpu calculations
if 'g' in choice:
    print "RUNNING GPU SYSTEM:"
    os.system("cmake -DNBODY_OPENCL=ON -DDOUBLEPREC=ON -DCMAKE_BUILD_TYPE=RELEASE ")
    os.system('make -j 9')
    executeString = './bin/milkyway_nbody -f nbody/sample_workunits/for_developers.lua -o GPUBRUTE.out -z GPU.hist -b -i 4 1 .2 .2 12 .2' # >> GPUACCTEST.out'
    os.system(executeString)

    if 'h' in choice:
        ghist = open('GPU.hist', 'r')
        gpu = []
        #gets the lambda and probability values
        for line in ghist:
            temp = line.split(" ")
            if temp[0] == "1":
                gpu.append([temp[1], temp[3]])

    if 'b' in choice:
        gbruteout = open('GPUBRUTE.out', 'r')
        gpbrute = []
        next(gbruteout)
        next(gbruteout)
        next(gbruteout)
        gpbrute.append(next(gbruteout).replace(',\n',''))
        temp = gpbrute[0].split(" ")
        for x in range(len(temp)):
            temp[x] = temp[x].replace(',','')
            if isFloat(temp[x]):
                gcom.append(float(temp[x]))
        next(gbruteout)
        #gpbrute.append(next(gbruteout).replace('\n',''))
        for line in gbruteout:
            line = line.replace(' ','').replace('\n','')
            line = line.split(',')
            line = list(map(float, line))
            gpbrute.append(line)

#prints out the lambdas and corresponding probability depending on the arguments
if 'h' in choice:
    if 'g' in choice and 'c' in choice:
        for val in range(len(cpu)):
            print "Lambda: {:>20}, CPU Prob.: {}, GPU Prob.: {}, Difference: {:>18} \
            ".format(cpu[val][0], cpu[val][1], gpu[val][1], float(cpu[val][1]) - float(gpu[val][1]))
    elif 'g' in choice:
        for val in range(len(gpu)):
            print "Lambda: {:>20} , GPU Probability: {}".format(gpu[val][0], gpu[val][1])
    elif 'c' in choice:
        for val in range(len(cpu)):
            print "Lambda: {:>20} , CPU Probability: {}".format(cpu[val][0], cpu[val][1])

#printing for BRUTEFORCE
if 'b' in choice:
    output = open("BRUTFORCE.out","w")
    if 'g' in choice and 'c' in choice:
        output.write("CPU: {}\n".format(cpbrute[0]))
        output.write("GPU: {}\n".format(gpbrute[0]))
        difference = []
        for x in range(len(ccom)):
            difference.append(abs(abs(ccom[x]) - abs(gcom[x])))
        output.write("     centerOfMass = {:>10}, {:>9}, {:>9},   centerOfMomentum = {:>11}, {:>9}, {:>10}\n".format(*difference))
        #headers = cpbrute[1].split(' ')
        #headers = [s.strip(" ") for s in headers]
        top = ""
        headers = ['x', 'y', 'z', 'l', 'b', 'r', 'v_x', 'v_y', 'v_z', 'mass', 'v_los', 'BodyID']
        for x in range(len(headers)):
            t = headers[x]
            #print t
            top += "{:>20}".format(t)
        output.write(top)
        temp = "\n"
        for body in range(2,len(cpbrute)):
            for val in range(1, len(cpbrute[body])):
                if val < len(cpbrute[body]) - 1:
                    temp += "{:>20}".format(str(cpbrute[body][val] - gpbrute[body][val]))
                else:
                    temp += "{:>20}".format(str(cpbrute[body][val]))
            temp += "\n"
        output.write(temp)
        output.close()
'''
os.system('./bin/milkyway_nbody -h CPU.hist -s GPU.hist')
print "PLOTTING DATA"
os.system('python2 PlotNbodyResidualHistograms.py GPUBRUTE.out CPUBRUTE.out GPU.hist CPU.hist')
'''
