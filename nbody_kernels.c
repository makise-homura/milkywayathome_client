#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


#define NBODIES (1024)
#define DWARFMASS (12)
#define MAX_SOURCE_SIZE (0x100000)
//kernels to test
//NBODY_KERNEL
//forceCalculationExact
//advanceHalfVelocity
//advancePosition
//boundingBox
//encodeTree


int main(int argc, char *argv[]) {
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobj = NULL;
  cl_program program = NULL;
  cl_kernel* kernel = (cl_kernel *) malloc((argc-1)*sizeof(cl_kernel));
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_uint ret;
  cl_uint address_bits;
  size_t size;
  /*staticBodies formatting:
  x
  y
  z
  vx
  vy
  vz
  */
  double* x = (double *) malloc(NBODIES *sizeof(double));
  double* y = (double *) malloc(NBODIES *sizeof(double));
  double* z = (double *) malloc(NBODIES *sizeof(double));
  double* vx = (double *) malloc(NBODIES *sizeof(double));
  double* vy = (double *) malloc(NBODIES *sizeof(double));
  double* vz = (double *) malloc(NBODIES *sizeof(double));
  double* ax = (double *) malloc(NBODIES *sizeof(double));
  double* ay = (double *) malloc(NBODIES *sizeof(double));
  double* az = (double *) malloc(NBODIES *sizeof(double));
  double* mass = (double *) malloc(NBODIES *sizeof(double));
  double* xMax = (double *) malloc(NBODIES *sizeof(double));
  double* yMax = (double *) malloc(NBODIES *sizeof(double));
  double* xMin = (double *) malloc(NBODIES *sizeof(double));
  double* zMax = (double *) malloc(NBODIES *sizeof(double));
  double* yMin = (double *) malloc(NBODIES *sizeof(double));
  double* zMin = (double *) malloc(NBODIES *sizeof(double));
  uint* mCodes = (uint *) malloc(NBODIES *sizeof(uint));

  char const* const Bodies = "staticBodies.txt";
  FILE* file = fopen(Bodies, "r"); /* should check the result */
  char line[256];
  int i = 0;
  while (fgets(line, sizeof(line), file)) {
      /* note that fgets don't strip the terminating \n, checking its
         presence would allow to handle lines longer that sizeof(line) */
      x[i] = (double) atof(line);
      fgets(line, sizeof(line), file);
      y[i] = (double) atof(line);
      fgets(line, sizeof(line), file);
      z[i] = (double) atof(line);
      fgets(line, sizeof(line), file);
      vx[i] = (double) atof(line);
      fgets(line, sizeof(line), file);
      vy[i] = (double) atof(line);
      fgets(line, sizeof(line), file);
      vz[i] = (double) atof(line);
      ax[i] = 0.0;
      ay[i] = 0.0;
      az[i] = 0.0;
      mass[i] = (double) DWARFMASS/NBODIES;
      i++;
  }
  fclose(file);
  memcpy(xMax, x, NBODIES *sizeof(double));
  memcpy(yMax, y, NBODIES *sizeof(double));
  memcpy(zMax, z, NBODIES *sizeof(double));
  memcpy(xMin, x, NBODIES *sizeof(double));
  memcpy(yMin, y, NBODIES *sizeof(double));
  memcpy(zMin, z, NBODIES *sizeof(double));





  FILE *fp;
  char fileName[] = "./all_kernels.cl";
  char *source_str;

  size_t source_size;

  fp = fopen(fileName, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose(fp);

  /*Get Devices and Platforms, along with ids*/
  ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  /*creates opencl context*/
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

  command_queue = clCreateCommandQueue(context, device_id,  0, &ret);

  /*Memory Buffers*/
  cl_mem memobjx = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjy = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjz = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjvx = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjvy = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjvz = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjax = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjay = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjaz = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjmass = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjxMax = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjyMax = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjzMax = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjxMin = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjyMin = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjzMin = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(double), NULL, &ret);
  cl_mem memobjmCodes = clCreateBuffer(context, CL_MEM_READ_WRITE,NBODIES * sizeof(uint), NULL, &ret);





  program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);


  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  /* USED FOR PRINTING ERROR LOGS. KEEP AROUND FOR FUTURE FAILURES*/

  if (ret == CL_BUILD_PROGRAM_FAILURE) {
    printf("oops\n");
    // Determine the size of the log
   size_t log_size;
   clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

   // Allocate memory for the log
   char *log = (char *) malloc(log_size);

   // Get the log
   clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

   // Print the log
   printf("%s\n", log);
  }

  for (int i = 0; i < argc-1; i++) {
    /*copy data from arrays to memobjects*/
    ret = clEnqueueWriteBuffer(command_queue, memobjx, CL_TRUE, 0, NBODIES *sizeof(double), x, 0, NULL, NULL);
  	ret = clEnqueueWriteBuffer(command_queue, memobjy, CL_TRUE, 0, NBODIES *sizeof(double), y, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjz, CL_TRUE, 0, NBODIES *sizeof(double), z, 0, NULL, NULL);
  	ret = clEnqueueWriteBuffer(command_queue, memobjvx, CL_TRUE, 0, NBODIES *sizeof(double), vx, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjvy, CL_TRUE, 0, NBODIES *sizeof(double), vy, 0, NULL, NULL);
  	ret = clEnqueueWriteBuffer(command_queue, memobjvz, CL_TRUE, 0, NBODIES *sizeof(double), vz, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjax, CL_TRUE, 0, NBODIES *sizeof(double), ax, 0, NULL, NULL);
  	ret = clEnqueueWriteBuffer(command_queue, memobjay, CL_TRUE, 0, NBODIES *sizeof(double), ay, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjaz, CL_TRUE, 0, NBODIES *sizeof(double), az, 0, NULL, NULL);
  	ret = clEnqueueWriteBuffer(command_queue, memobjmass, CL_TRUE, 0, NBODIES *sizeof(double), mass, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjxMax, CL_TRUE, 0, NBODIES *sizeof(double), xMax, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjyMax, CL_TRUE, 0, NBODIES *sizeof(double), yMax, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjzMax, CL_TRUE, 0, NBODIES *sizeof(double), zMax, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjxMin, CL_TRUE, 0, NBODIES *sizeof(double), xMin, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjyMin, CL_TRUE, 0, NBODIES *sizeof(double), yMin, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjzMin, CL_TRUE, 0, NBODIES *sizeof(double), zMin, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, memobjmCodes, CL_TRUE, 0, NBODIES *sizeof(uint), mCodes, 0, NULL, NULL);

    kernel[i] = clCreateKernel(program, argv[i+1], &ret);
    /*SET KERNEL ARGS*/
    ret = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *)&memobjx);
    ret = clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *)&memobjy);
    ret = clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void *)&memobjz);
    ret = clSetKernelArg(kernel[i], 3, sizeof(cl_mem), (void *)&memobjvx);
    ret = clSetKernelArg(kernel[i], 4, sizeof(cl_mem), (void *)&memobjvy);
    ret = clSetKernelArg(kernel[i], 5, sizeof(cl_mem), (void *)&memobjvz);
    ret = clSetKernelArg(kernel[i], 6, sizeof(cl_mem), (void *)&memobjax);
    ret = clSetKernelArg(kernel[i], 7, sizeof(cl_mem), (void *)&memobjay);
    ret = clSetKernelArg(kernel[i], 8, sizeof(cl_mem), (void *)&memobjaz);
    ret = clSetKernelArg(kernel[i], 9, sizeof(cl_mem), (void *)&memobjmass);
    if (strcmp(argv[i+1],"boundingBox") == 0 || strcmp(argv[i+1],"encodeTree") == 0) {
      ret = clSetKernelArg(kernel[i], 10, sizeof(cl_mem), (void *)&memobjxMax);
      ret = clSetKernelArg(kernel[i], 11, sizeof(cl_mem), (void *)&memobjyMax);
      ret = clSetKernelArg(kernel[i], 12, sizeof(cl_mem), (void *)&memobjzMax);
      ret = clSetKernelArg(kernel[i], 13, sizeof(cl_mem), (void *)&memobjxMin);
      ret = clSetKernelArg(kernel[i], 14, sizeof(cl_mem), (void *)&memobjyMin);
      ret = clSetKernelArg(kernel[i], 15, sizeof(cl_mem), (void *)&memobjzMin);
      ret = clSetKernelArg(kernel[i], 16, sizeof(cl_mem), (void *)&memobjmCodes);
    }






  clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, 0, NULL, &size);
  clGetDeviceInfo(device_id, CL_DEVICE_ADDRESS_BITS, size, (void *)&address_bits, NULL);
  size_t temp = address_bits;
    if (strcmp(argv[i+1],"boundingBox") == 0) {
      for (int j = 0; j < ceil(log2(NBODIES)); j++) {
        ret = clEnqueueNDRangeKernel(command_queue, kernel[i], 1, NULL, &temp, NULL, 0, NULL,NULL);
      }
    } else {
      ret = clEnqueueNDRangeKernel(command_queue, kernel[i], 1, NULL, &temp, NULL, 0, NULL,NULL);
    }


    printf("==================\n");
    printf("%s\n", argv[i+1]);
    printf("==================\n");

    printf("particle 1: \n");
    printf("x: %.15f, y: %.15f, z: %.15f\nvx: %.15f, vy: %.15f, vz: %.15f\nax: %.15f, ay: %.15f, az: %.15f\n\n", x[0],y[0],z[0],vx[0],vy[0],vz[0],ax[0],ay[0],az[0]);
    printf("particle 2: \n");
    printf("x: %.15f, y: %.15f, z: %.15f\nvx: %.15f, vy: %.15f, vz: %.15f\nax: %.15f, ay: %.15f, az: %.15f\n\n", x[1],y[1],z[1],vx[1],vy[1],vz[1],ax[1],ay[1],az[1]);

    /*copy info from buffers to local arrays*/
    ret = clEnqueueReadBuffer(command_queue, memobjx, CL_TRUE, 0, NBODIES *sizeof(double), (void *)x, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjy, CL_TRUE, 0, NBODIES *sizeof(double), (void *)y, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjz, CL_TRUE, 0, NBODIES *sizeof(double), (void *)z, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjvx, CL_TRUE, 0, NBODIES *sizeof(double), (void *)vx, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjvy, CL_TRUE, 0, NBODIES *sizeof(double), (void *)vy, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjvz, CL_TRUE, 0, NBODIES *sizeof(double), (void *)vz, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjax, CL_TRUE, 0, NBODIES *sizeof(double), (void *)ax, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjay, CL_TRUE, 0, NBODIES *sizeof(double), (void *)ay, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjaz, CL_TRUE, 0, NBODIES *sizeof(double), (void *)az, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, memobjmass, CL_TRUE, 0, NBODIES *sizeof(double), (void *)mass, 0, NULL, NULL);
    if (strcmp(argv[i+1],"boundingBox") == 0 || strcmp(argv[i+1],"encodeTree") == 0) {
      ret = clEnqueueReadBuffer(command_queue, memobjxMax, CL_TRUE, 0, NBODIES *sizeof(double), (void *)xMax, 0, NULL, NULL);
      ret = clEnqueueReadBuffer(command_queue, memobjyMax, CL_TRUE, 0, NBODIES *sizeof(double), (void *)yMax, 0, NULL, NULL);
      ret = clEnqueueReadBuffer(command_queue, memobjzMax, CL_TRUE, 0, NBODIES *sizeof(double), (void *)zMax, 0, NULL, NULL);
      ret = clEnqueueReadBuffer(command_queue, memobjxMin, CL_TRUE, 0, NBODIES *sizeof(double), (void *)xMin, 0, NULL, NULL);
      ret = clEnqueueReadBuffer(command_queue, memobjyMin, CL_TRUE, 0, NBODIES *sizeof(double), (void *)yMin, 0, NULL, NULL);
      ret = clEnqueueReadBuffer(command_queue, memobjzMin, CL_TRUE, 0, NBODIES *sizeof(double), (void *)zMin, 0, NULL, NULL);
      ret = clEnqueueReadBuffer(command_queue, memobjmCodes, CL_TRUE, 0, NBODIES *sizeof(uint), (void *)mCodes, 0, NULL, NULL);
    }
    //ret = clFlush(command_queue);



    printf("particle 1: \n");
    printf("x: %.15f, y: %.15f, z: %.15f\nvx: %.15f, vy: %.15f, vz: %.15f\nax: %.15f, ay: %.15f, az: %.15f\n\n", x[0],y[0],z[0],vx[0],vy[0],vz[0],ax[0],ay[0],az[0]);
    printf("particle 2: \n");
    printf("x: %.15f, y: %.15f, z: %.15f\nvx: %.15f, vy: %.15f, vz: %.15f\nax: %.15f, ay: %.15f, az: %.15f\n\n", x[1],y[1],z[1],vx[1],vy[1],vz[1],ax[1],ay[1],az[1]);
    if (strcmp(argv[i+1],"boundingBox") == 0) {
      printf("xMin: %.15f, yMin: %.15f, zMin: %.15f\nxMax: %.15f, yMax: %.15f, zMax: %.15f\n\n", xMin[0],yMin[0],zMin[0],xMax[0],yMax[0],zMax[0]);
    }
    if (strcmp(argv[i+1],"encodeTree") == 0) {
      printf("%i %i %i\n",mCodes[0],mCodes[1],mCodes[2]);
    }
  }


  /*cleanup*/
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  for (int i = 0; i < argc-1; i++) {ret = clReleaseKernel(kernel[i]);}
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(memobjx);
  ret = clReleaseMemObject(memobjy);
  ret = clReleaseMemObject(memobjz);
  ret = clReleaseMemObject(memobjvx);
  ret = clReleaseMemObject(memobjvy);
  ret = clReleaseMemObject(memobjvz);
  ret = clReleaseMemObject(memobjax);
  ret = clReleaseMemObject(memobjay);
  ret = clReleaseMemObject(memobjaz);
  ret = clReleaseMemObject(memobjmass);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(source_str);

  return 0;
}
