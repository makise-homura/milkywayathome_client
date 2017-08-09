  typedef __global double* restrict RVPtr;
typedef __global volatile uint* restrict UVPtr;
#define EPS2 0.002312
#define WARPSIZE 64
#define EFFNBODY 1024
#define TIMESTEP 1
#define USE_EXTERNAL_POTENTIAL false

inline uint expandBits(uint v){
  v = (v * 0x00010001) & 0xFF0000FF;
  v = (v * 0x00000101) & 0x0F00F00F;
  v = (v * 0x00000011) & 0xC30C30C3;
  v = (v * 0x00000005) & 0x49249249;
  return v;
}

inline uint encodeLocation(double4 pos){
  pos.x = (min(max(pos.x * 1024.0, 0.0), 1023.0));
  pos.y = (min(max(pos.y * 1024.0, 0.0), 1023.0));
  pos.z = (min(max(pos.z * 1024.0, 0.0), 1023.0));

  uint xx = expandBits((uint)pos.x);
  uint yy = expandBits((uint)pos.y);
  uint zz = expandBits((uint)pos.z);

  return xx * 4 + yy * 2 + zz;
}

__kernel void advancePosition(RVPtr x, RVPtr y, RVPtr z,
                              RVPtr vx, RVPtr vy, RVPtr vz,
                              RVPtr ax, RVPtr ay, RVPtr az,
                              RVPtr mass){
  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);

  x[g] = mad(TIMESTEP, vx[g], x[g]);
  y[g] = mad(TIMESTEP, vy[g], y[g]);
  z[g] = mad(TIMESTEP, vz[g], z[g]);
}

__kernel void advanceHalfVelocity(RVPtr x, RVPtr y, RVPtr z,
                                  RVPtr vx, RVPtr vy, RVPtr vz,
                                  RVPtr ax, RVPtr ay, RVPtr az,
                                  RVPtr mass){
  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);

  double dtHalf = 0.5 * TIMESTEP;
  vx[g] = mad(dtHalf, ax[g], vx[g]);
  vy[g] = mad(dtHalf, ay[g], vy[g]);
  vz[g] = mad(dtHalf, az[g], vz[g]);
}

__kernel void boundingBox(RVPtr x, RVPtr y, RVPtr z,
                        RVPtr vx, RVPtr vy, RVPtr vz,
                        RVPtr ax, RVPtr ay, RVPtr az,
                        RVPtr mass, RVPtr xMax, RVPtr yMax,
                        RVPtr zMax, RVPtr xMin, RVPtr yMin,
                        RVPtr zMin, UVPtr mCodes){

  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);

  event_t e[6];

  //Create local variables and copy global data into them:
  __local double maxTemp[3][WARPSIZE];
  __local double minTemp[3][WARPSIZE];
  e[0] = async_work_group_copy(maxTemp[0], xMax + group * WARPSIZE, WARPSIZE, 0);
  e[1] = async_work_group_copy(maxTemp[1], yMax + group * WARPSIZE, WARPSIZE, 0);
  e[2] = async_work_group_copy(maxTemp[2], zMax + group * WARPSIZE, WARPSIZE, 0);

  e[3] = async_work_group_copy(minTemp[0], xMin + group * WARPSIZE, WARPSIZE, 0);
  e[4] = async_work_group_copy(minTemp[1], yMin + group * WARPSIZE, WARPSIZE, 0);
  e[5] = async_work_group_copy(minTemp[2], zMin + group * WARPSIZE, WARPSIZE, 0);

  wait_group_events(6, e);


  // maxTemp[0][l] = xMax[g];
  // maxTemp[1][l] = yMax[g];
  // maxTemp[2][l] = zMax[g];

  // minTemp[0][l] = xMin[g];
  // minTemp[1][l] = yMin[g];
  // minTemp[2][l] = zMin[g];

  // xMax[g] = xMin[g] = x[g];
  // yMax[g] = yMin[g] = y[g];
  // zMax[g] = zMin[g] = z[g];

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);


  int iter = (int)log2((double)WARPSIZE);

  for(int i = 0; i < iter; ++i){
    int expVal = (int)exp2((double)i);
    int nextVal = min((int)l + expVal, (int)WARPSIZE - 1);
    if(l % (expVal) * 2 == 0){
      int gt = maxTemp[0][l] > maxTemp[0][nextVal];
      int lt = minTemp[0][l] < minTemp[0][nextVal];
      maxTemp[0][l] = maxTemp[0][l] * gt + maxTemp[0][nextVal] * (gt^1);
      minTemp[0][l] = minTemp[0][l] * lt + minTemp[0][nextVal] * (lt^1);

      gt = maxTemp[1][l] > maxTemp[1][nextVal];
      lt = minTemp[1][l] < minTemp[1][nextVal];
      maxTemp[1][l] = maxTemp[1][l] * gt + maxTemp[1][nextVal] * (gt^1);
      minTemp[1][l] = minTemp[1][l] * lt + minTemp[1][nextVal] * (lt^1);

      gt = maxTemp[2][l] > maxTemp[2][nextVal];
      lt = minTemp[2][l] < minTemp[2][nextVal];
      maxTemp[2][l] = maxTemp[2][l] * gt + maxTemp[2][nextVal] * (gt^1);
      minTemp[2][l] = minTemp[2][l] * lt + minTemp[2][nextVal] * (lt^1);
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  }


  //Copy back from local memory to global memory:

  e[0] = async_work_group_copy(xMax + group, maxTemp[0], 1, 0);
  e[1] = async_work_group_copy(yMax + group, maxTemp[1], 1, 0);
  e[2] = async_work_group_copy(zMax + group, maxTemp[2], 1, 0);

  e[3] = async_work_group_copy(xMin + group, minTemp[0], 1, 0);
  e[4] = async_work_group_copy(yMin + group, minTemp[1], 1, 0);
  e[5] = async_work_group_copy(zMin + group, minTemp[2], 1, 0);
  wait_group_events(6, e);

  // if(l == 0){
  //   xMax[group] = maxTemp[0][l];
  //   xMin[group] = minTemp[0][l];
  //   yMax[group] = maxTemp[1][l];
  //   yMin[group] = minTemp[1][l];
  //   zMax[group] = maxTemp[2][l];
  //   zMin[group] = minTemp[2][l];
  // }
  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);


}

__kernel void forceCalculationExact(RVPtr x, RVPtr y, RVPtr z,
                                    RVPtr vx, RVPtr vy, RVPtr vz,
                                    RVPtr ax, RVPtr ay, RVPtr az,
                                    RVPtr mass){
  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);


  __local double posX[WARPSIZE];
  __local double posY[WARPSIZE];
  __local double posZ[WARPSIZE];
  __local double accTempX[WARPSIZE];
  __local double accTempY[WARPSIZE];
  __local double accTempZ[WARPSIZE];

  __private double4 particle;
  __private double4 accPrivate;
  __private double4 drVec;

  __private double dr2;
  __private double dr;
  __private double m2;
  __private double ai;

  event_t e[3];


  accTempX[l] = 0;
  accTempY[l] = 0;
  accTempZ[l] = 0;

  particle.x = x[g];
  particle.y = y[g];
  particle.z = z[g];

  barrier(CLK_LOCAL_MEM_FENCE);
  for(int i = 0; i < EFFNBODY/WARPSIZE; ++i){
     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
     e[0] = async_work_group_copy(posX, x+i*WARPSIZE, WARPSIZE, 0);
     e[1] = async_work_group_copy(posY, y+i*WARPSIZE, WARPSIZE, 0);
     e[2] = async_work_group_copy(posZ, z+i*WARPSIZE, WARPSIZE, 0);
     wait_group_events(3, e);
    for(int j = 0; j < WARPSIZE; ++j){
      drVec.x = posX[j] - particle.x;
      drVec.y = posY[j] - particle.y;
      drVec.z = posZ[j] - particle.z;
      dr2 = mad(drVec.z, drVec.z, mad(drVec.y, drVec.y, mad(drVec.x, drVec.x,EPS2)));
      dr = sqrt(dr2);
      m2 = mass[j];
      ai = m2/(dr*dr2);
      accTempX[l] += ai * drVec.x;
      accTempY[l] += ai * drVec.y;
      accTempZ[l] += ai * drVec.z;
    }
  }
  if(USE_EXTERNAL_POTENTIAL)
  {
    //double4 externAcc = externalAcceleration(particle.x, particle.y, particle.z);
    accTempX[l] += 1/*externAcc.x*/;
    accTempY[l] += 1/*externAcc.y*/;
    accTempZ[l] += 1/*externAcc.z*/;
  }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    e[0] = async_work_group_copy(ax + group * WARPSIZE, accTempX, WARPSIZE, 0);
    e[1] = async_work_group_copy(ay + group * WARPSIZE, accTempY, WARPSIZE, 0);
    e[2] = async_work_group_copy(az + group * WARPSIZE, accTempZ, WARPSIZE, 0);
    wait_group_events(3, e);
}

__kernel void encodeTree(RVPtr x, RVPtr y, RVPtr z,
                        RVPtr vx, RVPtr vy, RVPtr vz,
                        RVPtr ax, RVPtr ay, RVPtr az,
                        RVPtr mass, RVPtr xMax, RVPtr yMax,
                        RVPtr zMax, RVPtr xMin, RVPtr yMin,
                        RVPtr zMin, UVPtr mCodes_G){

  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);


  __local double4 pos_local[WARPSIZE];
  __local uint mCodes_L[WARPSIZE];


  pos_local[l].x = (x[g] - xMin[0])/(xMax[0]-xMin[0]);
  pos_local[l].y = (y[g] - yMin[0])/(yMax[0]-yMin[0]);
  pos_local[l].z = (z[g] - zMin[0])/(zMax[0]-zMin[0]);

  //CALCULATE MORTON CODE
  mCodes_L[l] = encodeLocation(pos_local[l]);

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  if(l == 0){
    for(int i = 0; i < WARPSIZE; ++i){
      mCodes_G[group * WARPSIZE + i] = mCodes_L[i];
    }
  }


  //Use global thread ID as a LSB identifier to seperate morton code collisions.
}
