/*
 * Copyright (c) 2010 The University of Texas at Austin
 * Copyright (c) 2010 Dr. Martin Burtscher
 * Copyright (c) 2011-2012 Matthew Arsenault
 * Copyright (c) 2011 Rensselaer Polytechnic Institute
 *
 * This file is part of Milkway@Home.
 *
 * Milkyway@Home is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Milkyway@Home is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
 */

/* In case there isn't a space a space between the -D and the
 * symbol. if the thing begins with D there's an Apple OpenCL compiler
 * bug on 10.6 where the D will be stripped. -DDOUBLEPREC=1 will
 * actually define OUBLEPREC */

#ifdef OUBLEPREC
  #define DOUBLEPREC OUBLEPREC
#endif

#ifdef EBUG
  #define DEBUG EBUG
#endif

#ifdef ISK_MASS
  #define DISK_MASS ISK_MASS
#endif

#ifdef ISK_SCALE_LENGTH
  #define DISK_SCALE_LENGTH ISK_SCALE_LENGTH
#endif

#ifdef ISK_SCALE_HEIGHT
  #define DISK_SCALE_HEIGHT ISK_SCALE_HEIGHT
#endif


#ifndef DOUBLEPREC
  #error Precision not defined
#endif


#if !BH86 && !SW93 && !NEWCRITERION && !EXACT
  #error Opening criterion not set
#endif

#if USE_EXTERNAL_POTENTIAL && ((!MIYAMOTO_NAGAI_DISK && !EXPONENTIAL_DISK) || (!LOG_HALO && !NFW_HALO && !TRIAXIAL_HALO))
  #error Potential defines misspecified
#endif

#if WARPSIZE <= 0
  #error Invalid warp size
#endif

/* These were problems when being lazy and writing it */
#if (THREADS6 / WARPSIZE) <= 0
  #error (THREADS6 / WARPSIZE) must be > 0
#elif (MAXDEPTH * THREADS6 / WARPSIZE) <= 0
  #error (MAXDEPTH * THREADS6 / WARPSIZE) must be > 0
#endif

#if DEBUG && cl_amd_printf
  #pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#if DOUBLEPREC
  /* double precision is optional core feature in 1.2, not an extension */
  #if __OPENCL_VERSION__ < 120
    #if cl_khr_fp64
      #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #elif cl_amd_fp64
      #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #else
      #error Missing double precision extension
    #endif
  #endif
#endif /* DOUBLEPREC */


#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable


/* Reserve positive numbers for reporting depth > MAXDEPTH. Should match on host */
typedef enum
{
    NBODY_KERNEL_OK                   = 0,
    NBODY_KERNEL_CELL_OVERFLOW        = -1,
    NBODY_KERNEL_TREE_INCEST          = -2,
    NBODY_KERNEL_TREE_STRUCTURE_ERROR = -3,
    NBODY_KERNEL_ERROR_OTHER          = -4
} NBodyKernelError;

#if DEBUG
/* Want first failed assertion to be where it is marked */
#define cl_assert(treeStatus, x)                        \
    do                                                  \
    {                                                   \
      if (!(x))                                         \
      {                                                 \
          if ((treeStatus)->assertionLine < 0)          \
          {                                             \
              (treeStatus)->assertionLine = __LINE__;   \
          }                                             \
      }                                                 \
    }                                                   \
    while (0)

#define cl_assert_rtn(treeStatus, x)                    \
    do                                                  \
    {                                                   \
      if (!(x))                                         \
      {                                                 \
          if ((treeStatus)->assertionLine < 0)          \
          {                                             \
              (treeStatus)->assertionLine = __LINE__;   \
          }                                             \
          return;                                       \
      }                                                 \
    }                                                   \
    while (0)

#else
#define cl_assert(treeStatus, x)
#define cl_assert_rtn(treeStatus, x)
#endif /* DEBUG */


#if DOUBLEPREC
typedef double real;
typedef double2 real2;
typedef double4 real4;
#else
typedef float real;
typedef float2 real2;
typedef float4 real4;
#endif /* DOUBLEPREC */


#if DOUBLEPREC
  #define REAL_EPSILON DBL_EPSILON
  #define REAL_MAX DBL_MAX
  #define REAL_MIN DBL_MIN
#else
  #define REAL_EPSILON FLT_EPSILON
  #define REAL_MAX FLT_MAX
  #define REAL_MIN FLT_MIN
#endif


#define sqr(x) ((x) * (x))
#define cube(x) ((x) * (x) * (x))

#define NSUB 8

#define isBody(n) ((n) < NBODY)
#define isCell(n) ((n) >= NBODY)

#define NULL_BODY (-1)
#define LOCK (-2)


/* This needs to be the same as on the host */
typedef struct __attribute__((aligned(64)))
{
    real radius;
    int bottom;
    uint maxDepth;
    uint blkCnt;
    int doneCnt;

    int errorCode;
    int assertionLine;

    char _pad[64 - (1 * sizeof(real) + 6 * sizeof(int))];

    struct
    {
        real f[32];
        int i[64];
        int wg1[256];
        int wg2[256];
        int wg3[256];
        int wg4[256];
    } debug;
} TreeStatus;


typedef struct
{
    real xx, xy, xz;
    real yy, yz;
    real zz;
} QuadMatrix;

//Structure that holds flattened GPU tree:
typedef struct 
{
    real pos[3];
    real vel[3];
    real acc[3];
    
    real mass;
    
    unsigned int bodyID;
    unsigned int next;
    unsigned int more;
    
    int isBody;
    struct
    {
        real xx, xy, xz;
        real yy, yz;
        real zz;
    }quad;
    
}gpuTree;

//gpuTree pointer:
typedef __global volatile gpuTree* restrict GTPtr;
typedef __global real* restrict RVPtr;
typedef __global volatile int* restrict IVPtr;
typedef __global uint* restrict UVPtr;

struct node;

typedef struct
{
    uint parent;
    
    int children[8];
    int leafIndex[8];

    uint next;
    uint more;

    uint prefix;
    uint delta;

    uint isLeaf;
    uint treeLevel;
    uint mortonCode;

    uint lock;

    uint id;
    uint chid[2];

    uint pid;

    real massEnclosed;
    real com[3];
    
}node;

typedef __global node* NVPtr;


inline real4 sphericalAccel(real4 pos, real r)
{
    const real tmp = SPHERICAL_SCALE + r;

    return (-SPHERICAL_MASS / (r * sqr(tmp))) * pos;
}

/* gets negative of the acceleration vector of this disk component */
inline real4 miyamotoNagaiDiskAccel(real4 pos, real r)
{
    real4 acc;
    const real a   = DISK_SCALE_LENGTH;
    const real b   = DISK_SCALE_HEIGHT;
    const real zp  = sqrt(sqr(pos.z) + sqr(b));
    const real azp = a + zp;

    const real rp  = sqr(pos.x) + sqr(pos.y) + sqr(azp);
    const real rth = sqrt(cube(rp));  /* rp ^ (3/2) */

    acc.x = -DISK_MASS * pos.x / rth;
    acc.y = -DISK_MASS * pos.y / rth;
    acc.z = -DISK_MASS * pos.z * azp / (zp * rth);
    acc.w = 0.0;

    return acc;
}

inline real4 exponentialDiskAccel(real4 pos, real r)
{
    const real b = DISK_SCALE_LENGTH;

    const real expPiece = exp(-r / b) * (r + b) / b;
    const real factor   = DISK_MASS * (expPiece - 1.0) / cube(r);

    return factor * pos;
}

inline real4 logHaloAccel(real4 pos, real r)
{
    real4 acc;

    const real tvsqr = -2.0 * sqr(HALO_VHALO);
    const real qsqr  = sqr(HALO_FLATTEN_Z);
    const real d     = HALO_SCALE_LENGTH;
    const real zsqr  = sqr(pos.z);

    const real arst  = sqr(d) + sqr(pos.x) + sqr(pos.y);
    const real denom = (zsqr / qsqr) +  arst;

    acc.x = tvsqr * pos.x / denom;
    acc.y = tvsqr * pos.y / denom;
    acc.z = tvsqr * pos.z / ((qsqr * arst) + zsqr);

    return acc;
}

inline real4 nfwHaloAccel(real4 pos, real r)
{
    const real a  = HALO_SCALE_LENGTH;
    const real ar = a + r;
    const real c  = a * sqr(HALO_VHALO) * (r - ar * log((a + r) / a)) / (0.2162165954 * cube(r) * ar);

    return c * pos;
}

inline real4 triaxialHaloAccel(real4 pos, real r)
{
    real4 acc;

    const real qzs      = sqr(HALO_FLATTEN_Z);
    const real rhalosqr = sqr(HALO_SCALE_LENGTH);
    const real mvsqr    = -sqr(HALO_VHALO);

    const real xsqr = sqr(pos.x);
    const real ysqr = sqr(pos.y);
    const real zsqr = sqr(pos.z);

    const real c1 = HALO_C1;
    const real c2 = HALO_C2;
    const real c3 = HALO_C3;

    const real arst  = rhalosqr + (c1 * xsqr) + (c3 * pos.x * pos.y) + (c2 * ysqr);
    const real arst2 = (zsqr / qzs) + arst;

    acc.x = mvsqr * (((2.0 * c1) * pos.x) + (c3 * pos.y) ) / arst2;

    acc.y = mvsqr * (((2.0 * c2) * pos.y) + (c3 * pos.x) ) / arst2;

    acc.z = (2.0 * mvsqr * pos.z) / ((qzs * arst) + zsqr);

    acc.w = 0.0;

    return acc;
}

inline real4 externalAcceleration(real x, real y, real z)
{
    real4 pos = { x, y, z, 0.0 };
    real r = sqrt(sqr(x) + sqr(y) + sqr(z));
    //real r = length(pos); // crashes AMD compiler
    real4 acc;

    if (MIYAMOTO_NAGAI_DISK)
    {
        acc = miyamotoNagaiDiskAccel(pos, r);
    }
    else if (EXPONENTIAL_DISK)
    {
        acc = exponentialDiskAccel(pos, r);
    }

    if (LOG_HALO)
    {
        acc += logHaloAccel(pos, r);
    }
    else if (NFW_HALO)
    {
        acc += nfwHaloAccel(pos, r);
    }
    else if (TRIAXIAL_HALO)
    {
        acc += triaxialHaloAccel(pos, r);
    }

    acc += sphericalAccel(pos, r);

    return acc;
}




/* All kernels will use the same parameters for now */

// #define NBODY_KERNEL(name) name(                        \
//     GTPtr _gTreeIn, GTPtr _gTreeOut                     \
//     )
//     

//OLD KERNEL ARGUMENTS:
#define NBODY_KERNEL(name) name(                        \
    RVPtr _posX, RVPtr _posY, RVPtr _posZ,              \
    RVPtr _velX, RVPtr _velY, RVPtr _velZ,              \
    RVPtr _accX, RVPtr _accY, RVPtr _accZ,              \
    RVPtr _mass,                                        \
                                                        \
                                                        \
    IVPtr _more, IVPtr _next,                           \
                                                        \
                                                        \
    RVPtr _quadXX, RVPtr _quadXY, RVPtr _quadXZ,        \
    RVPtr _quadYY, RVPtr _quadYZ,                       \
    RVPtr _quadZZ                                       \
    )




#if HAVE_INLINE_PTX
inline void strong_global_mem_fence_ptx()
{
    asm("{\n\t"
        "membar.gl;\n\t"
        "}\n\t"
        );
}
#endif

#if HAVE_INLINE_PTX
  #define maybe_strong_global_mem_fence() strong_global_mem_fence_ptx()
#else
  #define maybe_strong_global_mem_fence() mem_fence(CLK_GLOBAL_MEM_FENCE)
#endif /* HAVE_INLINE_PTX */


/* FIXME: should maybe have separate threadcount, but
   Should have attributes most similar to integration */


/* Used by sw93 */
inline real bmax2Inc(real cmPos, real pPos, real psize)
{
    real dmin = cmPos - (pPos - 0.5 * psize);         /* dist from 1st corner */
    real tmp = fmax(dmin, psize - dmin);
    return tmp * tmp;      /* sum max distance^2 */
}

inline bool checkTreeDim(real cmPos, real pPos, real halfPsize)
{
    return (cmPos < pPos - halfPsize || cmPos > pPos + halfPsize);
}


/*
  According to the OpenCL specification, global memory consistency is
  only guaranteed between workitems in the same workgroup.

  We rely on AMD and Nvidia GPU implementation details and pretend
  this doesn't exist when possible.

  - On AMD GPUs, mem_fence(CLK_GLOBAL_MEM_FENCE) compiles to a
  fence_memory instruction which ensures a write is not in the cache
  and is committed to memory before completing.
  We have to be more careful when it comes to reading.

  On previous AMD architectures it was sufficient to have a write
  fence and then other items, not necessarily in the same workgroup,
  would read the value committed by the fence.

  On GCN/Tahiti, the caching architecture was changed. A single
  workgroup will run on the same compute unit. A GCN compute unit has
  it's own incoherent L1 cache (and workgroups have always stayed on
  the same compute unit). A write by one compute unit will be
  committed to memory by a fence there, but a second compute unit may
  read a stale value from its private L1 cache afterwards.

  We may need to use an atomic to ensure we bypass the L1 cache in
  places where we need stronger consistency across workgroups.

  Since Evergreen, the hardware has had a "GDS" buffer for global
  synchronization, however 3 years later we still don't yet have an
  extension to access it from OpenCL. When that finally happens, it
  will probably be a better option to use that for these places.


  - On Nvidia, mem_fence(CLK_GLOBAL_MEM_FENCE) seems to compile to a
  membar.gl instruction, the same as the global sync
  __threadfence(). It may change to a workgroup level membar.cta at
  some point. To be sure we use the correct global level sync, use
  inline PTX to make sure we use membar.gl

  Not sure what to do about Nvidia on Apple's implementation. I'm not
  sure how to even see what PTX it is generating, and there is no
  inline PTX.

*/


#if DOUBLEPREC

inline real atomic_read_real(RVPtr arr, int idx)
{
    union
    {
        int2 i;
        double f;
    } u;

    IVPtr src = (IVPtr) &arr[idx];

    /* Breaks aliasing rules */
    u.i.x = atomic_or(src + 0, 0);
    u.i.y = atomic_or(src + 1, 0);

    return u.f;
}

#else

inline real atomic_read_real(RVPtr arr, int idx)
{
    union
    {
        int2 i;
        float f;
    } u;

    IVPtr src = (IVPtr) &arr[idx];

    u.i = atomic_or(src, 0);

    return u.f;
}
#endif /* DOUBLEPREC */

#if HAVE_CONSISTENT_MEMORY
  #define read_bypass_cache_int(arr, idx) ((arr)[idx])
  #define read_bypass_cache_real(arr, idx) ((arr)[idx])
#else
  #define read_bypass_cache_int(arr, idx) atomic_or(&((arr)[idx]), 0)
  #define read_bypass_cache_real(base, idx) atomic_read_real(base, idx)
#endif /* HAVE_CONSISTENT_MEMORY */

// __attribute__ ((reqd_work_group_size(THREADS7, 1, 1)))
// __kernel void NBODY_KERNEL(summarizationClear)
// {
//     __local int bottom;
// 
//     if (get_local_id(0) == 0)
//     {
//         bottom = _treeStatus->bottom;
//     }
// 
//     barrier(CLK_LOCAL_MEM_FENCE);
// 
//     int inc = get_local_size(0) * get_num_groups(0);
//     int k = (bottom & (-WARPSIZE)) + get_global_id(0);  /* Align to warp size */
// 
//     if (k < bottom)
//     {
//         k += inc;
//     }
// 
//     while (k < NNODE)
//     {
//         _mass[k] = -1.0;
//         _start[k] = NULL_BODY;
// 
//         if (USE_QUAD)
//         {
//             _quadXX[k] = NAN;
//             _quadXY[k] = NAN;
//             _quadXZ[k] = NAN;
// 
//             _quadYY[k] = NAN;
//             _quadYZ[k] = NAN;
// 
//             _quadZZ[k] = NAN;
//         }
// 
//         k += inc;
//     }
// }

// __attribute__ ((reqd_work_group_size(THREADS3, 1, 1)))
// __kernel void NBODY_KERNEL(summarization)
// {
//     __local int bottom;
//     __local volatile int child[NSUB * THREADS3];
//     __local real rootSize;
// 
//     if (get_local_id(0) == 0)
//     {
//         rootSize = _treeStatus->radius;
//         bottom = _treeStatus->bottom;
//     }
//     barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
// 
//     int inc = get_local_size(0) * get_num_groups(0);
//     int k = (bottom & (-WARPSIZE)) + get_global_id(0);  /* Align to warp size */
//     if (k < bottom)
//         k += inc;
// 
//     int missing = 0;
//     while (k <= NNODE) /* Iterate over all cells assigned to thread */
//     {
//         real m, cm, px, py, pz;
//         int cnt, ch;
//         real mk;
// 
//         if (!HAVE_CONSISTENT_MEMORY)
//         {
//             mk = _mass[k];
//         }
// 
//         if (HAVE_CONSISTENT_MEMORY || mk < 0.0)         /* Skip if we finished this cell already */
//         {
//             if (missing == 0)
//             {
//                 /* New cell, so initialize */
//                 cm = px = py = pz = 0.0;
//                 cnt = 0;
//                 int j = 0;
// 
//                 #pragma unroll NSUB
//                 for (int i = 0; i < NSUB; ++i)
//                 {
//                     ch = _child[NSUB * k + i];
//                     if (ch >= 0)
//                     {
// 
//                         if (i != j)
//                         {
//                             /* Move children to front (needed later for speed) */
//                             _child[NSUB * k + i] = -1;
//                             _child[NSUB * k + j] = ch;
//                         }
// 
//                         m = _mass[ch];
//                         child[THREADS3 * missing + get_local_id(0)] = ch; /* Cache missing children */
// 
//                         ++missing;
// 
//                         if (m >= 0.0)
//                         {
//                             /* Child is ready */
//                             --missing;
//                             if (ch >= NBODY) /* Count bodies (needed later) */
//                             {
//                                 cnt += read_bypass_cache_int(_count, ch) - 1;
//                             }
// 
//                             real chx = read_bypass_cache_real(_posX, ch);
//                             real chy = read_bypass_cache_real(_posY, ch);
//                             real chz = read_bypass_cache_real(_posZ, ch);
// 
// 
//                             /* Add child's contribution */
// 
//                             cm += m;
//                             px = mad(m, chx, px);
//                             py = mad(m, chy, py);
//                             pz = mad(m, chz, pz);
//                         }
//                         ++j;
//                     }
//                 }
//                 mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); /* Only for performance */
//                 cnt += j;
//             }
// 
//             if ((HAVE_CONSISTENT_MEMORY || get_num_groups(0) == 1) && missing != 0)
//             {
//                 do
//                 {
//                     /* poll missing child */
//                     ch = child[THREADS3 * (missing - 1) + get_local_id(0)];
//                     m = _mass[ch];
//                     if (m >= 0.0) /* Body children can never be missing, so this is a cell */
//                     {
//                         cl_assert(_treeStatus, ch >= NBODY /* Missing child must be a cell */);
// 
//                         /* child is now ready */
//                         --missing;
// 
//                         /* count bodies (needed later) */
//                         cnt += _count[ch] - 1;
// 
//                         real chx = _posX[ch];
//                         real chy = _posY[ch];
//                         real chz = _posZ[ch];
// 
// 
//                         /* add child's contribution */
//                         cm += m;
//                         px = mad(m, chx, px);
//                         py = mad(m, chy, py);
//                         pz = mad(m, chz, pz);
//                     }
//                     /* repeat until we are done or child is not ready */
//                 }
//                 while ((m >= 0.0) && (missing != 0));
//             }
// 
//             if (missing == 0)
//             {
//                 /* All children are ready, so store computed information */
//                 _count[k] = cnt;
//                 real cx = _posX[k];  /* Load geometric center */
//                 real cy = _posY[k];
//                 real cz = _posZ[k];
// 
//                 real psize;
// 
//                 if (SW93 || NEWCRITERION)
//                 {
//                     psize = _critRadii[k]; /* Get saved size (half cell = radius) */
//                 }
// 
//                 m = 1.0 / cm;
//                 px *= m; /* Scale up to position */
//                 py *= m;
//                 pz *= m;
// 
//                 /* Calculate opening criterion if necessary */
//                 real rc2;
// 
//                 if (THETA == 0.0)
//                 {
//                     rc2 = sqr(2.0 * rootSize);
//                 }
//                 else if (SW93)
//                 {
//                     real bmax2 = bmax2Inc(px, cx, psize);
//                     bmax2 += bmax2Inc(py, cy, psize);
//                     bmax2 += bmax2Inc(pz, cz, psize);
//                     rc2 = bmax2 / (THETA * THETA);
//                 }
//                 else if (NEWCRITERION)
//                 {
//                     real dx = px - cx;  /* Find distance from center of mass to geometric center */
//                     real dy = py - cy;
//                     real dz = pz - cz;
//                     real dr = sqrt(mad(dz, dz, mad(dy, dy, dx * dx)));
// 
//                     real rc = (psize / THETA) + dr;
// 
//                     rc2 = rc * rc;
//                 }
// 
//                 if (SW93 || NEWCRITERION)
//                 {
//                     /* We don't have the size of the cell for BH86, but really still should check */
//                     bool xTest = checkTreeDim(px, cx, psize);
//                     bool yTest = checkTreeDim(py, cy, psize);
//                     bool zTest = checkTreeDim(pz, cz, psize);
//                     bool structureCheck = xTest || yTest || zTest;
//                     if (structureCheck)
//                     {
//                         _treeStatus->errorCode = NBODY_KERNEL_TREE_STRUCTURE_ERROR;
//                     }
//                 }
// 
//                 _posX[k] = px;
//                 _posY[k] = py;
//                 _posZ[k] = pz;
// 
//                 if (SW93 || NEWCRITERION)
//                 {
//                     _critRadii[k] = rc2;
//                 }
// 
//                 if (USE_QUAD)
//                 {
//                     /* We must initialize all cells quad moments to NaN */
//                     _quadXX[k] = NAN;
//                 }
// 
//                 maybe_strong_global_mem_fence(); /* Make sure data is visible before setting mass */
//                 _mass[k] = cm;
// 
//                 if (HAVE_CONSISTENT_MEMORY)
//                 {
//                     k += inc;  /* Move on to next cell */
//                 }
//             }
//         }
// 
//         if (!HAVE_CONSISTENT_MEMORY)
//         {
//             missing = 0;
//             cnt = 0;
//             k += inc;  /* Move on to next cell */
//         }
//     }
// }


#if NOSORT
/* Debugging */
__attribute__ ((reqd_work_group_size(THREADS4, 1, 1)))
__kernel void NBODY_KERNEL(sort)
{
    __local int bottoms;

    if (get_local_id(0) == 0)
    {
        bottoms = _treeStatus->bottom;
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    int bottom = bottoms;
    int dec = get_local_size(0) * get_num_groups(0);
    int k = NNODE + 1 - dec + get_global_id(0);

    while (k >= bottom)
    {
        _sort[k] = k;
        k -= dec;  /* Move on to next cell */
    }
}

#else

/* Real sort kernel, will never finish unless all threads can be launched at once */
// __attribute__ ((reqd_work_group_size(THREADS4, 1, 1)))
// __kernel void NBODY_KERNEL(sort)
// {
//     __local int bottoms;
// 
//     if (get_local_id(0) == 0)
//     {
//         bottoms = _treeStatus->bottom;
//     }
//     barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
// 
//     int bottom = bottoms;
//     int dec = get_local_size(0) * get_num_groups(0);
//     int k = NNODE + 1 - dec + get_global_id(0);
// 
//     while (k >= bottom) /* Iterate over all cells assigned to thread */
//     {
//         int start = _start[k];
//         if (start >= 0)
//         {
//             #pragma unroll NSUB
//             for (int i = 0; i < NSUB; ++i)
//             {
//                 int ch = _child[NSUB * k + i];
//                 if (ch >= NBODY)         /* Child is a cell */
//                 {
//                     _start[ch] = start;  /* Set start ID of child */
//                     start += _count[ch]; /* Add #bodies in subtree */
//                 }
//                 else if (ch >= 0)        /* Child is a body */
//                 {
//                     _sort[start] = ch;   /* Record body in sorted array */
//                     ++start;
//                 }
//             }
//         }
// 
//         k -= dec;  /* Move on to next cell */
//     }
// }

#endif /* NOSORT */

inline void incAddMatrix(QuadMatrix* restrict a, QuadMatrix* restrict b)
{
    a->xx += b->xx;
    a->xy += b->xy;
    a->xz += b->xz;

    a->yy += b->yy;
    a->yz += b->yz;

    a->zz += b->zz;
}

inline void quadCalc(QuadMatrix* quad, real4 chCM, real4 kp)
{
    real4 dr;
    dr.x = chCM.x - kp.x;
    dr.y = chCM.y - kp.y;
    dr.z = chCM.z - kp.z;

    real drSq = mad(dr.z, dr.z, mad(dr.y, dr.y, dr.x * dr.x));

    quad->xx = chCM.w * (3.0 * (dr.x * dr.x) - drSq);
    quad->xy = chCM.w * (3.0 * (dr.x * dr.y));
    quad->xz = chCM.w * (3.0 * (dr.x * dr.z));

    quad->yy = chCM.w * (3.0 * (dr.y * dr.y) - drSq);
    quad->yz = chCM.w * (3.0 * (dr.y * dr.z));

    quad->zz = chCM.w * (3.0 * (dr.z * dr.z) - drSq);
}


/* Very similar to summarization kernel. Calculate the quadrupole
 * moments for the cells in an almost identical way */
// __attribute__ ((reqd_work_group_size(THREADS5, 1, 1)))
// __kernel void NBODY_KERNEL(quadMoments)
// {
//     __local int bottom;
//     __local volatile int child[NSUB * THREADS5];
//     __local real rootSize;
//     __local int maxDepth;
// 
//     if (get_local_id(0) == 0)
//     {
//         rootSize = _treeStatus->radius;
//         bottom = _treeStatus->bottom;
//         maxDepth = _treeStatus->maxDepth;
//     }
//     barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
// 
//     int inc = get_local_size(0) * get_num_groups(0);
//     int k = (bottom & (-WARPSIZE)) + get_global_id(0);  /* Align to warp size */
//     if (k < bottom)
//         k += inc;
// 
//     if (maxDepth > MAXDEPTH)
//     {
//         _treeStatus->errorCode = maxDepth;
//         return;
//     }
// 
//     int missing = 0;
//     while (k <= NNODE)   /* Iterate over all cells assigned to thread */
//     {
//         int ch;          /* Child index */
//         real4 kp;        /* Position of this cell k */
//         QuadMatrix kq;   /* Quad moment for this cell */
//         QuadMatrix qCh;  /* Loads of child quad moments */
//         real kQxx;
// 
//         kp.x = _posX[k]; /* This cell's center of mass position */
//         kp.y = _posY[k];
//         kp.z = _posZ[k];
// 
//         if (!HAVE_CONSISTENT_MEMORY)
//         {
//             kQxx = _quadXX[k];
//         }
// 
//         if (HAVE_CONSISTENT_MEMORY || isnan(kQxx))
//         {
//             if (missing == 0)
//             {
//                 /* New cell, so initialize */
//                 kq.xx = kq.xy = kq.xz = 0.0;
//                 kq.yy = kq.yz = 0.0;
//                 kq.zz = 0.0;
// 
//                 int j = 0;
// 
//                 #pragma unroll NSUB
//                 for (int i = 0; i < NSUB; ++i)
//                 {
//                     QuadMatrix quad; /* Increment from this descendent */
//                     ch = _child[NSUB * k + i];
// 
//                     if (ch >= 0)
//                     {
//                         if (isBody(ch))
//                         {
//                             real4 chCM;
// 
//                             chCM.x = _posX[ch];
//                             chCM.y = _posY[ch];
//                             chCM.z = _posZ[ch];
//                             chCM.w = _mass[ch];
// 
//                             quadCalc(&quad, chCM, kp);
//                             incAddMatrix(&kq, &quad);  /* Add to total moment */
//                         }
// 
//                         if (isCell(ch))
//                         {
//                             child[THREADS5 * missing + get_local_id(0)] = ch; /* Cache missing children */
//                             ++missing;
// 
//                             qCh.xx = _quadXX[ch];
//                             if (!isnan(qCh.xx))
//                             {
//                                 real4 chCM;
// 
//                                 /* Load the rest */
//                               //qCh.xx = read_bypass_cache_real(_quadXX, ch);
//                                 qCh.xy = read_bypass_cache_real(_quadXY, ch);
//                                 qCh.xz = read_bypass_cache_real(_quadXZ, ch);
// 
//                                 qCh.yy = read_bypass_cache_real(_quadYY, ch);
//                                 qCh.yz = read_bypass_cache_real(_quadYZ, ch);
// 
//                                 qCh.zz = read_bypass_cache_real(_quadZZ, ch);
// 
//                                 chCM.x = _posX[ch];
//                                 chCM.y = _posY[ch];
//                                 chCM.z = _posZ[ch];
//                                 chCM.w = _mass[ch];
// 
//                                 quadCalc(&quad, chCM, kp);
// 
//                                 --missing;  /* Child is ready */
// 
//                                 incAddMatrix(&quad, &qCh);  /* Add child's contribution */
//                                 incAddMatrix(&kq, &quad);   /* Add to total moment */
//                             }
//                         }
// 
//                         ++j;
//                     }
//                 }
//                 mem_fence(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE); /* Only for performance */
//             }
// 
//             if ((HAVE_CONSISTENT_MEMORY || get_num_groups(0) == 1) && missing != 0)
//             {
//                 do
//                 {
//                     QuadMatrix quad; /* Increment from this missing child */
// 
//                     /* poll missing child */
//                     ch = child[THREADS5 * (missing - 1) + get_local_id(0)];
//                     cl_assert(_treeStatus, ch > 0);
//                     cl_assert(_treeStatus, ch >= NBODY);
//                     if (ch >= NBODY) /* Is a cell */
//                     {
//                         qCh.xx = _quadXX[ch];
//                         if (!isnan(qCh.xx))
//                         {
//                             real4 chCM;
// 
//                             chCM.x = _posX[ch];
//                             chCM.y = _posY[ch];
//                             chCM.z = _posZ[ch];
//                             chCM.w = _mass[ch];
// 
//                             //qCh.xx = _quadXX[ch];
//                             qCh.xy = _quadXY[ch];
//                             qCh.xz = _quadXZ[ch];
// 
//                             qCh.yy = _quadYY[ch];
//                             qCh.yz = _quadYZ[ch];
// 
//                             qCh.zz = _quadZZ[ch];
// 
//                             quadCalc(&quad, chCM, kp);
// 
//                             --missing;  /* Child is now ready */
// 
//                             incAddMatrix(&quad, &qCh);  /* Add subcell's moment */
//                             incAddMatrix(&kq, &quad);   /* add child's contribution */
//                         }
//                     }
//                     /* repeat until we are done or child is not ready */
//                 }
//                 while ((!isnan(qCh.xx)) && (missing != 0));
//             }
// 
//             if (missing == 0)
//             {
//                 /* All children are ready, so store computed information */
//                 //_quadXX[k] = kq.xx;  /* Store last */
//                 _quadXY[k] = kq.xy;
//                 _quadXZ[k] = kq.xz;
// 
//                 _quadYY[k] = kq.yy;
//                 _quadYZ[k] = kq.yz;
// 
//                 _quadZZ[k] = kq.zz;
// 
//                 write_mem_fence(CLK_GLOBAL_MEM_FENCE); /* Make sure data is visible before setting tested quadx */
//                 _quadXX[k] = kq.xx;
//                 write_mem_fence(CLK_GLOBAL_MEM_FENCE);
// 
//                 if (HAVE_CONSISTENT_MEMORY)
//                 {
//                     k += inc;  /* Move on to next cell */
//                 }
//             }
//         }
// 
//         if (!HAVE_CONSISTENT_MEMORY)
//         {
//             missing = 0;
//             k += inc;
//         }
//     }
// }

#if HAVE_INLINE_PTX
inline int warpAcceptsCellPTX(real rSq, real rCritSq)
{
    uint result;

  #if DOUBLEPREC
    asm("{\n\t"
        ".reg .pred cond, out;\n\t"
        "setp.ge.f64 cond, %1, %2;\n\t"
        "vote.all.pred out, cond;\n\t"
        "selp.u32 %0, 1, 0, out;\n\t"
        "}\n\t"
        : "=r"(result)
        : "d"(rSq), "d"(rCritSq));
  #else
    asm("{\n\t"
        ".reg .pred cond, out;\n\t"
        "setp.ge.f32 cond, %1, %2;\n\t"
        "vote.all.pred out, cond;\n\t"
        "selp.u32 %0, 1, 0, out;\n\t"
        "}\n\t"
        : "=r"(result)
        : "f"(rSq), "f"(rCritSq));
  #endif /* DOUBLEPREC */

    return result;
}
#endif /* HAVE_INLINE_PTX */

real clampValue(real v, real clampVal){
  return(floor(pow((real)10, clampVal) * v)/pow((real)10, clampVal));
}

/*
 * This should be equivalent roughtly to CUDA's __all() with the conditions
 * A barrier should be unnecessary here since
 * all the threads in a wavefront should be
 * forced to run simulatenously. This is not
 * over the workgroup, but the actual
 * wavefront.
 */
inline int warpAcceptsCellSurvey(__local volatile int allBlock[THREADS6 / WARPSIZE], int warpId, int cond)
{
    /* Relies on underlying wavefronts (not whole workgroup)
       executing in lockstep to not require barrier */

    int old = allBlock[warpId];

    /* Increment if true, or leave unchanged */
    (void) atom_add(&allBlock[warpId], cond);

    int ret = (allBlock[warpId] == WARPSIZE);
    allBlock[warpId] = old;

    return ret;
}

#if HAVE_INLINE_PTX
/* Need to do this horror to avoid wasting __local if we can use the real warp vote */
#define warpAcceptsCell(allBlock, base, rSq, dq) warpAcceptsCellPTX(rSq, dq)
#else
#define warpAcceptsCell(allBlock, base, rSq, dq) warpAcceptsCellSurvey(allBlock, base, (rSq) >= (dq))
#endif
//TODO: Write Treecode Force Kernel
__attribute__ ((reqd_work_group_size(THREADS6, 1, 1)))
__kernel void forceCalculation(GTPtr _gTreeIn, GTPtr _gTreeOut)
{
  _gTreeOut[0].mass = 20;
}


//__attribute__ ((reqd_work_group_size(THREADS6, 1, 1)))


// __kernel void forceCalculationExact(GTPtr _gTreeIn, GTPtr _gTreeOut)
// {
    
//     int a = (int)get_global_id(0);
// //     if(a == 0){ //We set the output array in the first thread, since we don't want to do it in every thread; That would be a waste.
// //         _gTreeOut = _gTreeIn;
// //     }
    
//     for(int i = 0; i < 3; ++i){
//        _gTreeIn[a].acc[i] = 0; //Initialize accelerations to zero before we do calculations
//     }
//     //TODO: start writing force calculations
//     if(_gTreeIn[a].isBody == 1){
//         GTPtr tmp = &_gTreeIn[0];
//         while(tmp != NULL){
//             if(tmp->isBody == 1){ //If it's a body we can go to the next value
//                 //if(tmp != &_gTreeIn[a]){    //make sure we aren't self-interacting:
//                     real pos1[3];
//                     real pos2[3];
//                     real drVec[3];
//                     real compVec[3];
//                     for(int i = 0; i < 3; ++i){
//                         pos1[i] = _gTreeIn[a].pos[i];
//                         pos2[i] = tmp->pos[i];
//                         drVec[i] = (pos2[i] - pos1[i]);
//                     }
//                     //Calculate distance between two bodies:
                    
//                     real dr2 = mad(drVec[2], drVec[2], mad(drVec[1], drVec[1], drVec[0] * drVec[0])) + EPS2;
//                     //real dr2 = (drVec[0] * drVec[0]) + (drVec[1] * drVec[1]) + (drVec[2] * drVec[2]) + EPS2;
//                     real dr = sqrt(dr2);
//                     real m2 = tmp->mass;
//                     real ai = m2/(dr*dr2);

//                     _gTreeIn[a].acc[0] += ai * drVec[0];
//                     _gTreeIn[a].acc[1] += ai * drVec[1];
//                     _gTreeIn[a].acc[2] += ai * drVec[2];
                    
//                     // _gTreeIn[a].acc[0] = mad(ai, drVec[0], _gTreeIn[a].acc[0]);
//                     // _gTreeIn[a].acc[1] = mad(ai, drVec[1], _gTreeIn[a].acc[1]);
//                     // _gTreeIn[a].acc[2] = mad(ai, drVec[2], _gTreeIn[a].acc[2]);
                       
//                 //}
                    
                
                
//                 if(tmp->next != 0){ //Check to see that we aren't at the end and pointing back to root
//                     tmp = &_gTreeIn[tmp->next];
//                 }
//                 else{ //If there are no more bodies in the (next) index, we must be at the end of the tree
//                     tmp = NULL;
//                 }
//             }
//             else{ //If not body, then must be cell
//                 tmp = &_gTreeIn[tmp->more]; //cells MUST have a (more) index
//             }
//         }        
//     }
//     barrier(CLK_GLOBAL_MEM_FENCE);
// //=========================================
// }

//SoA FORCE CALCULATION KERNEL:
// __attribute__ ((reqd_work_group_size(THREADS6, 1, 1)))
__kernel void forceCalculationExact(RVPtr x, RVPtr y, RVPtr z,
                                    RVPtr vx, RVPtr vy, RVPtr vz,
                                    RVPtr ax, RVPtr ay, RVPtr az,
                                    RVPtr mass){
  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);


  __local real posX[WARPSIZE];
  __local real posY[WARPSIZE];
  __local real posZ[WARPSIZE];
  __local real accTempX[WARPSIZE];
  __local real accTempY[WARPSIZE];
  __local real accTempZ[WARPSIZE];

  __private real4 particle;
  __private real4 accPrivate;
  __private real4 drVec;

  __private real dr2;
  __private real dr;
  __private real m2;
  __private real ai;

  event_t e[3];


  accTempX[l] = 0;
  accTempY[l] = 0;
  accTempZ[l] = 0;

  particle.x = x[g];
  particle.y = y[g];
  particle.z = z[g];

  barrier(CLK_LOCAL_MEM_FENCE);

  int comp1;
  int comp2; 
  for(int i = 0; i < EFFNBODY/WARPSIZE; ++i){
     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
     e[0] = async_work_group_copy(posX, x+i*WARPSIZE, WARPSIZE, 0);
     e[1] = async_work_group_copy(posY, y+i*WARPSIZE, WARPSIZE, 0);
     e[2] = async_work_group_copy(posZ, z+i*WARPSIZE, WARPSIZE, 0);
     wait_group_events(3, e);
    for(int j = 0; j < WARPSIZE; ++j){
      comp1 = i * WARPSIZE + j < NBODY;
      comp2 = g < NBODY;
      drVec.x = posX[j] - particle.x;
      drVec.y = posY[j] - particle.y;
      drVec.z = posZ[j] - particle.z;
      dr2 = mad(drVec.z, drVec.z, mad(drVec.y, drVec.y, mad(drVec.x, drVec.x,EPS2)));
      dr = sqrt(dr2);
      m2 = mass[j];
      ai = m2/(dr*dr2) * comp1 * comp2;
      accTempX[l] += ai * drVec.x;
      accTempY[l] += ai * drVec.y;
      accTempZ[l] += ai * drVec.z;
    }
  }
  if(USE_EXTERNAL_POTENTIAL)
  {
    real4 externAcc = externalAcceleration(particle.x, particle.y, particle.z);
    accTempX[l] += externAcc.x;
    accTempY[l] += externAcc.y;
    accTempZ[l] += externAcc.z;
  }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    // accTempX[l] = mass[l];
    e[0] = async_work_group_copy(ax + group * WARPSIZE, accTempX, WARPSIZE, 0);
    e[1] = async_work_group_copy(ay + group * WARPSIZE, accTempY, WARPSIZE, 0);
    e[2] = async_work_group_copy(az + group * WARPSIZE, accTempZ, WARPSIZE, 0);
    wait_group_events(3, e);
}

__kernel void advanceHalfVelocity(RVPtr x, RVPtr y, RVPtr z,
                                    RVPtr vx, RVPtr vy, RVPtr vz,
                                    RVPtr ax, RVPtr ay, RVPtr az,
                                    RVPtr mass){
  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);

  real dtHalf = 0.5 * TIMESTEP;
  vx[g] = mad(dtHalf, ax[g], vx[g]);
  vy[g] = mad(dtHalf, ay[g], vy[g]);
  vz[g] = mad(dtHalf, az[g], vz[g]);
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


////////////////////////////////////
// GPU TREECODE 
////////////////////////////////////
__kernel void boundingBox(RVPtr x, RVPtr y, RVPtr z,
                        RVPtr vx, RVPtr vy, RVPtr vz,
                        RVPtr ax, RVPtr ay, RVPtr az,
                        RVPtr mass, RVPtr xMax, RVPtr yMax,
                        RVPtr zMax, RVPtr xMin, RVPtr yMin,
                        RVPtr zMin, UVPtr mCodes, UVPtr iteration){
 
  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);

  event_t e[6];

  //Create local variables and copy global data into them:
  __local real maxTemp[3][WARPSIZE];
  __local real minTemp[3][WARPSIZE];
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

  int iter = (int)log2((real)WARPSIZE);

  for(int i = 0; i < iter; ++i){
    int expVal = (int)exp2((real)i);
    int nextVal = min((int)l + expVal, (int)WARPSIZE - 1);
    if(l % (expVal) * 2 == 0){
      int gt = (maxTemp[0][l] > maxTemp[0][nextVal]);
      int lt = (minTemp[0][l] < minTemp[0][nextVal]);
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


inline uint expandBits(uint v){
  v = (v * 0x00010001) & 0xFF0000FF;
  v = (v * 0x00000101) & 0x0F00F00F;
  v = (v * 0x00000011) & 0xC30C30C3;
  v = (v * 0x00000005) & 0x49249249;
  return v;
}

inline uint encodeLocation(real4 pos){
  pos.x = (min(max(pos.x * 1024.0, 0.0), 1023.0));
  pos.y = (min(max(pos.y * 1024.0, 0.0), 1023.0));
  pos.z = (min(max(pos.z * 1024.0, 0.0), 1023.0));

  uint xx = expandBits((uint)pos.x);
  uint yy = expandBits((uint)pos.y);
  uint zz = expandBits((uint)pos.z);

  return xx * 4 + yy * 2 + zz;
}

//This kernel uses a bitonic sorting algorithm to sort each warp's morton codes:
__kernel void bitonicMortonSort(RVPtr x, RVPtr y, RVPtr z,
    RVPtr vx, RVPtr vy, RVPtr vz,
    RVPtr ax, RVPtr ay, RVPtr az,
    RVPtr mass, RVPtr xMax, RVPtr yMax,
    RVPtr zMax, RVPtr xMin, RVPtr yMin,
    RVPtr zMin, UVPtr mCodes_G, UVPtr iteration, int inc, int len){


    uint gid = get_global_id(0);
    uint low = gid & (inc - 1);
    uint g = (gid<<1) - low;
    uint j = g | inc;
    
    //Create local variables and copy global data into them:
    real iDataR[7] = {x[g], y[g], z[g], vx[g], vy[g], vz[g], mass[g]};
    uint iKey = mCodes_G[g];//iData[0]; //getKey(iData);
    real jDataR[7] = {x[j], y[j], z[j], vx[j], vy[j], vz[j], mass[j]};
    uint jKey = mCodes_G[j]; //getKey(jData);

    bool smaller = (jKey < iKey) || (jKey == iKey && j < g);
    bool swap = smaller ^ (j < g) ^ (((len<<1) & g) != 0);
    // barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    if(swap){
        mCodes_G[g] = jKey;
        x[g] = jDataR[0];
        y[g] = jDataR[1];
        z[g] = jDataR[2];
        vx[g] = jDataR[3];
        vy[g] = jDataR[4];
        vz[g] = jDataR[5];
        mass[g] = jDataR[6];

        mCodes_G[j] = iKey;
        x[j] = iDataR[0];
        y[j] = iDataR[1];
        z[j] = iDataR[2];
        vx[j] = iDataR[3];
        vy[j] = iDataR[4];
        vz[j] = iDataR[5];
        mass[j] = iDataR[6];
        
    }
}

__kernel void encodeTree(RVPtr x, RVPtr y, RVPtr z,
                        RVPtr vx, RVPtr vy, RVPtr vz,
                        RVPtr ax, RVPtr ay, RVPtr az,
                        RVPtr mass, RVPtr xMax, RVPtr yMax,
                        RVPtr zMax, RVPtr xMin, RVPtr yMin,
                        RVPtr zMin, UVPtr mCodes_G, UVPtr iteration){

  uint g = (uint) get_global_id(0);
  uint l = (uint) get_local_id(0);
  uint group = (uint) get_group_id(0);


  __private real4 pos_local;
  
  pos_local.x = (x[g] - xMin[0])/(xMax[0]-xMin[0]);
  pos_local.y = (y[g] - yMin[0])/(yMax[0]-yMin[0]);
  pos_local.z = (z[g] - zMin[0])/(zMax[0]-zMin[0]);

  //CALCULATE MORTON CODE
  mCodes_G[g] = encodeLocation(pos_local);

  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  //Use global thread ID as a LSB identifier to seperate morton code collisions.
}


//Split function available here: https://devblogs.nvidia.com/parallelforall/thinking-parallel-part-iii-tree-construction-gpu/
inline int findSplit(UVPtr mCodes, int first, int last){
    uint fCode = mCodes[first];
    uint lCode = mCodes[last];

    if(fCode == lCode){
        return first;
    }

    int prefix = clz(fCode^lCode);
    int split = first;
    
    int step = last - first;

    do{
        step = (step + 1) >> 1;
        int newSplit = split + step;
        
        if(newSplit < last){
            uint splitCode = mCodes[newSplit];
            int splitPrefix = clz(fCode^splitCode);
            if(splitPrefix > prefix){
                split = newSplit;
            }
        }
    }while(step > 1);
    return split;
}

inline int2 findRange(UVPtr mCodes, int size, int index){
    int lso = size - 1;
    if(index ==  0){
        return (int2)(0, lso);
    }
    int dir;
    int dMin;
    int initialIndex = index;

    uint minone = mCodes[index - 1];
    uint precis = mCodes[index];
    uint pluone = mCodes[index + 1];
    if((minone == precis && pluone == precis)){
        while(index > 0 && index < lso){
            ++index;
            if(index >=lso){
                break;
            }
            if(mCodes[index] != mCodes[index+1]){
                break;
            }
        }
        return (int2)(initialIndex, index);
    }
    else{
        int2 lr = (int2)(clz(precis^minone), clz(precis^pluone));
        if(lr.x > lr.y){
            dir = -1;
            dMin = lr.y;
        }
        else{
            dir = 1;
            dMin = lr.x;
        }
    }

    int lMax = 2;
    int testIndex = index + lMax * dir;

    //Find search range:
    while((testIndex <= lso && testIndex >= 0)?(clz(precis ^ mCodes[testIndex]) > dMin):(false)){
        lMax *= 2;
        testIndex = index + lMax * dir;
    }
    int l = 0;

    for(int div = 2; lMax/div >=1; div *=2){
        int t = lMax/div;
        int newTest = index + (l + t) * dir;

        if(newTest <= lso && newTest >= 0){
            int splitPrefix = clz(precis^mCodes[newTest]);
            if(splitPrefix>dMin){
                l = l+t;
            }
        }
    }

    if(dir == 1){
        return (int2)(index, index + l*dir);
    }
    else{
        return (int2)(index + l*dir, index);
    }
    
}

__kernel void constructTree(RVPtr x, RVPtr y, RVPtr z,
                            RVPtr vx, RVPtr vy, RVPtr vz,
                            RVPtr ax, RVPtr ay, RVPtr az,
                            RVPtr mass, RVPtr xMax, RVPtr yMax,
                            RVPtr zMax, RVPtr xMin, RVPtr yMin,
                            RVPtr zMin, UVPtr mCodes_G, UVPtr iteration,
                            NVPtr gpuBinaryTree, NVPtr gpuLeafs, UVPtr nodeCounts){

    uint g = (uint) get_global_id(0);
    uint l = (uint) get_local_id(0);
    uint group = (uint) get_group_id(0);

    gpuLeafs[g].id = g;
    gpuBinaryTree[g].id = g;
    
    gpuLeafs[EFFNBODY - 1].id = 100 + EFFNBODY - 1;
    gpuBinaryTree[EFFNBODY - 1].id = EFFNBODY - 1;

    nodeCounts[g] = 0;

    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    int2 range = findRange(mCodes_G, EFFNBODY, g);
    gpuBinaryTree[g].massEnclosed = 0;
    gpuBinaryTree[g].com[0] = 0;
    gpuBinaryTree[g].com[1] = 0;
    gpuBinaryTree[g].com[2] = 0;
    for(int i = range.x; i < range.y; ++i){
        real com_[3] = {0};
        gpuBinaryTree[g].massEnclosed += mass[i];
        gpuBinaryTree[g].com[0] += mass[i] * x[i];
        gpuBinaryTree[g].com[1] += mass[i] * y[i];
        gpuBinaryTree[g].com[2] += mass[i] * z[i];
    }
    gpuBinaryTree[g].com[0] = gpuBinaryTree[g].com[0]/gpuBinaryTree[g].massEnclosed;
    gpuBinaryTree[g].com[1] = gpuBinaryTree[g].com[1]/gpuBinaryTree[g].massEnclosed;
    gpuBinaryTree[g].com[2] = gpuBinaryTree[g].com[2]/gpuBinaryTree[g].massEnclosed;

    int split = findSplit(mCodes_G, range.x, range.y);

    uint delta = clz(mCodes_G[split]^mCodes_G[split+1]) - 2;
    gpuBinaryTree[g].delta = delta;    
    if(split == range.x){
        gpuBinaryTree[g].leafIndex[0] = split;
        gpuBinaryTree[g].children[0] = 0;
        gpuLeafs[split].delta = delta;
        gpuLeafs[split].parent = g;
    }
    else{
        gpuBinaryTree[g].children[0] = split;
        gpuBinaryTree[split].parent = g;
    }

    if(split + 1 == range.y){
        gpuBinaryTree[g].leafIndex[1] = split + 1;
        gpuBinaryTree[g].children[1] = 0;
        gpuLeafs[split + 1].delta = delta;
        gpuLeafs[split + 1].parent = g;
    }
    else{
        gpuBinaryTree[g].children[1] = split + 1;
        gpuBinaryTree[split + 1].parent = g;
    }
    
    gpuBinaryTree[g].prefix = mCodes_G[g];
    gpuBinaryTree[g].prefix >>= (30 - delta);
}

__kernel void countOctNodes(RVPtr x, RVPtr y, RVPtr z,
    RVPtr vx, RVPtr vy, RVPtr vz,
    RVPtr ax, RVPtr ay, RVPtr az,
    RVPtr mass, RVPtr xMax, RVPtr yMax,
    RVPtr zMax, RVPtr xMin, RVPtr yMin,
    RVPtr zMin, UVPtr mCodes_G, UVPtr iteration,
    NVPtr gpuBinaryTree, NVPtr gpuLeafs, UVPtr nodeCounts){

    uint g = (uint) get_global_id(0);
    uint l = (uint) get_local_id(0);
    uint group = (uint) get_group_id(0);

    if(g != 0){
        // NVPtr node = &gpuBinaryTree[g];
        uint delta = gpuBinaryTree[g].delta;
        uint parentDelta = gpuBinaryTree[gpuBinaryTree[g].parent].delta;
        nodeCounts[g] = delta/3 - parentDelta/3;
    }
    else{
        nodeCounts[g] = 0;
    }
}

__kernel void prefixSumInclusiveUtil(UVPtr nodeCounts, UVPtr swap){
    uint g = (uint) get_global_id(0);
    
    swap[g] += nodeCounts[g];
}

#define STRIDE(iteration, offset) (1 << (iteration + offset))

__kernel void prefixSumUpsweep(UVPtr nodeCounts, UVPtr iteration){
    uint g = (uint) get_global_id(0);
    uint currentIteration = (*iteration);
    uint index = g * STRIDE(currentIteration, 1);

    barrier(CLK_GLOBAL_MEM_FENCE);

    nodeCounts[index + STRIDE(currentIteration, 1) - 1] = nodeCounts[index + STRIDE(currentIteration, 0) - 1] 
                                                + nodeCounts[index + STRIDE(currentIteration, 1) - 1];
    
    (*iteration) = currentIteration + 1;
}

__kernel void prefixSumDownsweep(UVPtr nodeCounts, UVPtr iteration){
    uint g = (uint) get_global_id(0);
    uint currentIteration = (*iteration);

    if(get_global_size(0) == 1){
        nodeCounts[EFFNBODY - 1] = 0;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
    
    uint index = g * STRIDE(currentIteration, 0);
    uint t = nodeCounts[index + STRIDE(currentIteration, -1) - 1];

    nodeCounts[index + STRIDE(currentIteration, -1) - 1] = nodeCounts[index + STRIDE(currentIteration, 0) - 1];
    nodeCounts[index + STRIDE(currentIteration, 0) - 1] = t + nodeCounts[index + STRIDE(currentIteration, 0) - 1];
    
    (*iteration) = currentIteration - 1;
}

__kernel void prefixSum(UVPtr nodeCounts, UVPtr swap, UVPtr iteration){
    uint g = (uint) get_global_id(0);
    uint l = (uint) get_local_id(0);
    uint group = (uint) get_group_id(0);
    swap[g] = nodeCounts[g]; 
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
    for(int i = 0; i < g; ++i){
       nodeCounts[g] += swap[i];
    }
}

inline uint extractBits(uint number, uint level){
    uint temp = 0;
    number >>= 3*level;
    temp = number & 7;
    return temp;
}


__kernel void constructOctTree(RVPtr x, RVPtr y, RVPtr z,
    RVPtr vx, RVPtr vy, RVPtr vz,
    RVPtr ax, RVPtr ay, RVPtr az,
    RVPtr mass, RVPtr xMax, RVPtr yMax,
    RVPtr zMax, RVPtr xMin, RVPtr yMin,
    RVPtr zMin, UVPtr mCodes_G, UVPtr iteration,
    NVPtr gpuBinaryTree, NVPtr gpuLeafs, UVPtr nodeCounts, NVPtr octree){

    uint g = (uint) get_global_id(0);
    uint l = (uint) get_local_id(0);
    uint group = (uint) get_group_id(0);

    if(g != 0){
        uint index = nodeCounts[g - 1] + 1;
        uint count = nodeCounts[g] - nodeCounts[g-1];
        if(count > 0){
            for(int i = 0; i < count; ++i){
                for(int j = 0; j < 8; ++j){
                    octree[index + i].leafIndex[j] = -1;    
                }
                octree[index + i].treeLevel = gpuBinaryTree[g].delta/3 - (count - 1 - i);
                octree[index + i].id = index + i;
                octree[index + i].prefix = mCodes_G[g] >> (30 - (3 * octree[index + i].treeLevel));
                octree[index + i].com[0] = gpuBinaryTree[g].com[0];
                octree[index + i].com[1] = gpuBinaryTree[g].com[1];
                octree[index + i].com[2] = gpuBinaryTree[g].com[2];
                octree[index + i].massEnclosed = gpuBinaryTree[g].massEnclosed;
                if(i > 0){
                    octree[index + i].parent = index + i - 1;
                    uint childIndex = extractBits(octree[index + i].prefix, 0);
                    octree[index + i - 1].children[childIndex] = index + i;
                }
            }
        }
        //Find parent node
        barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
        if(count > 0){
            uint testIndex = gpuBinaryTree[g].parent;
            uint childIndex = extractBits(octree[index].prefix, 0);
            while(testIndex != 0 && nodeCounts[testIndex] - nodeCounts[testIndex - 1] == 0){
                testIndex = gpuBinaryTree[testIndex].parent;
            }
            if(testIndex != 0){
                octree[index].parent = nodeCounts[testIndex - 1] + 1;
                octree[nodeCounts[testIndex - 1] + 1].children[childIndex] = octree[index].id;
                // octree[index].children[childIndex] = 1;
            }
            else{
                octree[index].parent = 0;
                octree[0].children[childIndex] = octree[index].id;
                // octree[index].children[childIndex] = 1;
            }
        }
    }
    else{
        octree[g].id = 0;
        octree[g].treeLevel = 0;
        octree[g].prefix = 0;
        octree[g].parent = 0;
        for(int j = 0; j < 8; ++j){
            octree[g].leafIndex[j] = -1;    
        }

        octree[g].massEnclosed = gpuBinaryTree[0].massEnclosed;
        octree[g].com[0] = gpuBinaryTree[0].com[0];
        octree[g].com[1] = gpuBinaryTree[0].com[1];
        octree[g].com[2] = gpuBinaryTree[0].com[2];
    }
}


kernel void linkOctree(RVPtr x, RVPtr y, RVPtr z,
                        RVPtr vx, RVPtr vy, RVPtr vz,
                        RVPtr ax, RVPtr ay, RVPtr az,
                        RVPtr mass, RVPtr xMax, RVPtr yMax,
                        RVPtr zMax, RVPtr xMin, RVPtr yMin,
                        RVPtr zMin, UVPtr mCodes_G, UVPtr iteration, UVPtr bodyParents,
                        NVPtr gpuBinaryTree, NVPtr gpuLeafs, UVPtr nodeCounts, NVPtr octree){



    uint g = (uint) get_global_id(0);
    uint l = (uint) get_local_id(0);
    uint group = (uint) get_group_id(0);

    uint index = 0;
    uint leafFound = 0;
    uint chunkLevel = 0;

    while(leafFound == 0){
        uint currentChunk = extractBits(mCodes_G[g], 9 - chunkLevel);
        if(octree[index].children[currentChunk] > 0){
            index = octree[index].children[currentChunk];
        }
        else{
            octree[index].leafIndex[currentChunk] = g;
            int i = 0;
            while(mCodes_G[g] == mCodes_G[g + i]){
                bodyParents[g + i] = index;
                ++i;
            }
            octree[index].children[currentChunk] = -i;
            leafFound = 1;
        }
        ++chunkLevel;
    }
}

kernel void threadOctree(NVPtr octree){
    uint g = (uint) get_global_id(0);

    uint chunk = extractBits(octree[g].prefix, 0);
    int nextFound = 0;

    uint parentNodeIndex = octree[g].parent;
    uint parentTreeLevel = octree[octree[g].parent].treeLevel;
    for(int i = 0; i < 8; ++i){
        if(octree[g].children[i] > 0){
            octree[g].more = octree[g].children[i];
            break;
        }
    }

    // octree[g].next = chunk;
    if(g != 0){
        while(nextFound != 1){
            for(int i = chunk + 1; i < 8; ++i){
                if(octree[parentNodeIndex].children[i] != 0 && nextFound != 1){
                    octree[g].next = octree[octree[g].parent].children[i];
                    nextFound = 1;
                }
            }
            if(parentNodeIndex == 0){
                uint currentIndex = 0;
                while(octree[currentIndex].treeLevel != parentTreeLevel){
                    currentIndex = octree[currentIndex].more;
                }
                nextFound = 1;
            }
            chunk = extractBits(octree[parentNodeIndex].prefix, 0);
            parentNodeIndex = octree[parentNodeIndex].parent;
        }
    }
}

kernel void forceCalculationTreecode(RVPtr x, RVPtr y, RVPtr z,
                                        RVPtr vx, RVPtr vy, RVPtr vz,
                                        RVPtr ax, RVPtr ay, RVPtr az,
                                        RVPtr mass, UVPtr bodyParents, NVPtr octree){
    uint g = (uint) get_global_id(0);
    uint currentIndex = g;

    //If center of mass of node is too close, particle must sum forces to all particles within that cell then go down another layer


    //Sum to bodies in current cell
    currentIndex = octree[g].parent;
    for(int i = 0; i < 8; ++i){
        //Calculate acceleration
    }
    //Sum to all other bodies
    while(currentIndex != g){
        currentIndex = octree[currentIndex].next;
        //while distance is less than rcrit:
            //currentIndex = octree[currentIndex].more
        //Calculate acceleration

        ////////////
        break;
        ////////////
    }

}

kernel void verifyOctree(NVPtr octree, UVPtr verifArry){
}

kernel void zeroBuffers(RVPtr x, RVPtr y, RVPtr z,
                        RVPtr vx, RVPtr vy, RVPtr vz,
                        RVPtr ax, RVPtr ay, RVPtr az,
                        RVPtr mass, RVPtr xMax, RVPtr yMax,
                        RVPtr zMax, RVPtr xMin, RVPtr yMin,
                        RVPtr zMin, UVPtr mCodes_G, UVPtr iteration, UVPtr bodyParents,
                        NVPtr gpuBinaryTree, NVPtr gpuLeafs, UVPtr nodeCounts, NVPtr octree, UVPtr swap){

    uint g = (uint) get_global_id(0);
    if(g == 0){
        *iteration = 0;
    }
    if(g < NBODY){
        xMin[g] = xMax[g] = x[g];
        yMin[g] = yMax[g] = y[g];
        zMin[g] = zMax[g] = z[g];
    }
    else{
        xMin[g] = DBL_MAX;
        yMin[g] = DBL_MAX;
        zMin[g] = DBL_MAX;
        xMax[g] = -DBL_MAX;
        yMax[g] = -DBL_MAX;
        zMax[g] = -DBL_MAX;
    }

    mCodes_G[g] = 0;
    nodeCounts[g] = 0;
    swap[g] = 0;
    bodyParents[g] = 0;
    ax[g] = 0;
    ay[g] = 0;
    az[g] = 0;
    octree[g].parent = gpuLeafs[g].parent = gpuBinaryTree[g].parent = 0;
    octree[g].next = gpuLeafs[g].next = gpuBinaryTree[g].next = 0;
    octree[g].more = gpuLeafs[g].more = gpuBinaryTree[g].more = 0;
    octree[g].prefix = gpuLeafs[g].prefix = gpuBinaryTree[g].prefix = 0;
    octree[g].delta = gpuLeafs[g].delta = gpuBinaryTree[g].delta = 0;
    octree[g].treeLevel = gpuLeafs[g].treeLevel = gpuBinaryTree[g].treeLevel = 0;
    octree[g].mortonCode = gpuLeafs[g].mortonCode = gpuBinaryTree[g].mortonCode = 0;
    octree[g].id = gpuLeafs[g].id = gpuBinaryTree[g].id = 0;
    octree[g].pid = gpuLeafs[g].pid = gpuBinaryTree[g].pid = 0;
    octree[g].massEnclosed = gpuLeafs[g].massEnclosed = gpuBinaryTree[g].massEnclosed = 0;

    for(int i = 0; i < 8; ++i){
        octree[g].children[i] = gpuLeafs[g].children[i] = gpuBinaryTree[g].children[i] = 0;
        octree[g].leafIndex[i] = gpuLeafs[g].leafIndex[i] = gpuBinaryTree[g].leafIndex[i] = -1;
    }
    for(int i = 0; i < 3; ++i){
        octree[g].com[i] = gpuLeafs[g].com[i] = gpuBinaryTree[g].com[i] = 0;
    }
}

kernel void computeNodeStats(RVPtr x, RVPtr y, RVPtr z,
                                RVPtr vx, RVPtr vy, RVPtr vz,
                                RVPtr ax, RVPtr ay, RVPtr az,
                                RVPtr mass, RVPtr xMax, RVPtr yMax,
                                RVPtr zMax, RVPtr xMin, RVPtr yMin,
                                RVPtr zMin, UVPtr mCodes_G, UVPtr iteration,
                                NVPtr gpuBinaryTree, NVPtr gpuLeafs, UVPtr nodeCounts, NVPtr octree){
    
    uint g = (uint) get_global_id(0);
    uint l = (uint) get_local_id(0);
    uint group = (uint) get_group_id(0);

}