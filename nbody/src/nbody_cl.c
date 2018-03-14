/*
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

#include <time.h> //Time how long filling GPU Vector takes

#include "milkyway_cl.h"
#include "milkyway_util.h"
#include "nbody_cl.h"
#include "nbody_show.h"
#include "nbody_util.h"
#include "nbody_curses.h"
#include "nbody_shmem.h"
#include "nbody_checkpoint.h"
#include "nbody_tree.h"
#include "nbody_types.h" //Include Dynamic GPU Vector

#ifdef NBODY_BLENDER_OUTPUT
    #include "blender_visualizer.h"
#endif

/* We want to restrict this a bit to ensure we can get better occupancy.
   TODO: Option to raise the maximum depth at expense of performance.
 */
#define NB_MAX_MAX_DEPTH 128

extern const unsigned char nbody_kernels_cl[];
extern const size_t nbody_kernels_cl_len;


typedef struct MW_ALIGN_TYPE_V(64)
{
    real radius;
    cl_int bottom;
    cl_uint maxDepth;
    cl_uint blkCnt;
    cl_int doneCnt;

    cl_int errorCode;
    cl_int assertionLine;

    char _pad[64 - (1 * sizeof(real) + 6 * sizeof(cl_int))];

    struct
    {
        real f[32];
        cl_int i[64];
        cl_int wg1[256];
        cl_int wg2[256];
        cl_int wg3[256];
        cl_int wg4[256];
    } debug;
} TreeStatus;


static cl_ulong nbCalculateDepthLimitationFromCalculatedForceKernelLocalMemoryUsage(const DevInfo* di, const NBodyWorkSizes* ws, cl_bool useQuad)
{
    cl_ulong estMaxDepth;
    cl_ulong wgSize = ws->threads[5];
    cl_ulong warpPerWG = wgSize / di->warpSize;

    /* I don't trust the device parameters reporting anymore */
    cl_ulong localMemSize = (di->localMemSize > 0) ? di->localMemSize : 16384;

    /* Pieces which are not part of the "stack" */
    cl_ulong maxDepth = sizeof(cl_int);
    cl_ulong rootCritRadius = sizeof(real);
    cl_ulong allBlock = warpPerWG * sizeof(cl_int);

    cl_ulong ch = warpPerWG * sizeof(cl_int);
    cl_ulong nx = warpPerWG * sizeof(cl_int);
    cl_ulong ny = warpPerWG * sizeof(cl_int);
    cl_ulong nz = warpPerWG * sizeof(cl_int);
    cl_ulong nm = warpPerWG * sizeof(cl_int);

    cl_ulong constantPieces = maxDepth + rootCritRadius + allBlock + ch + nx + ny + nz + nm;

    /* Individual sizes of elements on the cell stack. */
    cl_ulong pos = sizeof(cl_int);
    cl_ulong node = sizeof(cl_int);
    cl_ulong dq = sizeof(real);
    cl_ulong quadPieces = 6 * sizeof(real);

    cl_ulong stackItemCount = pos + node + dq;

    if (useQuad)
    {
        stackItemCount += quadPieces;
    }

    /* We now have the size requirement as:
       d * warpPerWG * stackItemCount + constantPieces <= localMemSize
       Solve for d.
    */
    estMaxDepth = (localMemSize - constantPieces) / (warpPerWG * stackItemCount);

    return estMaxDepth - 1;  /* A bit extra will be used. Some kind of rounding up */
}

//NOTE: not sure what this function actually does. Might not need if it's tree related.
cl_uint nbFindMaxDepthForDevice(const DevInfo* di, const NBodyWorkSizes* ws, cl_bool useQuad)
{
    cl_ulong d;

    d = nbCalculateDepthLimitationFromCalculatedForceKernelLocalMemoryUsage(di, ws, useQuad);

    /* TODO: We should be able to reduce this; this is usually quite a
     * bit deeper than we can go before hitting precision limits */

    return (cl_uint) mwMin(d, NB_MAX_MAX_DEPTH);
}


//TODO: EDIT FOR CURRENT WORKGROUPS:
static void nbPrintNBodyWorkSizes(const NBodyWorkSizes* ws)
{
    mw_printf("\n"
              "Kernel launch sizes:\n"
              "  Bounding box kernel:  "ZU", "ZU"\n"
              "  Tree build kernel:    "ZU", "ZU"\n"
              "  Summarization kernel: "ZU", "ZU"\n"
              "  Sort kernel:          "ZU", "ZU"\n"
              "  Quadrupole kernel:    "ZU", "ZU"\n"
              "  Force kernel:         "ZU", "ZU"\n"
              "  Integration kernel:   "ZU", "ZU"\n"
              "\n",
              ws->global[0], ws->local[0],
              ws->global[1], ws->local[1],
              ws->global[2], ws->local[2],
              ws->global[3], ws->local[3],
              ws->global[4], ws->local[4],
              ws->global[5], ws->local[5],
              ws->global[6], ws->local[6]
        );
}

/* In case work sizes are larger than the maximum, clamp them to the maximum */
static cl_uint nbClampWorkSizes(NBodyWorkSizes* ws, const DevInfo* di)
{
    cl_uint i;
    cl_uint clamped = 0;
    size_t maxGlobalSize = di->maxWorkItemSizes[0] * di->maxWorkItemSizes[1] * di->maxWorkItemSizes[2];

    for (i = 0; i < 8; ++i)
    {
        if (ws->global[i] > maxGlobalSize)
        {
            if (mwDivisible(maxGlobalSize, ws->local[i]))
            {
                ws->global[i] = maxGlobalSize;
            }
            else
            {
                ws->global[i] = mwNextMultiple(ws->local[i], maxGlobalSize - ws->local[i]);
            }

            ++clamped;
        }
    }

    if (clamped != 0)
    {
        mw_printf("Warning: %u work sizes clamped to maximum\n", clamped);
    }

    return clamped;
}

cl_bool nbSetWorkSizes(NBodyWorkSizes* ws, const DevInfo* di, cl_int nbody, cl_bool ignoreResponsive)
{
    cl_uint i;
    cl_uint blocks = di->maxCompUnits;

    for (i = 0; i < 8; ++i)
    {
        ws->global[i] = ws->threads[i] * ws->factors[i] * blocks;
        ws->local[i] = ws->threads[i];
    }

    nbClampWorkSizes(ws, di);

    return CL_FALSE;
}

/* CHECKME: May not apply on GT200? */
static cl_bool nbShouldForceLargeGroup(const DevInfo* di, const NBodyCtx* ctx)
{
    return !ctx->useQuad && mwIsNvidiaGPUDevice(di) && mwHasNvidiaCompilerFlags(di);
}

static const char* nbMaybeNvMaxRegCount(const DevInfo* di, const NBodyCtx* ctx)
{
    return nbShouldForceLargeGroup(di, ctx) ? "-cl-nv-maxrregcount=32 " : "";
}

/* Return CL_TRUE if some error */
cl_bool nbSetThreadCounts(NBodyWorkSizes* ws, const DevInfo* di, const NBodyCtx* ctx)
{
    /* Numbers need playing for float and different opening criteria */

    ws->factors[0] = 1;
    ws->factors[1] = 1;
    ws->factors[2] = 1;  /* Must be 1. All workitems must be resident */
    ws->factors[3] = 1;  /* Also must be 1 for the same reason */
    ws->factors[4] = 1;  /* Also must be 1 for the same reason */
    ws->factors[5] = 1;
    ws->factors[6] = 1;
    ws->factors[7] = 1;

    ws->threads[0] = 32;
    ws->threads[1] = 32;
    ws->threads[2] = 32;
    ws->threads[3] = 32;
    ws->threads[4] = 32;
    ws->threads[5] = 32;
    ws->threads[6] = 32;
    ws->threads[7] = 32;


    if (di->devType == CL_DEVICE_TYPE_CPU)
    {
        ws->threads[0] = 1;
        ws->threads[1] = 1;
        ws->threads[2] = 1;
        ws->threads[3] = 1;
        ws->threads[4] = 1;
        ws->threads[5] = 1;
        ws->threads[6] = 1;
        ws->threads[7] = 1;
    }
    else if (mwComputeCapabilityIs(di, 1, 3))
    {
        ws->threads[0] = 256;
        ws->threads[1] = 288;
        ws->threads[2] = 256;
        ws->threads[3] = 512;
        ws->threads[4] = 256;

        /* TODO: We can decrease the thread count and rebuild if we
         * hit the much lower maximum depths on local memory
         * constrained GPUs
         */
        if (ctx->useQuad && DOUBLEPREC)
        {
            /* Constrains to maxdepth ~= 30. Higher brings it unacceptably low */
            ws->threads[5] = 256;
        }
        else
        {
            ws->threads[5] = 384;
        }

        ws->threads[6] = 512;
        ws->threads[7] = 448;
    }
    else if (mwMinComputeCapabilityCheck(di, 2, 0))
    {
        ws->factors[0] = 1;
        ws->factors[1] = 1;
        ws->factors[2] = 1;
        ws->factors[3] = 1;
        ws->factors[4] = 1;
        ws->factors[5] = 1;
        ws->factors[6] = 4;
        ws->factors[7] = 4;

        ws->threads[0] = 32;
        ws->threads[1] = 64;
        ws->threads[2] = 128;
        ws->threads[3] = 256;
        ws->threads[4] = 512;

        /* It's faster to restrain the used number of registers and
         * get a larger workgroup size, but when using quadrupole
         * moments this gives a very small constraining maximum depth */
        ws->threads[5] = nbShouldForceLargeGroup(di, ctx) ? 1024 : 512;

        ws->threads[6] = 1024;
        ws->threads[7] = 1024;
    }
    else if (di->calTarget >= MW_CAL_TARGET_TAHITI)
    {
        ws->factors[0] = 1;
        ws->factors[1] = 4;
        ws->factors[2] = 4;
        ws->factors[3] = 1;
        ws->factors[4] = 4;
        ws->factors[5] = 4 * (4 * 10); /* Max at 10 per vector unit, 4 vector units */
        ws->factors[6] = 4 * 10;
        ws->factors[7] = 2;

        ws->threads[0] = 256;
        ws->threads[1] = 256;
        ws->threads[2] = 256;
        ws->threads[3] = 256;
        ws->threads[4] = 256;
        ws->threads[5] = 64;
        ws->threads[6] = 64;
        ws->threads[7] = 256;
    }
    else
    {
        ws->factors[0] = 1;
        ws->factors[1] = 1;
        ws->factors[2] = 1;
        ws->factors[3] = 1;
        ws->factors[4] = 1;
        ws->factors[5] = 1;
        ws->factors[6] = 2;
        ws->factors[7] = 1;

        ws->threads[0] = 256;
        ws->threads[1] = 256;
        ws->threads[2] = 256;
        ws->threads[3] = 256;
        ws->threads[4] = 256;
        ws->threads[5] = 256;
        ws->threads[6] = 256;
        ws->threads[7] = 256;
    }

    return CL_FALSE;
}

//NOTE: This is an important function:
static void* mapBuffer(CLInfo* ci, cl_mem mem, cl_map_flags flags, size_t size)
{
    return clEnqueueMapBuffer(ci->queue, mem, CL_TRUE, flags, 0,
                              size,
                              0, NULL, NULL, NULL);
}

static void nbPrintDebug(const TreeStatus* ts)
{
    int i;
    for (i = 0; i < 32; ++i)
    {
        mw_printf("Debug.int[%d] = %d\n", i, ts->debug.i[i]);
    }

    for (i = 0; i < 32; ++i)
    {
        mw_printf("Debug.float[%d] = %.15f\n", i, ts->debug.f[i]);
    }
}

//NOTE: NOT NEEDED FOR CURRENT IMPLEMENTATION?
static void nbPrintTreeStatus(const TreeStatus* ts)
{
    mw_printf("TreeStatus = {\n"
              "  radius        = %.15f\n"
              "  bottom        = %d\n"
              "  maxDepth      = %d\n"
              "  blckCnt       = %u\n"
              "  doneCnt       = %u\n"
              "  errorCode     = %d\n"
              "  assertionLine = %d\n"
              "}\n",
              ts->radius,
              ts->bottom,
              ts->maxDepth,
              ts->blkCnt,
              ts->doneCnt,
              ts->errorCode,
              ts->assertionLine
        );
}

/* Set arguments (idx, idx + 1, idx + 2) to the buffers in mem[3] */
static cl_int nbSetMemArrayArgs(cl_kernel kern, cl_mem mem[3], cl_uint idx)
{
    cl_uint i;
    cl_int err = CL_SUCCESS;

    for (i = 0; i < 3; ++i)
    {
        err |= clSetKernelArg(kern, idx + i, sizeof(cl_mem), &mem[i]);
    }

    return err;
}

//NOTE: UPDATE TO CURRENT ARGS
static cl_int nbSetKernelArguments(cl_kernel kern, NBodyBuffers* nbb, cl_bool exact)
{
    cl_int err = CL_SUCCESS;
    cl_int zeroVal = 0;
    if (!exact)
    {
        err = nbSetMemArrayArgs(kern, nbb->pos, 0);
        err = nbSetMemArrayArgs(kern, nbb->vel, 3);
        err = nbSetMemArrayArgs(kern, nbb->acc, 6);
        err = clSetKernelArg(kern, 9, sizeof(cl_mem), &(nbb->mass) );
        err = nbSetMemArrayArgs(kern, nbb->max, 10);
        err = nbSetMemArrayArgs(kern, nbb->min, 13);
        err = clSetKernelArg(kern, 16, sizeof(cl_mem), &(nbb->mCodes));
        err = clSetKernelArg(kern, 17, sizeof(cl_mem), &(nbb->iteration));
    }
    else
    {
        err = nbSetMemArrayArgs(kern, nbb->pos, 0);
        err = nbSetMemArrayArgs(kern, nbb->vel, 3);
        err = nbSetMemArrayArgs(kern, nbb->acc, 6);
        err = clSetKernelArg(kern, 9, sizeof(cl_mem), &(nbb->mass) );
    }

    return err;
}

// static cl_int nbSetKernelArgumentsOutput(cl_kernel kern, NBodyBuffers* nbb, cl_bool exact){
//   cl_int err = CL_SUCCESS;
//   cl_int zeroVal = 0;
//   if (!exact)
//   {
//       err = clSetKernelArg(kern, 0, sizeof(cl_mem), &(nbb->input) );
//       err = clSetKernelArg(kern, 1, sizeof(cl_mem), &(nbb->output) );        
//   }
//   else
//   {
//       err = clSetKernelArg(kern, 0, sizeof(cl_mem), &(nbb->input) );
//       err = clSetKernelArg(kern, 1, sizeof(cl_mem), &(nbb->output) );
//   }

//   return err;
// }

//NOTE: UPDATE TO CURRENT ARGS
cl_int nbSetAllKernelArguments(NBodyState* st)
{
    cl_int err = CL_SUCCESS;
    NBodyKernels* k = st->kernels;
    cl_bool exact = st->usesExact;

    if (!exact)
    {
        err |= nbSetKernelArguments(k->boundingBox, st->nbb, exact);
        err |= nbSetKernelArguments(k->encodeTree, st->nbb, exact);
        err |= nbSetKernelArguments(k->bitonicMortonSort, st->nbb, exact);
        err |= nbSetKernelArguments(k->constructTree, st->nbb, exact);
        err |= nbSetKernelArguments(k->countOctNodes, st->nbb, exact);
        err |= nbSetKernelArguments(k->constructOctTree, st->nbb, exact);
        err |= nbSetKernelArguments(k->linkOctree, st->nbb, exact);
        err |= nbSetKernelArguments(k->zeroBuffers, st->nbb, exact);
//         err |= nbSetKernelArguments(k->buildTreeClear, st->nbb, exact);
//         err |= nbSetKernelArguments(k->buildTree, st->nbb, exact);
//         err |= nbSetKernelArguments(k->summarizationClear, st->nbb, exact);
//         err |= nbSetKernelArguments(k->summarization, st->nbb, exact);
//         err |= nbSetKernelArguments(k->sort, st->nbb, exact);
//         err |= nbSetKernelArguments(k->quadMoments, st->nbb, exact);
//         err |= nbSetKernelArguments(k->forceCalculation, st->nbb, exact);
//         err |= nbSetKernelArguments(k->integration, st->nbb, exact);
    }
    else
    {
        //TESTING: Return to forceCalculation_Exact:
        err |= nbSetKernelArguments(k->forceCalculationExact, st->nbb, exact);
        err |= nbSetKernelArguments(k->advanceHalfVelocity, st->nbb, exact);
        err |= nbSetKernelArguments(k->advancePosition, st->nbb, exact);
        // err |= nbSetKernelArguments(k->outputData, st->nbb, exact);
        //err |= nbSetKernelArguments(k->integration, st->nbb, exact);
    }

    if (err != CL_SUCCESS)
    {
        mwPerrorCL(err, "Error setting kernel arguments");
    }

    return err;
}

//NOTE: UPDATE KERNELS
static cl_int clReleaseKernel_quiet(cl_kernel kern)
{
    return kern ? clReleaseKernel(kern) : CL_SUCCESS;
}

//NOTE: UPDATE KERNELS
cl_int nbReleaseKernels(NBodyState* st)
{
    cl_int err = CL_SUCCESS;
    NBodyKernels* kernels = st->kernels;

    err |= clReleaseKernel_quiet(kernels->boundingBox);
    err |= clReleaseKernel_quiet(kernels->encodeTree);
    err |= clReleaseKernel_quiet(kernels->bitonicMortonSort);
    err |= clReleaseKernel_quiet(kernels->constructTree);
    err |= clReleaseKernel_quiet(kernels->countOctNodes);
    err |= clReleaseKernel_quiet(kernels->prefixSumUpsweep);
    err |= clReleaseKernel_quiet(kernels->prefixSumDownsweep);
    err |= clReleaseKernel_quiet(kernels->prefixSumInclusiveUtil);
    err |= clReleaseKernel_quiet(kernels->constructOctTree);
    err |= clReleaseKernel_quiet(kernels->linkOctree);
    err |= clReleaseKernel_quiet(kernels->threadOctree);
    err |= clReleaseKernel_quiet(kernels->forceCalculationTreecode);
    err |= clReleaseKernel_quiet(kernels->zeroBuffers);

//     err |= clReleaseKernel_quiet(kernels->buildTreeClear);
//     err |= clReleaseKernel_quiet(kernels->buildTree);
//     err |= clReleaseKernel_quiet(kernels->summarizationClear);
//     err |= clReleaseKernel_quiet(kernels->summarization);
//     err |= clReleaseKernel_quiet(kernels->quadMoments);
//     err |= clReleaseKernel_quiet(kernels->sort);
     err |= clReleaseKernel_quiet(kernels->forceCalculationExact);
     err |= clReleaseKernel_quiet(kernels->advancePosition);
     err |= clReleaseKernel_quiet(kernels->advanceHalfVelocity);
     //err |= clReleaseKernel_quiet(kernels->outputData);

    if (err != CL_SUCCESS)
        mwPerrorCL(err, "Error releasing kernels");

    return err;
}

//NOTE: NOT NEEDED
static cl_uint nbFindNNode(const DevInfo* di, cl_int nbody)
{
    cl_uint nNode = 2 * nbody;

    if (nNode < 1024 * di->maxCompUnits)
        nNode = 1024 * di->maxCompUnits;
    while ((nNode & (di->warpSize - 1)) != 0)
        ++nNode;

    return nNode - 1;
}

#ifndef NDEBUG
  #define NBODY_DEBUG_KERNEL 1
#else
  #define NBODY_DEBUG_KERNEL 0
#endif

static char* nbGetCompileFlags(const NBodyCtx* ctx, const NBodyState* st, const DevInfo* di)
{
    char* buf;
    const NBodyWorkSizes* ws = st->workSizes;
    const Potential* p = &ctx->pot;

    if (asprintf(&buf,
                 "-DDEBUG=%d "

                 "-DDOUBLEPREC=%d "
               #if !DOUBLEPREC
                 "-cl-single-precision-constant "
               #endif
                 "-cl-mad-enable "
                 //"-cl-opt-disable "

                 "-DNBODY=%d "
                 "-DEFFNBODY=%d "
                 "-DNNODE=%u "
                 "-DWARPSIZE=%u "

                 "-DNOSORT=%d "

                 "-DTHREADS1="ZU" "
                 "-DTHREADS2="ZU" "
                 "-DTHREADS3="ZU" "
                 "-DTHREADS4="ZU" "
                 "-DTHREADS5="ZU" "
                 "-DTHREADS6="ZU" "
                 "-DTHREADS7="ZU" "
                 "-DTHREADS8="ZU" "

                 "-DMAXDEPTH=%u "

                 "-DTIMESTEP=%a "
                 "-DEPS2=%a "
                 "-DTHETA=%a "
                 "-DUSE_QUAD=%d "

                 "-DNEWCRITERION=%d "
                 "-DSW93=%d "
                 "-DBH86=%d "
                 "-DEXACT=%d "

                 /* Potential */
                 "-DUSE_EXTERNAL_POTENTIAL=%d "
                 "-DMIYAMOTO_NAGAI_DISK=%d "
                 "-DEXPONENTIAL_DISK=%d "
                 "-DLOG_HALO=%d "
                 "-DNFW_HALO=%d "
                 "-DTRIAXIAL_HALO=%d "

                 /* Spherical constants */
                 "-DSPHERICAL_MASS=%a "
                 "-DSPHERICAL_SCALE=%a "

                 /* Disk constants */
                 "-DDISK_MASS=%a "
                 "-DDISK_SCALE_LENGTH=%a "
                 "-DDISK_SCALE_HEIGHT=%a "

                 /* Halo constants */
                 "-DHALO_VHALO=%a "
                 "-DHALO_SCALE_LENGTH=%a "
                 "-DHALO_FLATTEN_Z=%a "
                 "-DHALO_FLATTEN_Y=%a "
                 "-DHALO_FLATTEN_X=%a "
                 "-DHALO_TRIAX_ANGLE=%a "
                 "-DHALO_C1=%a "
                 "-DHALO_C2=%a "
                 "-DHALO_C3=%a "

                 "%s "
                 "%s "
                 "-DHAVE_INLINE_PTX=%d "
                 "-DHAVE_CONSISTENT_MEMORY=%d ",
                 NBODY_DEBUG_KERNEL,
                 DOUBLEPREC,

                 st->nbody,
                 st->effNBody,
                 nbFindNNode(di, st->nbody),
                 di->warpSize,

                 (di->devType == CL_DEVICE_TYPE_CPU),

                 ws->threads[0],
                 ws->threads[1],
                 ws->threads[2],
                 ws->threads[3],
                 ws->threads[4],
                 ws->threads[5],
                 ws->threads[6],
                 ws->threads[7],

                 st->maxDepth,

                 ctx->timestep,
                 ctx->eps2,
                 ctx->theta,
                 ctx->useQuad,

                 /* Set criterion */
                 ctx->criterion == NewCriterion,
                 ctx->criterion == SW93,
                 ctx->criterion == BH86,
                 ctx->criterion == Exact,


                 /* Set potential */
                 ctx->potentialType == EXTERNAL_POTENTIAL_DEFAULT,

                 p->disk.type == MiyamotoNagaiDisk,
                 p->disk.type == ExponentialDisk,
                 p->halo.type == LogarithmicHalo,
                 p->halo.type == NFWHalo,
                 p->halo.type == TriaxialHalo,

                 /* Set potential constants */
                 /* Spherical constants */
                 p->sphere[0].mass,
                 p->sphere[0].scale,

                 /* Disk constants */
                 p->disk.mass,
                 p->disk.scaleLength,
                 p->disk.scaleHeight,

                 /* Halo constants */
                 p->halo.vhalo,
                 p->halo.scaleLength,
                 p->halo.flattenZ,
                 p->halo.flattenY,
                 p->halo.flattenX,
                 p->halo.triaxAngle,
                 p->halo.c1,
                 p->halo.c2,
                 p->halo.c3,

                 /* Misc. other stuff */
                 mwHasNvidiaCompilerFlags(di) ? "-cl-nv-verbose" : "",
                 nbMaybeNvMaxRegCount(di, ctx),
                 mwNvidiaInlinePTXAvailable(st->ci->plat),
                 st->usesConsistentMemory
            ) < 1)
    {
        mw_printf("Error getting compile flags\n");
        return NULL;
    }

    return buf;
}

/* Yo momma's so fat she has little mommas in orbit around her. */
//NOTE: UPDATE KERNELS
static cl_bool nbCreateKernels(cl_program program, NBodyKernels* kernels)
{
//    kernels->testAddition = mwCreateKernel(program, "testAddition");
    kernels->boundingBox = mwCreateKernel(program, "boundingBox");
    kernels->encodeTree = mwCreateKernel(program, "encodeTree");
    kernels->bitonicMortonSort = mwCreateKernel(program, "bitonicMortonSort");
    kernels->constructTree = mwCreateKernel(program, "constructTree");
    kernels->countOctNodes = mwCreateKernel(program, "countOctNodes");
    kernels->prefixSum = mwCreateKernel(program, "prefixSum");
    kernels->prefixSumUpsweep = mwCreateKernel(program, "prefixSumUpsweep");
    kernels->prefixSumDownsweep = mwCreateKernel(program, "prefixSumDownsweep");
    kernels->prefixSumInclusiveUtil = mwCreateKernel(program, "prefixSumInclusiveUtil");
    kernels->constructOctTree = mwCreateKernel(program, "constructOctTree");
    kernels->linkOctree = mwCreateKernel(program, "linkOctree");
    kernels->threadOctree = mwCreateKernel(program, "threadOctree");
    kernels->forceCalculationTreecode = mwCreateKernel(program, "forceCalculationTreecode");
    kernels->zeroBuffers = mwCreateKernel(program, "zeroBuffers");
//     kernels->buildTreeClear = mwCreateKernel(program, "buildTreeClear");
//     kernels->buildTree = mwCreateKernel(program, "buildTree");
//     kernels->summarizationClear = mwCreateKernel(program, "summarizationClear");
//     kernels->summarization = mwCreateKernel(program, "summarization");
//     kernels->quadMoments = mwCreateKernel(program, "quadMoments");
//     kernels->sort = mwCreateKernel(program, "sort");
    // kernels->forceCalculation = mwCreateKernel(program, "forceCalculation");
    //kernels->integration = mwCreateKernel(program, "integration");
    kernels->forceCalculationExact = mwCreateKernel(program, "forceCalculationExact");
    kernels->advanceHalfVelocity = mwCreateKernel(program, "advanceHalfVelocity");
    kernels->advancePosition = mwCreateKernel(program, "advancePosition");
    // kernels->outputData = mwCreateKernel(program, "outputData");
    return(     kernels->boundingBox
            &&  kernels->encodeTree
            &&  kernels->prefixSumUpsweep
            &&  kernels->prefixSumDownsweep
            &&  kernels->prefixSumInclusiveUtil
            &&  kernels->constructTree
            &&  kernels->countOctNodes
            &&  kernels->linkOctree
            &&  kernels->threadOctree
            &&  kernels->forceCalculationTreecode
            &&  kernels->zeroBuffers
            &&  kernels->forceCalculationExact
            &&  kernels->advanceHalfVelocity
            &&  kernels->advancePosition);
}

//NOTE: This function is necessary:
cl_bool nbLoadKernels(const NBodyCtx* ctx, NBodyState* st)
{
    CLInfo* ci = st->ci;
    char* compileFlags = NULL;
    cl_program program;
    const char* src = (const char*) nbody_kernels_cl;
    size_t srcLen = nbody_kernels_cl_len;

    compileFlags = nbGetCompileFlags(ctx, st, &ci->di);
    assert(compileFlags);

    program = mwCreateProgramFromSrc(ci, 1, &src, &srcLen, compileFlags);
    free(compileFlags);
    if (!program)
    {
        mw_printf("Failed to create program\n");
        return CL_TRUE;
    }

    if (!nbCreateKernels(program, st->kernels))
        return CL_TRUE;

    clReleaseProgram(program);

    return CL_FALSE;
}

/* Return CL_FALSE if device isn't capable of running this */
cl_bool nbCheckDevCapabilities(const DevInfo* di, const NBodyCtx* ctx, cl_uint nbody)
{
    cl_ulong nNode = (cl_ulong) nbFindNNode(di, nbody) + 1;
    cl_ulong maxNodes = di->maxMemAlloc / (NSUB * sizeof(cl_int));

    (void) ctx;

    if (di->devType != CL_DEVICE_TYPE_GPU)
    {
        mw_printf("Device is not a GPU.\n");
        return CL_FALSE;
    }

    if (!mwIsNvidiaGPUDevice(di) && !mwIsAMDGPUDevice(di))
    {
        /* There is reliance on implementation details for Nvidia and
         * AMD GPUs. If some other kind of GPU decides to exist, it
         * would need to be tested.*/
        mw_printf("Only Nvidia and AMD GPUs are supported\n");
        return CL_FALSE;
    }

    if (DOUBLEPREC && !mwSupportsDoubles(di))
    {
        mw_printf("Device does not have usable double precision extension\n");
        return CL_FALSE;
    }

    if (   !strstr(di->exts, "cl_khr_global_int32_base_atomics")
        || !strstr(di->exts, "cl_khr_global_int32_extended_atomics")
        || !strstr(di->exts, "cl_khr_local_int32_base_atomics"))
    {
        mw_printf("Device lacks necessary atomics extensions\n");
        return CL_FALSE;
    }

    if (nNode > maxNodes)
    {
        mw_printf("Simulation of %u bodies requires "LLU" nodes, "
                  "however maximum allocation size only allows for "LLU"\n",
                  nbody,
                  nNode,
                  maxNodes
            );
        return CL_FALSE;
    }

    /* if TAHITI and < 12.3 driver bug requires additional clFinishes */
    if (di->calTarget >= MW_CAL_TARGET_TAHITI
        && !mwAMDCLVersionMin(di, 900, 0) /* No idea, but if using new format new enough */
        && !mwCALVersionMin(di, 1, 4, 1720))
    {
        mw_printf("Tahiti driver bug requires Catalyst 12.3 or newer\n");
        return CL_FALSE;
    }

    return CL_TRUE;
}

//NOTE: NOT NEEDED FOR CURRENT IMPLEMENTATION?
// static cl_int nbEnqueueReadTreeStatus(TreeStatus* tc, CLInfo* ci, NBodyBuffers* nbb, cl_bool blocking)
// {
//     return clEnqueueReadBuffer(ci->queue,
//                               nbb->treeStatus,
//                               blocking,
//                               0, sizeof(*tc), tc,
//                               0, NULL, NULL);
// }

static cl_int printBuffer(CLInfo* ci, cl_mem mem, size_t n, const char* name, int type)
{
    size_t i;
    void* p;

    p = mapBuffer(ci, mem, CL_MAP_READ, n * (type == 0 ? sizeof(real) : sizeof(int)));
    if (!p)
    {
        mw_printf("Fail to map buffer for printing\n");
        return MW_CL_ERROR;
    }

    if (type == 0)
    {
        const real* pr = (const real*) p;
        for (i = 0; i < n; ++i)
        {
            mw_printf("%s["ZU"] = %.15f\n", name, i, pr[i]);
        }
    }
    else
    {
        const int* ip = (const int*) p;
        for (i = 0; i < n; ++i)
        {
            mw_printf("%s["ZU"] = %d\n", name, i, ip[i]);
        }
    }

    return clEnqueueUnmapMemObject(ci->queue, mem, p, 0, NULL, NULL);
}


//NOTE: UPDATE ERRORS
// static NBodyStatus nbKernelErrorToNBodyStatus(NBodyKernelError x)
// {
//     switch (x)
//     {
//         case NBODY_KERNEL_OK:
//             return NBODY_SUCCESS;
//         case NBODY_KERNEL_CELL_OVERFLOW:
//             return NBODY_CELL_OVERFLOW_ERROR;
//         case NBODY_KERNEL_TREE_INCEST:
//             return NBODY_TREE_INCEST_FATAL; /* Somewhat inaccurate but shouldn't happen  */
//         case NBODY_KERNEL_TREE_STRUCTURE_ERROR:
//             return NBODY_TREE_STRUCTURE_ERROR;
//         case NBODY_KERNEL_ERROR_OTHER:
//             return NBODY_ERROR;
//         default:
//             return NBODY_ERROR;
//     }



/* Check the error code */
//NOTE UPDATE ERROR CODES
// static NBodyStatus nbCheckKernelErrorCode(const NBodyCtx* ctx, NBodyState* st)
// {
//     cl_int err;
//     TreeStatus ts;
//     CLInfo* ci = st->ci;
//     NBodyBuffers* nbb = st->nbb;
// 
//     err = nbEnqueueReadTreeStatus(&ts, ci, nbb, CL_TRUE);
//     if (mw_unlikely(err != CL_SUCCESS))
//     {
//         mwPerrorCL(err, "Error reading tree status");
//         return NBODY_CL_ERROR;
//     }
// 
//     if (mw_unlikely(ts.assertionLine >= 0))
//     {
//         mw_printf("Kernel assertion failed: line %d\n", ts.assertionLine);
//         return NBODY_ASSERTION_FAILURE;
//     }
// 
//     if (mw_unlikely(ts.errorCode != 0))
//     {
//         /* Incest is special because we can choose to ignore it */
//         if (ts.errorCode == NBODY_KERNEL_TREE_INCEST)
//         {
//             nbReportTreeIncest(ctx, st);
//             return ctx->allowIncest ? NBODY_TREE_INCEST_NONFATAL : NBODY_TREE_INCEST_FATAL;
//         }
//         else
//         {
//             mw_printf("Kernel reported error: %d ", ts.errorCode);
// 
//             if (ts.errorCode > 0)
//             {
//                 mw_printf("(%s (%u))\n", showNBodyKernelError(ts.errorCode), st->maxDepth);
//                 return NBODY_MAX_DEPTH_ERROR;
//             }
//             else
//             {
//                 mw_printf("(%s)\n", showNBodyKernelError(ts.errorCode));
//                 return nbKernelErrorToNBodyStatus(ts.errorCode);
//             }
//         }
//     }
// 
//     return NBODY_SUCCESS;
// }

//NOTE: unsure what this function actually does, seems to wait for a CL event to happen then continue
static cl_double waitReleaseEventWithTime(cl_event ev)
{
    cl_double t;
    cl_int err;

    err = clWaitForEvents(1, &ev);
    if (err != CL_SUCCESS)
        return 0.0;

    t = mwEventTimeMS(ev);

    err = clReleaseEvent(ev);
    if (err != CL_SUCCESS)
        return 0.0;

    return t;
}

//NOTE: NOT NEEDED
// static cl_int nbEnqueueReadRootQuadMoment(NBodyState* st, NBodyQuadMatrix* quad)
// {
//     cl_int err = CL_SUCCESS;
//     NBodyBuffers* nbb = st->nbb;
//     cl_command_queue queue = st->ci->queue;
//     cl_uint nNode = nbFindNNode(&st->ci->di, st->nbody);
// 
//     if (!nbb->quad.xx)
//     {
//         return MW_CL_ERROR;
//     }
// 
//     err |= clEnqueueReadBuffer(queue, nbb->quad.xx, CL_FALSE, nNode * sizeof(real), sizeof(real), &quad->xx, 0, NULL, NULL);
//     err |= clEnqueueReadBuffer(queue, nbb->quad.xy, CL_FALSE, nNode * sizeof(real), sizeof(real), &quad->xy, 0, NULL, NULL);
//     err |= clEnqueueReadBuffer(queue, nbb->quad.xz, CL_FALSE, nNode * sizeof(real), sizeof(real), &quad->xz, 0, NULL, NULL);
// 
//     err |= clEnqueueReadBuffer(queue, nbb->quad.yy, CL_FALSE, nNode * sizeof(real), sizeof(real), &quad->yy, 0, NULL, NULL);
//     err |= clEnqueueReadBuffer(queue, nbb->quad.yz, CL_FALSE, nNode * sizeof(real), sizeof(real), &quad->yz, 0, NULL, NULL);
// 
//     err |= clEnqueueReadBuffer(queue, nbb->quad.zz, CL_FALSE, nNode * sizeof(real), sizeof(real), &quad->zz, 0, NULL, NULL);
// 
//     err |= clFlush(queue);
// 
//     return err;
// }

//NOTE: NOT NEEDED
static cl_int nbEnqueueReadCenterOfMass(NBodyState* st, mwvector* cmPos)
{
//     cl_int err = CL_SUCCESS;
//     cl_mem* positions = st->nbb->pos;
//     cl_command_queue queue = st->ci->queue;
//     cl_uint nNode = nbFindNNode(&st->ci->di, st->nbody);
// 
//     err |= clEnqueueReadBuffer(queue, positions[0], CL_FALSE, nNode * sizeof(real), sizeof(real), &cmPos->x, 0, NULL, NULL);
//     err |= clEnqueueReadBuffer(queue, positions[1], CL_FALSE, nNode * sizeof(real), sizeof(real), &cmPos->y, 0, NULL, NULL);
//     err |= clEnqueueReadBuffer(queue, positions[2], CL_FALSE, nNode * sizeof(real), sizeof(real), &cmPos->z, 0, NULL, NULL);
//     err |= clEnqueueReadBuffer(queue, st->nbb->masses, CL_FALSE, nNode * sizeof(real), sizeof(real), &cmPos->w, 0, NULL, NULL);
// 
//     err |= clFlush(queue);
// 
//     return err;
}

//NOTE: NOT NEEDED
int nbDisplayUpdateMarshalBodies(NBodyState* st, mwvector* cmPosOut)
{
    static cl_bool hadMarshalError = CL_FALSE;
    cl_int err;

    /* TODO: If this fails we are probably in a bad state and should abort everything */
    /* TODO: CL-GL sharing would be nice for CL graphics */

    if (hadMarshalError)
        return 1;

    err = nbEnqueueReadCenterOfMass(st, cmPosOut);
    if (err != CL_SUCCESS)
    {
        mwPerrorCL(err, "Failed to read center of mass");
        hadMarshalError = CL_TRUE;
        return 1;
    }

    err = nbMarshalBodies(st, CL_FALSE);
    if (err != CL_SUCCESS)
    {
        mwPerrorCL(err, "Error marshalling bodies for display update");
        hadMarshalError = CL_TRUE;
        return 1;
    }

    /* Center of mass read should be complete since body marshal uses blocking maps */

    return 0;
}

/* Run force calculation and integration kernels */
//NOTE: NOT NEEDED
static cl_int nbExecuteForceKernels(NBodyState* st, cl_bool updateState)
{
    cl_int err;
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event integrateEv;
    cl_kernel forceKern;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;

    //Determine which kernel to use:
    if (st->usesExact)
    {
        forceKern = kernels->forceCalculationExact;
        global[0] = st->effNBody;
        local[0] = ws->local[0];
    }
    else
    {
        forceKern = kernels->forceCalculationTreecode;
        global[0] = st->effNBody;
        local[0] = ws->local[0];
    }
    //Run kernel:
    
    /////////////////////////////////
    //Not sure what this chunk does:
    //nChunk = st->ignoreResponsive ? 1 : mwDivRoundup((size_t) effNBody, global[0]);
    //upperBound = st->ignoreResponsive ? effNBody : (cl_int) global[0];
    /////////////////////////////////
    
    //printf("%i and %i\n", global[0], local[0]);
    
    cl_event ev;

    //Not sure what this is for, seems like it just sets max nbody:
    //err = clSetKernelArg(forceKern, 28, sizeof(cl_uint), &upperBound);
    //if (err != CL_SUCCESS)
        //return err;
    
    err = clEnqueueNDRangeKernel(ci->queue, forceKern, 1,
                                    0, global, local,
                                    0, NULL, &ev);
    if (err != CL_SUCCESS)
        return err;
    clWaitForEvents(1, &ev);
    clFinish(ci->queue);
    return CL_SUCCESS;
}


static cl_int nbAdvanceHalfVelocity(NBodyState* st, cl_bool updateState)
{
    cl_int err;
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event integrateEv;
    cl_kernel velKern;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;

    
    velKern = kernels->advanceHalfVelocity;
    global[0] = st->effNBody;
    local[0] = ws->local[0];
    
    cl_event ev;
    err = clEnqueueNDRangeKernel(ci->queue, velKern, 1,
                                    0, global, local,
                                    0, NULL, &ev);
    if (err != CL_SUCCESS)
        return err;
    clWaitForEvents(1, &ev);
    clFinish(ci->queue);
    return CL_SUCCESS;
}

static cl_int nbAdvancePosition(NBodyState* st, cl_bool updateState)
{
    cl_int err;
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event integrateEv;
    cl_kernel posKern;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;

    
    posKern = kernels->advancePosition;
    global[0] = st->effNBody;
    local[0] = ws->local[0];
    
    cl_event ev;
    err = clEnqueueNDRangeKernel(ci->queue, posKern, 1,
                                    0, global, local,
                                    0, NULL, &ev);
    if (err != CL_SUCCESS)
        return err;
    clWaitForEvents(1, &ev);
    clFinish(ci->queue);
    return CL_SUCCESS;
}


static cl_int nbBoundingBox(NBodyState* st, cl_bool updateState)
{
    cl_int err;
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event boundingEv;
    cl_kernel boundingBox;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;


    
    boundingBox = kernels->boundingBox;
    global[0] = st->effNBody;
    local[0] = ws->local[0];
    int iterations = ceil(log(st->effNBody)/log(local[0]));
    // printf("ITERATIONS: %d\n", iterations);
    
    cl_event ev;
    for(int i = 0; i < iterations; ++i){
        err = clEnqueueNDRangeKernel(ci->queue, boundingBox, 1,
                                    0, global, local,
                                    0, NULL, &ev);
        if (err != CL_SUCCESS)
        return err;
        err = clEnqueueBarrier(ci->queue);
        if (err != CL_SUCCESS)
            return err;
    }

    clFinish(ci->queue);    
    return CL_SUCCESS;
}


static cl_int nbBitonicMortonSort(NBodyState* st, cl_bool updateState)
{
    cl_int err;
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event mortonEv;
    cl_kernel bitonicMortonSort;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;


    
    bitonicMortonSort = kernels->bitonicMortonSort;
    global[0] = st->effNBody/2;
    local[0] = ws->local[0]/2;
    // local[0] = (global[0] % local[0] == 0)?ws->local[0]:ws->local[0]/2;
    // printf("EFFNBODY: %d\n", st->effNBody);
    // printf("GLOBAL WORKGROUP SIZE: %d\n", global[0]);
    // printf("LOCAL WORKGROUP SIZE: %d\n", local[0]);
    
    cl_event ev;

    if (err != CL_SUCCESS)
        return err;

    for(int length = 1; length < st->effNBody; length <<= 1){
        for(int increment = length; increment > 0; increment >>= 1){
            err = clSetKernelArg(kernels->bitonicMortonSort, 18, sizeof(int), &increment );
            err = clSetKernelArg(kernels->bitonicMortonSort, 19, sizeof(int), &length );

            err = clEnqueueNDRangeKernel(ci->queue, bitonicMortonSort, 1,
                                        0, global, local,
                                        0, NULL, &ev);
            if (err != CL_SUCCESS)
                return err;
            err = clEnqueueBarrier(ci->queue);
            if (err != CL_SUCCESS)
                return err;
        }
    }
    clFinish(ci->queue);    
    return CL_SUCCESS;
}

static cl_int nbEncodeTree(NBodyState* st, cl_bool updateState)
{
    cl_int err;
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event encodeEv;
    cl_kernel encodeTree;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;


    
    encodeTree = kernels->encodeTree;
    global[0] = st->effNBody;
    local[0] = ws->local[0];
    
    // printf("BEGINNING TREE ENCODING\n");
    cl_event ev;
    err = clEnqueueNDRangeKernel(ci->queue, encodeTree, 1,
                                0, global, local,
                                0, NULL, &ev);
    if (err != CL_SUCCESS)
        return err;

    clFinish(ci->queue);
    // printf("COMPLETED TREE ENCODING\n");
    return CL_SUCCESS;
}

static cl_int nbClearBuffers(NBodyState* st, cl_bool updateState)
{
    cl_int err; 
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event encodeEv;
    cl_kernel constructTree;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;

    global[0] = st->effNBody;
    local[0] = ws->local[0];
    cl_event ev;

    //CLEAR TREE BUFFERS:
    err |= clSetKernelArg(kernels->zeroBuffers, 18, sizeof(cl_mem), &(st->nbb->bodyParents));    
    err |= clSetKernelArg(kernels->zeroBuffers, 19, sizeof(cl_mem), &(st->nbb->gpuTree));
    err |= clSetKernelArg(kernels->zeroBuffers, 20, sizeof(cl_mem), &(st->nbb->gpuLeafs));
    err |= clSetKernelArg(kernels->zeroBuffers, 21, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err |= clSetKernelArg(kernels->zeroBuffers, 22, sizeof(cl_mem), &(st->nbb->gpuOctree));
    err |= clSetKernelArg(kernels->zeroBuffers, 23, sizeof(cl_mem), &(st->nbb->swap));
    err |= clEnqueueNDRangeKernel(ci->queue, kernels->zeroBuffers, 1,
                                0, global, NULL,
                                0, NULL, &ev);
}

static cl_int nbConstructTree(NBodyState* st, cl_bool updateState)
{
    cl_int err; 
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event encodeEv;
    cl_kernel constructTree;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;

    global[0] = st->effNBody - 1;
    local[0] = ws->local[0];
    cl_event ev;

    // err |= clEnqueueBarrier(ci->queue);
    // if (err != CL_SUCCESS)
    //     return err;

    clFinish(ci->queue);
    // printf("NODE COUNTS:\n");
    

    // printf("BEGINNING TREE CONSTRUCTION\n");
    err = clSetKernelArg(kernels->constructTree, 18, sizeof(cl_mem), &(st->nbb->gpuTree));
    err = clSetKernelArg(kernels->constructTree, 19, sizeof(cl_mem), &(st->nbb->gpuLeafs));
    err = clSetKernelArg(kernels->constructTree, 20, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err = clEnqueueNDRangeKernel(ci->queue, kernels->constructTree, 1,
                                0, global, NULL,
                                0, NULL, &ev);
    if (err != CL_SUCCESS)
        return err;

    err = clEnqueueBarrier(ci->queue);
    if (err != CL_SUCCESS)
        return err;

    global[0] = st->effNBody;

    err = clSetKernelArg(kernels->countOctNodes, 18, sizeof(cl_mem), &(st->nbb->gpuTree));
    err = clSetKernelArg(kernels->countOctNodes, 19, sizeof(cl_mem), &(st->nbb->gpuLeafs));
    err = clSetKernelArg(kernels->countOctNodes, 20, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err = clEnqueueNDRangeKernel(ci->queue, kernels->countOctNodes, 1,
                                0, global, NULL,
                                0, NULL, &ev);
    if (err != CL_SUCCESS)
    return err;


    // STORE NODE COUNTS FOR INCLUSIVE PREFIX SUM:
    err |= clSetKernelArg(kernels->prefixSumInclusiveUtil, 0, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err |= clSetKernelArg(kernels->prefixSumInclusiveUtil, 1, sizeof(cl_mem), &(st->nbb->swap));
    err |= clEnqueueNDRangeKernel(ci->queue, kernels->prefixSumInclusiveUtil, 1,
        0, global, NULL,
        0, NULL, &ev);
        
    if(err != CL_SUCCESS)
        return err;


    // PREFIX SUM UPSWEEP:

    uint *value = 0;
    err |= clEnqueueWriteBuffer(st->ci->queue,
                            st->nbb->iteration,
                            CL_TRUE,
                            0, sizeof(uint), value,
                            0, NULL, NULL);
    err = clSetKernelArg(kernels->prefixSumUpsweep, 0, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err = clSetKernelArg(kernels->prefixSumUpsweep, 1, sizeof(cl_mem), &(st->nbb->iteration));
    for(int i = 0; i < log2(st->effNBody); ++i){
        global[0] = st->effNBody >> (i + 1);
        err = clEnqueueNDRangeKernel(ci->queue, kernels->prefixSumUpsweep, 1,
                                    0, global, NULL,
                                    0, NULL, &ev);
        if (err != CL_SUCCESS)
        return err;
    }


    //PREFIX SUM DOWNSWEEP:
    err = clSetKernelArg(kernels->prefixSumDownsweep, 0, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err = clSetKernelArg(kernels->prefixSumDownsweep, 1, sizeof(cl_mem), &(st->nbb->iteration));
    for(int i = 0; i < log2(st->effNBody); ++i){
        global[0] = 1 << (i);
        // printf("%d\n", global[0]);
        err = clEnqueueNDRangeKernel(ci->queue, kernels->prefixSumDownsweep, 1,
                                    0, global, NULL,
                                    0, NULL, &ev);
        if (err != CL_SUCCESS)
        return err;
    }

    
    // ADD STORED NODE COUNTS FOR INCLUSIVE PREFIX SUM:
    global[0] = st->effNBody;
    err |= clSetKernelArg(kernels->prefixSumInclusiveUtil, 1, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err |= clSetKernelArg(kernels->prefixSumInclusiveUtil, 0, sizeof(cl_mem), &(st->nbb->swap));
    err |= clEnqueueNDRangeKernel(ci->queue, kernels->prefixSumInclusiveUtil, 1,
        0, global, NULL,
        0, NULL, &ev);
        
    if(err != CL_SUCCESS)
        return err;



 
    uint32_t* nC = calloc(st->effNBody, sizeof(uint32_t));
    err |= clEnqueueReadBuffer(st->ci->queue,
                            st->nbb->nodeCounts,
                            CL_TRUE,
                            0, st->effNBody*sizeof(uint32_t), nC,
                            0, NULL, NULL);
    err |= clEnqueueBarrier(ci->queue);
    if (err != CL_SUCCESS)
        return err;
    nC[st->effNBody - 1] += 1; //Add one to account for root node
    
    
    // printf("%d<<<<<<<<<<<<<<<<<<\n", nC[st->effNBody - 1]);
    
    //THIS IS THE PROBLEM HERE:
    global[0] = st->effNBody - 1;
    err = clSetKernelArg(kernels->constructOctTree, 18, sizeof(cl_mem), &(st->nbb->gpuTree));
    err = clSetKernelArg(kernels->constructOctTree, 19, sizeof(cl_mem), &(st->nbb->gpuLeafs));
    err = clSetKernelArg(kernels->constructOctTree, 20, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err = clSetKernelArg(kernels->constructOctTree, 21, sizeof(cl_mem), &(st->nbb->gpuOctree));
    err = clEnqueueNDRangeKernel(ci->queue, kernels->constructOctTree, 1,
                                0, global, NULL,
                                0, NULL, &ev);

    err = clEnqueueBarrier(ci->queue);
    if (err != CL_SUCCESS)
        return err;

        
    // NODE COUNTS:
        
    global[0] = nC[st->effNBody - 1];
    err = clSetKernelArg(kernels->linkOctree, 18, sizeof(cl_mem), &(st->nbb->bodyParents));    
    err = clSetKernelArg(kernels->linkOctree, 19, sizeof(cl_mem), &(st->nbb->gpuTree));
    err = clSetKernelArg(kernels->linkOctree, 20, sizeof(cl_mem), &(st->nbb->gpuLeafs));
    err = clSetKernelArg(kernels->linkOctree, 21, sizeof(cl_mem), &(st->nbb->nodeCounts));
    err = clSetKernelArg(kernels->linkOctree, 22, sizeof(cl_mem), &(st->nbb->gpuOctree));
    err = clEnqueueNDRangeKernel(ci->queue, kernels->linkOctree, 1,
                                0, global, NULL,
                                0, NULL, &ev);


    err = clSetKernelArg(kernels->threadOctree, 0, sizeof(cl_mem), &(st->nbb->gpuOctree));
    err = clEnqueueNDRangeKernel(ci->queue, kernels->threadOctree, 1,
                                0, global, NULL,
                                0, NULL, &ev);



    err |= clEnqueueBarrier(ci->queue);
    if (err != CL_SUCCESS)
        return err;

    clFinish(ci->queue);
    // printf("NODE COUNTS:\n");
    // for(int i = 0; i < st->effNBody; ++i){
    //     printf("%d\n", nC[i]);
    // }
    // printf("=========================\n");

    // printf("COMPLETED TREE CONSTRUCTION\n");
    free(nC);
    return CL_SUCCESS;
}



static cl_int nbForceCalculationTreecode(NBodyState* st, cl_bool updateState)
{
    cl_int err; 
    size_t chunk;
    size_t nChunk;
    cl_int upperBound;
    size_t global[1];
    size_t local[1];
    size_t offset[1];
    cl_event encodeEv;
    cl_kernel constructTree;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;
    NBodyWorkSizes* ws = st->workSizes;
    cl_int effNBody = st->effNBody;

    global[0] = st->effNBody;
    local[0] = ws->local[0];
    cl_event ev;

    err |= nbSetMemArrayArgs(kernels->forceCalculationTreecode, st->nbb->pos, 0);
    err |= nbSetMemArrayArgs(kernels->forceCalculationTreecode, st->nbb->vel, 3);
    err |= nbSetMemArrayArgs(kernels->forceCalculationTreecode, st->nbb->acc, 6);
    err = clSetKernelArg(kernels->forceCalculationTreecode, 9, sizeof(cl_mem), &(st->nbb->bodyParents));        
    err |= clSetKernelArg(kernels->forceCalculationTreecode, 10, sizeof(cl_mem), &(st->nbb->mass));
    err |= clSetKernelArg(kernels->forceCalculationTreecode, 11, sizeof(cl_mem), &(st->nbb->gpuOctree));
    err |= clEnqueueNDRangeKernel(ci->queue, kernels->forceCalculationTreecode, 1,
                                0, global, NULL,
                                0, NULL, &ev);
}





//NOTE: NOT NEEDED
// static NBodyStatus nbCheckpointCL(const NBodyCtx* ctx, NBodyState* st)
// {
//     if (st->useCLCheckpointing && nbTimeToCheckpoint(ctx, st))
//     {
//         cl_int err;
// 
//         err = nbMarshalBodies(st, CL_FALSE);
//         if (err != CL_SUCCESS)
//         {
//             return NBODY_CL_ERROR;
//         }
// 
//         if (nbWriteCheckpoint(ctx, st))
//         {
//             return NBODY_CHECKPOINT_ERROR;
//         }
// 
//         mw_checkpoint_completed();
//     }
// 
//     return NBODY_SUCCESS;
// }

/* We need to run a fake step to get the initial accelerations without
 * touching the positons/velocities */
// static cl_int nbRunPreStep(NBodyState* st)
// {
//     static const cl_int trueVal = TRUE;    /* Need an lvalue */
//     static const cl_int falseVal = FALSE;
//     cl_kernel kernel = st->usesExact ? st->kernels->forceCalculation_Exact : st->kernels->forceCalculation;
//     cl_int err;
// 
//     /* Only calculate accelerations*/
//     err = clSetKernelArg(kernel, 29, sizeof(cl_int), &falseVal);
//     if (err != CL_SUCCESS)
//         return err;
// 
//     if (!st->usesExact)
//     {
//         err = nbExecuteTreeConstruction(st);
//         if (err != CL_SUCCESS)
//             return err;
//     }
// 
//     err = nbExecuteForceKernels(st, CL_FALSE);
//     if (err != CL_SUCCESS)
//         return err;
// 
//     /* All later steps will be real timesteps */
//     return clSetKernelArg(kernel, 29, sizeof(cl_int), &trueVal);
// }


//TODO: REWRITE:
static NBodyStatus nbMainLoopCL(const NBodyCtx* ctx, NBodyState* st)
{
//     printf("HERE \n");
//     NBodyStatus rc = NBODY_SUCCESS;
//     cl_int err;
// 
//     err = nbRunPreStep(st);
//     if (err != CL_SUCCESS)
//     {
//         mwPerrorCL(err, "Error running pre step");
//         return NBODY_CL_ERROR;
//     }
//     #ifdef NBODY_BLENDER_OUTPUT
//         deleteOldFiles(st);
//         mwvector startCmPos;
//         mwvector perpendicularCmPos;
//         mwvector nextCmPos;
//         nbFindCenterOfMass(&startCmPos, st);
//         perpendicularCmPos=startCmPos;
//         printf("*Total frames: %d\n", kept_frames);
//     #endif
//         
//     //Main loop:
//     while (st->step < ctx->nStep)
//     {
//         #ifdef NBODY_BLENDER_OUTPUT
//             nbFindCenterOfMass(&nextCmPos, st);
//             blenderPossiblyChangePerpendicularCmPos(&nextCmPos,&perpendicularCmPos,&startCmPos);
//         #endif
//         rc = nbCheckKernelErrorCode(ctx, st);
//         if (nbStatusIsFatal(rc))
//         {
//             return rc;
//         }
// 
//         rc = nbStepSystemCL(ctx, st);
//         if (nbStatusIsFatal(rc))
//         {
//             return rc;
//         }
// 
//         rc = nbCheckpointCL(ctx, st);
//         if (nbStatusIsFatal(rc))
//         {
//             return rc;
//         }
// 
//         st->step++;
//         #ifdef NBODY_BLENDER_OUTPUT
//             if (frame_progress < st->step)
//             {
//                 frame_progress+=blender_frame_skip;
//                 blenderPrintBodies(st, ctx);
//                 printf("Frame: %d\n", (int)(st->step/blender_frame_skip));
//             }
//         #endif
//     }
//     #ifdef NBODY_BLENDER_OUTPUT
//         mwvector finalcmPos;
//         blenderPrintMisc(st, ctx, startCmPos, perpendicularCmPos);
//     #endif
// 
//     return rc;
}

/* This is dumb and errors if mem isn't set */
static cl_int clReleaseMemObject_quiet(cl_mem mem)
{
    return mem ? clReleaseMemObject(mem) : CL_SUCCESS;
}

static cl_int _nbReleaseBuffers(NBodyBuffers* nbb)
{
    cl_uint i;
    cl_int err = CL_SUCCESS;
    int j;
    //const int nDummy = sizeof(nbb->dummy) / sizeof(nbb->dummy[0]);

    if (!nbb)
    {
        return CL_SUCCESS;
    }

    for(int i = 0; i < 3; ++i){
      err |= clReleaseMemObject_quiet(nbb->pos[i]);
      err |= clReleaseMemObject_quiet(nbb->vel[i]);
      err |= clReleaseMemObject_quiet(nbb->acc[i]);
      err |= clReleaseMemObject_quiet(nbb->max[i]);
      err |= clReleaseMemObject_quiet(nbb->min[i]);

    }
    err |= clReleaseMemObject_quiet(nbb->mass);
    err |= clReleaseMemObject_quiet(nbb->mCodes);
    err |= clReleaseMemObject_quiet(nbb->iteration);
    err |= clReleaseMemObject_quiet(nbb->gpuTree);
    err |= clReleaseMemObject_quiet(nbb->gpuLeafs);
    err |= clReleaseMemObject_quiet(nbb->gpuOctree);
    err |= clReleaseMemObject_quiet(nbb->nodeCounts);
    err |= clReleaseMemObject_quiet(nbb->swap);

    if (err != CL_SUCCESS)
    {
        mwPerrorCL(err, "Error releasing buffers");
    }

    return err;
}

cl_int nbReleaseBuffers(NBodyState* st)
{
    return _nbReleaseBuffers(st->nbb);
}

//NOTE: NOT NEEDED
// cl_int nbSetInitialTreeStatus(NBodyState* st)
// {
//     cl_int err;
//     TreeStatus iniTreeStatus;
//     CLInfo* ci = st->ci;
//     NBodyBuffers* nbb = st->nbb;
// 
//     memset(&iniTreeStatus, 0, sizeof(iniTreeStatus));
// 
//     iniTreeStatus.radius = 0.0;
//     iniTreeStatus.bottom = 0;
//     iniTreeStatus.maxDepth = 1;
//     iniTreeStatus.assertionLine = -1;
//     iniTreeStatus.errorCode = 0;
//     iniTreeStatus.blkCnt = 0;
// 
//     err = clEnqueueWriteBuffer(ci->queue,
//                                nbb->treeStatus,
//                                CL_TRUE,
//                                0, sizeof(TreeStatus), &iniTreeStatus,
//                                0, NULL, NULL);
//     if (err != CL_SUCCESS)
//     {
//         mwPerrorCL(err, "Error writing initial tree status");
//     }
// 
//     return err;
// }

//NOTE: NOT NEEDED
static cl_uint nbFindInc(cl_int warpSize, cl_uint nbody)
{
    return (nbody + warpSize - 1) & (-warpSize);
}

/* In some cases to avoid conditionally barriering we want to round up to nearest workgroup size.
   Find the least common multiple necessary for kernels that need to avoid the issue
 */
cl_int nbFindEffectiveNBody(const NBodyWorkSizes* workSizes, cl_bool exact, cl_int nbody)
{
    if (exact)
    {
        /* Exact force kernel needs this */
        return mwNextMultiple((cl_int) workSizes->local[0], nbody);
        // return workSizes->local[0];
    }
    else
    {
        /* Maybe tree construction will need this later */
        // return mwNextMultiple((cl_int) workSizes->local[7], nbody);
        int i = 1;
        while(i < nbody){
            i<<=1;  
        }
        return i;
        // return mwNextMultiple((cl_int) (workSizes->local[0]), nbody);
    }
}


//NOTE: Works So Far
cl_int nbCreateBuffers(const NBodyCtx* ctx, NBodyState* st)
{
    printf("CREATING BUFFERS\n");
    cl_uint i;
    CLInfo* ci = st->ci;
    NBodyBuffers* nbb = st->nbb;
    size_t massSize;
    cl_int err;
    cl_uint nNode = nbFindNNode(&ci->di, st->effNBody);
    int n = st->effNBody;
    printf("Buffer Size: %d\n", n);
    for(int i = 0; i < 3; ++i){
      st->nbb->pos[i] = mwCreateZeroReadWriteBuffer(ci, n * sizeof(real));
      st->nbb->vel[i] = mwCreateZeroReadWriteBuffer(ci, n * sizeof(real));
      st->nbb->acc[i] = mwCreateZeroReadWriteBuffer(ci, n * sizeof(real));
    }
    st->nbb->mass = mwCreateZeroReadWriteBuffer(ci, n * sizeof(real));
    if(!st->usesExact){
        st->nbb->bodyParents = mwCreateZeroReadWriteBuffer(ci, n * sizeof(uint32_t));
        st->nbb->mCodes = mwCreateZeroReadWriteBuffer(ci, n * sizeof(uint32_t));
        st->nbb->iteration = mwCreateZeroReadWriteBuffer(ci, sizeof(uint32_t));
        st->nbb->gpuTree = mwCreateZeroReadWriteBuffer(ci, n * sizeof(gpuNode));
        st->nbb->zeroTree = mwCreateZeroReadWriteBuffer(ci, n  * sizeof(gpuNode));
        for(int i = 0; i < 3; ++i){
            st->nbb->max[i] = mwCreateZeroReadWriteBuffer(ci, n * sizeof(real));
            st->nbb->min[i] = mwCreateZeroReadWriteBuffer(ci, n * sizeof(real));
        }
        st->nbb->gpuTree = mwCreateZeroReadWriteBuffer(ci, n * sizeof(gpuNode));
        st->nbb->gpuLeafs = mwCreateZeroReadWriteBuffer(ci, n* sizeof(gpuNode));
        st->nbb->gpuOctree = mwCreateZeroReadWriteBuffer(ci, n * sizeof(gpuNode));
        st->nbb->nodeCounts = mwCreateZeroReadWriteBuffer(ci, n * sizeof(uint32_t));
        st->nbb->swap = mwCreateZeroReadWriteBuffer(ci, n* sizeof(uint32_t));
    }

    return CL_SUCCESS;
}

//NOTE: NOT NEEDED
static cl_int nbMapBodies(real* pos[3], real* vel[3], real** mass, NBodyBuffers* nbb, CLInfo* ci, cl_map_flags flags, NBodyState* st)
{
    cl_uint i;

    for (i = 0; i < 3; ++i)
    {
        // pos[i] = (real*) mapBuffer(ci, nbb->pos[i], flags, st->nbody * sizeof(real));
        // vel[i] = (real*) mapBuffer(ci, nbb->vel[i], flags, st->nbody * sizeof(real));
        // if (!pos[i] || !vel[i])
        // {
        //     return MW_CL_ERROR;
        // }
    }

    //*mass = (real*) mapBuffer(ci, nbb->mass, flags, st->nbody * sizeof(real));

    return CL_SUCCESS;
}

//NOTE: NOT NEEDED
static cl_int nbUnmapBodies(real* pos[3], real* vel[3], real* mass, NBodyBuffers* nbb, CLInfo* ci)
{
    cl_uint i;
    cl_int err = CL_SUCCESS;

    for (i = 0; i < 3; ++i)
    {
        if (pos[i])
        {
            //err |= clEnqueueUnmapMemObject(ci->queue, nbb->pos[i], pos[i], 0, NULL, NULL);
        }

        if (vel[i])
        {
            // err |= clEnqueueUnmapMemObject(ci->queue, nbb->vel[i], vel[i], 0, NULL, NULL);
        }
    }

    if (mass)
    {
        // err |= clEnqueueUnmapMemObject(ci->queue, nbb->mass, mass, 0, NULL, NULL);
    }

    return err;
}

//NOTE: NOT NEEDED
/* If last parameter is true, copy to the buffers. if false, copy from the buffers */
cl_int nbMarshalBodies(NBodyState* st, cl_bool marshalIn)
{
    cl_int i;
    cl_int err = CL_SUCCESS;
    const Body* b;
    real* pos[3] = { NULL, NULL, NULL };
    real* vel[3] = { NULL, NULL, NULL };
    real* mass = NULL;
    CLInfo* ci = st->ci;
    NBodyBuffers* nbb = st->nbb;
    cl_map_flags flags = marshalIn ? CL_MAP_WRITE : CL_MAP_READ;

    if (!marshalIn && !st->dirty)     /* Skip copying if already up to date */
    {
        return CL_SUCCESS;
    }

    err = nbMapBodies(pos, vel, &mass, nbb, ci, flags, st);
    if (err != CL_SUCCESS)
    {
        nbUnmapBodies(pos, vel, mass, nbb, ci);
        return err;
    }

    if (marshalIn)
    {
        for (i = 0, b = st->bodytab; b < st->bodytab + st->nbody; ++i, ++b)
        {
            pos[0][i] = X(Pos(b));
            pos[1][i] = Y(Pos(b));
            pos[2][i] = Z(Pos(b));

            vel[0][i] = X(Vel(b));
            vel[1][i] = Y(Vel(b));
            vel[2][i] = Z(Vel(b));

            mass[i] = Mass(b);
        }
    }
    else
    {
        for (i = 0, b = st->bodytab; b < st->bodytab + st->nbody; ++i, ++b)
        {
            X(Pos(b)) = pos[0][i];
            Y(Pos(b)) = pos[1][i];
            Z(Pos(b)) = pos[2][i];

            X(Vel(b)) = vel[0][i];
            Y(Vel(b)) = vel[1][i];
            Z(Vel(b)) = vel[2][i];

            Mass(b) = mass[i];
        }
    }

    st->dirty = FALSE; /* Host and GPU state match */

    return nbUnmapBodies(pos, vel, mass, nbb, ci);
}

/* FIXME: This will be completely wrong with checkpointing */
void nbPrintKernelTimings(const NBodyState* st)
{
    double totalTime = 0.0;
    cl_uint i;
    double nStep = (double) st->step;
    const double* kernelTimings = st->workSizes->kernelTimings;

    for (i = 0; i < 7; ++i)
    {
        totalTime += kernelTimings[i];
    }

    mw_printf("\n--------------------------------------------------------------------------------\n"
              "Total timing over %d steps:\n"
              "                         Average             Total            Fraction\n"
              "                    ----------------   ----------------   ----------------\n"
              "  boundingBox:      %16f   %16f   %15.4f%%\n"
              "  buildTree:        %16f   %16f   %15.4f%%\n"
              "  summarization:    %16f   %16f   %15.4f%%\n"
              "  sort:             %16f   %16f   %15.4f%%\n"
              "  quad moments:     %16f   %16f   %15.4f%%\n"
              "  forceCalculation: %16f   %16f   %15.4f%%\n"
              "  integration:      %16f   %16f   %15.4f%%\n"
              "  ==============================================================================\n"
              "  total             %16f   %16f   %15.4f%%\n"
              "\n--------------------------------------------------------------------------------\n"
              "\n",
              st->step,
              kernelTimings[0] / nStep, kernelTimings[0], 100.0 * kernelTimings[0] / totalTime,
              kernelTimings[1] / nStep, kernelTimings[1], 100.0 * kernelTimings[1] / totalTime,
              kernelTimings[2] / nStep, kernelTimings[2], 100.0 * kernelTimings[2] / totalTime,
              kernelTimings[3] / nStep, kernelTimings[3], 100.0 * kernelTimings[3] / totalTime,
              kernelTimings[4] / nStep, kernelTimings[4], 100.0 * kernelTimings[4] / totalTime,
              kernelTimings[5] / nStep, kernelTimings[5], 100.0 * kernelTimings[5] / totalTime,
              kernelTimings[6] / nStep, kernelTimings[6], 100.0 * kernelTimings[6] / totalTime,
              totalTime / nStep,        totalTime,        100.0 * totalTime / totalTime
        );
}

//NOTE: UPDATE KERNES
void nbPrintKernelLimits(NBodyState* st)
{
    WGInfo wgi;
    CLInfo* ci = st->ci;
    NBodyKernels* kernels = st->kernels;

//     mw_printf("Bounding box:\n");
//     mwGetWorkGroupInfo(kernels->boundingBox, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Tree Build Clear:\n");
//     mwGetWorkGroupInfo(kernels->buildTreeClear, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Tree Build:\n");
//     mwGetWorkGroupInfo(kernels->buildTree, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Summarization Clear:\n");
//     mwGetWorkGroupInfo(kernels->summarizationClear, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Summarization:\n");
//     mwGetWorkGroupInfo(kernels->summarization, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Sort:\n");
//     mwGetWorkGroupInfo(kernels->sort, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Quad moments:\n");
//     mwGetWorkGroupInfo(kernels->quadMoments, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Force calculation:\n");
//     mwGetWorkGroupInfo(kernels->forceCalculation, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Integration:\n");
//     mwGetWorkGroupInfo(kernels->integration, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
// 
//     mw_printf("Force calculation (Exact):\n");
//     mwGetWorkGroupInfo(kernels->forceCalculation_Exact, ci, &wgi);
//     mwPrintWorkGroupInfo(&wgi);
}

/* Debug function */
static cl_int nbPrintQuadMomentDifferences(const NBodyCtx* ctx, NBodyState* st)
{
    cl_int err;
    NBodyQuadMatrix quad;
    NBodyQuadMatrix quadRef;
    NBodyQuadMatrix quadDiff;
    NBodyQuadMatrix relDiff;

  #if DOUBLEPREC
    const real threshold = 1.0e-9;
  #else
    const real threshold = 1.0e-6;
  #endif

    // err = nbEnqueueReadRootQuadMoment(st, &quad);
    err |= nbMarshalBodies(st, CL_FALSE);
    if (err != CL_SUCCESS)
        return err;

    nbMakeTree(ctx, st);
    quadRef = Quad(st->tree.root);


    quadDiff.xx = quad.xx - quadRef.xx;
    quadDiff.xy = quad.xy - quadRef.xy;
    quadDiff.xz = quad.xz - quadRef.xz;

    quadDiff.yy = quad.yy - quadRef.yy;
    quadDiff.yz = quad.yz - quadRef.yz;

    quadDiff.zz = quad.zz - quadRef.zz;

    mw_printf("\n\nQuad: {\n"
              "  %21.15f, %21.15f, %21.15f\n"
              "  %21.15f, %21.15f\n"
              "  %21.15f\n"
              "}\n",
              quad.xx, quad.xy, quad.xz,
              quad.yy, quad.yz,
              quad.zz
        );

    mw_printf("QuadRef: {\n"
              "  %21.15f, %21.15f, %21.15f\n"
              "  %21.15f, %21.15f\n"
              "  %21.15f\n"
              "}\n",
              quadRef.xx, quadRef.xy, quadRef.xz,
              quadRef.yy, quadRef.yz,
              quadRef.zz
        );

    mw_printf("Diff: {\n"
              "  %21.15f, %21.15f, %21.15f\n"
              "  %21.15f, %21.15f\n"
              "  %21.15f\n"
              "}\n",
              quadDiff.xx, quadDiff.xy, quadDiff.xz,
              quadDiff.yy, quadDiff.yz,
              quadDiff.zz
        );

    relDiff.xx = 100.0 * quadDiff.xx / quad.xx;
    relDiff.xy = 100.0 * quadDiff.xy / quad.xy;
    relDiff.xz = 100.0 * quadDiff.xz / quad.xz;

    relDiff.yy = 100.0 * quadDiff.yy / quad.yy;
    relDiff.yz = 100.0 * quadDiff.yz / quad.yz;

    relDiff.zz = 100.0 * quadDiff.zz / quad.zz;

    mw_printf("RelativeDiff: {\n"
              "  %21.15f, %21.15f, %21.15f\n"
              "  %21.15f, %21.15f\n"
              "  %21.15f\n"
              "}\n"
              "\n",
              relDiff.xx, relDiff.xy, relDiff.xz,
              relDiff.yy, relDiff.yz,
              relDiff.zz
        );

    if (   mw_fabs(quadDiff.xx) >= threshold || mw_fabs(quadDiff.xy) >= threshold || mw_fabs(quadDiff.xz) >= threshold
        || mw_fabs(quadDiff.yy) >= threshold || mw_fabs(quadDiff.yz) >= threshold
        || mw_fabs(quadDiff.zz) >= threshold)
    {
        mw_printf("WARNING: Quad moment summarization results greatly differs from reference\n");
    }

    return CL_SUCCESS;
}


/*
  Run the tree building stages and then compare the resulting center
  of masses and quadrupole moments calculated from the normal CPU
  method and the summarization/quad moment kernels
 */
static cl_int nbPrintSummarizationDifferences(NBodyState* st)
{
    cl_int err;
    mwvector cm, cmRef;
    mwvector dcm;
    mwvector dcmRel;

  #if DOUBLEPREC
    const real threshold = 1.0e-9;
  #else
    const real threshold = 1.0e-6f;
  #endif

    err = nbEnqueueReadCenterOfMass(st, &cm);
    if (err != CL_SUCCESS)
        return err;

    err = nbMarshalBodies(st, CL_FALSE);
    if (err != CL_SUCCESS)
        return err;


    cmRef = nbCenterOfMass(st);

    dcm = mw_subv(cm, cmRef);

    dcmRel = mw_mulvs(dcm, 100.0);
    mw_incdivv(dcmRel, cmRef);

    dcm.w = cm.w - cmRef.w;
    dcmRel.w = 100.0 * dcm.w / cmRef.w;

    mw_printf("\nReference center of mass: %21.15f, %21.15f, %21.15f, %21.15f\n",
              cmRef.x, cmRef.y, cmRef.z, cmRef.w);

    mw_printf("Center of mass:           %21.15f, %21.15f, %21.15f, %21.15f\n",
              cm.x, cm.y, cm.z, cm.w);

    mw_printf("Difference:               %21.15f, %21.15f, %21.15f, %21.15f\n",
              dcm.x, dcm.y, dcm.z, dcm.w);

    mw_printf("Relative difference %%:    %21.15f, %21.15f, %21.15f, %21.15f\n\n",
              dcmRel.x, dcmRel.y, dcmRel.z, dcmRel.w);

    if (mw_fabs(dcm.x) >= threshold || mw_fabs(dcm.y) >= threshold || mw_fabs(dcm.z) >= threshold || mw_fabs(dcm.w) >= threshold)
    {
        mw_printf("WARNING: Summarization results greatly differs from reference\n");
    }

    return CL_SUCCESS;
}

static cl_int nbDebugSummarization(const NBodyCtx* ctx, NBodyState* st)
{
    cl_uint i;
    const cl_uint nSamples = 10;

    for (i = 0; i < nSamples; ++i)
    {
        cl_int err;

        // err = nbExecuteTreeConstruction(st);
        if (err != CL_SUCCESS)
            return err;

        err = nbPrintSummarizationDifferences(st);
        if (err != CL_SUCCESS)
            return err;

        if (ctx->useQuad && ctx->criterion != Exact)
        {
            err = nbPrintQuadMomentDifferences(ctx, st);
            if (err != CL_SUCCESS)
                return err;
        }
    }

    return CL_SUCCESS;
}

void initGPUDataArrays(NBodyState* st, gpuData* gData){
  int n = st->effNBody;
  for(int i = 0; i < 3; ++i){
    gData->pos[i] = calloc(n, sizeof(real));
    gData->vel[i] = calloc(n, sizeof(real));
    gData->acc[i] = calloc(n, sizeof(real));
    if(!st->usesExact){
      gData->max[i] = calloc(n, sizeof(real));
      gData->min[i] = calloc(n, sizeof(real));
    }
  }
  gData->mass = calloc(n, sizeof(real));
  gData->mCodes = calloc(n, sizeof(uint32_t));
  gData->gpuTree = calloc(n, sizeof(gpuNode));
  gData->gpuOctree = calloc(n, sizeof(gpuNode));
  gData->nodeCounts = calloc(n, sizeof(uint32_t));
}

void fillGPUDataOnlyBodies(NBodyState* st, gpuData* gData){
  int n = st->effNBody;
  for(int i = 0; i < n; ++i){
    if(i < st->nbody){
      gData->pos[0][i] = st->bodytab[i].bodynode.pos.x;
      gData->pos[1][i] = st->bodytab[i].bodynode.pos.y;
      gData->pos[2][i] = st->bodytab[i].bodynode.pos.z;
      gData->vel[0][i] = st->bodytab[i].vel.x;
      gData->vel[1][i] = st->bodytab[i].vel.y;
      gData->vel[2][i] = st->bodytab[i].vel.z;
      gData->acc[0][i] = 0;
      gData->acc[1][i] = 0;
      gData->acc[2][i] = 0;
      gData->mass[i] = st->bodytab[i].bodynode.mass;
      if(!st->usesExact){
        gData->max[0][i] = st->bodytab[i].bodynode.pos.x;
        gData->max[1][i] = st->bodytab[i].bodynode.pos.y;
        gData->max[2][i] = st->bodytab[i].bodynode.pos.z;
        gData->min[0][i] = st->bodytab[i].bodynode.pos.x;
        gData->min[1][i] = st->bodytab[i].bodynode.pos.y;
        gData->min[2][i] = st->bodytab[i].bodynode.pos.z;
      }
    }
    else{
      gData->pos[0][i] = 0;
      gData->pos[1][i] = 0;
      gData->pos[2][i] = 0;
      gData->vel[0][i] = 0;
      gData->vel[1][i] = 0;
      gData->vel[2][i] = 0; 
      gData->acc[0][i] = 0;
      gData->acc[1][i] = 0;
      gData->acc[2][i] = 0;
      gData->mass[i] = 0;
      if(!st->usesExact){
        gData->max[0][i] = -DBL_MAX;
        gData->max[1][i] = -DBL_MAX;
        gData->max[2][i] = -DBL_MAX;
        gData->min[0][i] = DBL_MAX;
        gData->min[1][i] = DBL_MAX;
        gData->min[2][i] = DBL_MAX;
      }
    }
  }
}

void writeGPUBuffers(NBodyState* st, gpuData* gData){
//   printf("WRITING TO GPU BUFFERS: %f\n", gData->max[0][0]);
  CLInfo* ci = st->ci;   
  cl_int err = CL_SUCCESS;
  cl_uint i;
  cl_command_queue q = st->ci->queue;
  int n = st->effNBody;
  for(int i = 0; i < 3; ++i){
    err |= clEnqueueWriteBuffer(st->ci->queue,
                          st->nbb->pos[i],
                          CL_TRUE,
                          0, n*sizeof(real), gData->pos[i],
                          0, NULL, NULL);
    err |= clEnqueueWriteBuffer(st->ci->queue,
                        st->nbb->vel[i],
                        CL_TRUE,
                        0, n*sizeof(real), gData->vel[i],
                        0, NULL, NULL);
    err |= clEnqueueWriteBuffer(st->ci->queue,
                        st->nbb->acc[i],
                        CL_TRUE,
                        0, n*sizeof(real), gData->acc[i],
                        0, NULL, NULL);
    if(!st->usesExact){
        err |= clEnqueueWriteBuffer(st->ci->queue,
                            st->nbb->max[i],
                            CL_TRUE,
                            0, n * sizeof(real), (gData->max[i]),
                            0, NULL, NULL);
        err |= clEnqueueWriteBuffer(st->ci->queue,
                            st->nbb->min[i],
                            CL_TRUE,
                            0, n * sizeof(real), (gData->min[i]),
                            0, NULL, NULL);
    }
  }
  err |= clEnqueueWriteBuffer(st->ci->queue,
                        st->nbb->mass,
                        CL_TRUE,
                        0, n*sizeof(real), gData->mass,
                        0, NULL, NULL);
  if(err != CL_SUCCESS)
        printf("%i, ERROR WRITING TO OPENCL BUFFERS \n", err);

  clFlush(st->ci->queue);
}

void readGPUBuffers(NBodyState* st, gpuData* gData){
//   printf("READING FROM GPU BUFFERS\n");
  CLInfo* ci = st->ci;   
  cl_int err = CL_SUCCESS;
  cl_uint i;
  cl_command_queue q = st->ci->queue;
  int n = st->effNBody;
  for(int i = 0; i < 3; ++i){
    err |= clEnqueueReadBuffer(st->ci->queue,
                          st->nbb->pos[i],
                          CL_TRUE,
                          0, n*sizeof(real), gData->pos[i],
                          0, NULL, NULL);
    err |= clEnqueueReadBuffer(st->ci->queue,
                          st->nbb->vel[i],
                          CL_TRUE,
                          0, n*sizeof(real), gData->vel[i],
                          0, NULL, NULL);
    err |= clEnqueueReadBuffer(st->ci->queue,
                          st->nbb->acc[i],
                          CL_TRUE,
                          0, n*sizeof(real), gData->acc[i],
                          0, NULL, NULL);
    if(!st->usesExact){
        err |= clEnqueueReadBuffer(st->ci->queue,
                            st->nbb->max[i],
                            CL_TRUE,
                            0, n * sizeof(real), (gData->max[i]),
                            0, NULL, NULL);
        err |= clEnqueueReadBuffer(st->ci->queue,
                            st->nbb->min[i],
                            CL_TRUE,
                            0, n * sizeof(real), (gData->min[i]),
                            0, NULL, NULL);
    }
  }
  err |= clEnqueueReadBuffer(st->ci->queue,
                        st->nbb->mass,
                        CL_TRUE,
                        0, n*sizeof(real), gData->mass,
                        0, NULL, NULL);

  if(!st->usesExact){
    err |= clEnqueueReadBuffer(st->ci->queue,
                            st->nbb->mCodes,
                            CL_TRUE,
                            0, n*sizeof(uint32_t), gData->mCodes,
                            0, NULL, NULL);

    err |= clEnqueueReadBuffer(st->ci->queue,
                            st->nbb->gpuTree,
                            CL_TRUE,
                            0, n*sizeof(gpuNode), gData->gpuTree,
                            0, NULL, NULL);
    
    err |= clEnqueueReadBuffer(st->ci->queue,
                            st->nbb->nodeCounts,
                            CL_TRUE,
                            0, n * sizeof(uint), gData->nodeCounts,
                            0, NULL, NULL);

    err |= clEnqueueReadBuffer(st->ci->queue,
                            st->nbb->gpuOctree,
                            CL_TRUE,
                            0, n * sizeof(gpuNode), gData->gpuOctree,
                            0, NULL, NULL);


  }

  if(err != CL_SUCCESS)
        printf("%i, ERROR READING FROM OPENCL BUFFERS \n", err);

  clFlush(st->ci->queue);
}

NBodyStatus nbRunSystemCLExact(const NBodyCtx* ctx, NBodyState* st){
  printf("RUNNING OPENCL EXACT NBODY APPLICATION\n");
  printf("THE GPU TREE SIZE IS: %d\n", st->effNBody);
  CLInfo* ci = st->ci;   
  cl_int err;
  cl_uint i;
  cl_command_queue q = st->ci->queue;
  gpuData gData;
  initGPUDataArrays(st, &gData);
  fillGPUDataOnlyBodies(st, &gData);
  int n = st->effNBody;
  writeGPUBuffers(st, &gData);

  printf("BEGINNING EXECUTION OF GPU NBODY\n");
  err = nbExecuteForceKernels(st, CL_TRUE);
  if (err != CL_SUCCESS)
  {
      mwPerrorCL(err, "Error executing force kernels");
      return NBODY_CL_ERROR;
  }
//   readGPUBuffers(st, &gData);

  while(st->step < ctx->nStep){
    err = nbAdvanceHalfVelocity(st, CL_TRUE);
    if (err != CL_SUCCESS)
    {
        mwPerrorCL(err, "Error executing half velocity kernel");
        return NBODY_CL_ERROR;
    }
    err = nbAdvancePosition(st, CL_TRUE);
    if (err != CL_SUCCESS)
    {
        mwPerrorCL(err, "Error executing advance position kernel");
        return NBODY_CL_ERROR;
    }
    err = nbAdvanceHalfVelocity(st, CL_TRUE);
    if (err != CL_SUCCESS)
    {
        mwPerrorCL(err, "Error executing half velocity kernel");
        return NBODY_CL_ERROR;
    }
    err = nbExecuteForceKernels(st, CL_TRUE);
    if (err != CL_SUCCESS)
    {
        mwPerrorCL(err, "Error executing advance position kernel");
        return NBODY_CL_ERROR;
    }
    // printf("%d<<<<<<<<<<<<<<<<\n", ctx->BestLikeStart);
    if(st->step / ctx->nStep >= ctx->BestLikeStart && ctx->useBestLike)
    {
        printf("DUMPING TO CPU FOR BEST LIKLIHOOD CALCULATION\n");
        fflush(NULL);
        // get_likelihood(ctx, st, nbf);
    }
    ++st->step;
  }
  printf("%d/%d Steps Completed\n", st->step, ctx->nStep);
  readGPUBuffers(st, &gData);
//   for(int i = 0; i < st->effNBody/2; ++i){
//     printf("%d  |  %f\n", i, gData.mass[i]);
//   }
  nbStripBodiesSoA(st, &gData);
  NBodyStatus rc = nbMakeTree(ctx, st);
    if (nbStatusIsFatal(rc))
        return rc;
}

void printBinary(uint32_t input){
    for(int i = 31; i > -1; --i){
        printf("%d", (input >> i) & 1);
        if(i % 3 == 0){
            printf("|");
        }
    }
}

void printDebugStatus(const NBodyCtx* ctx, NBodyState* st, gpuData* gData){
    printf("\n\n===================\n");
    printf("      STEP %d\n", st->step);
    printf("===================\n");
    readGPUBuffers(st, gData);
    printf("----------------------------\n");
    printf("BODIES: \n");
    printf("----------------------------\n");
    for(int i = 0; i < st->effNBody; ++i){
        if(i < st->nbody){
            printf("%f, %f, %f\n", gData->pos[0][i], gData->pos[1][i], gData->pos[2][i]);
        }
    }
    printf("----------------------------\n");
    printf("BOUNDING BOX:\n");
    printf("        X        Y       Z\n");
    printf("----------------------------\n");
    printf("MAX: %.2f | %.2f | %.2f\nMIN: %.2f | %.2f | %.2f\n", gData->max[0][0], gData->max[1][0], gData->max[2][0],
                                                gData->min[0][0], gData->min[1][0], gData->min[2][0]);
    printf("----------------------------\n");
    fflush(NULL);

    printf("----------------------------\n");
    printf("TREE CONSTRUCTION:\n");
    printf("MORTON CODES:\n");
    printf("- - - - - - - - - - - - - - \n");
    for(int i = 0; i < st->effNBody; ++i){
        printBinary(gData->mCodes[i]);
        printf("\n");
    }
    printf("----------------------------\n");


    int octCount = gData->nodeCounts[st->effNBody - 1] + 1;
    printf("REQUIRED OCTREE NODES: %d\n", octCount);
    fflush(NULL);
    printf("----------------------------\n");
    printf("GPU OCTREE:\n");
    int count[10] = {0};
        for(int i = 0; i < octCount; ++i){
            ++count[gData->gpuOctree[i].treeLevel];
            printBinary(gData->gpuOctree[i].prefix); 
            printf("\tID: %d\tL: %d\tD: %d\tP: %d\tC:", gData->gpuOctree[i].id, gData->gpuOctree[i].treeLevel, gData->gpuOctree[i].delta, gData->gpuOctree[i].parent);

            for(int j = 0; j < 8; ++j){
                if(gData->gpuOctree[i].children[j] != 0){
                    printf("(%d)\t", gData->gpuOctree[i].children[j]);
                }
                else if(gData->gpuOctree[i].leafIndex[j] != -1){
                    printf(" %d \t", gData->gpuOctree[i].leafIndex[j]);
                }
                else{
                    printf(" - \t");
                }
            }
            printf("\n");
        }

    printf("---------------------------\n");
    printf("LEVEL INFORMATION:\n");
    for(int i = 0; i < 10; ++i){
        printf("Level %d: %d\n", i, count[i]);
    }
}

void printDebugBodies(const NBodyCtx* ctx, NBodyState* st, gpuData gData){
    for(int i = 0; i < st->effNBody; ++i){
        if(i < st->nbody){
            printf("%f, %f, %f\n", gData.pos[0][i], gData.pos[1][i], gData.pos[2][i]);
        }
    }
}

//TODO: Write Barnes-Hut kernel handler
NBodyStatus nbRunSystemCLTreecode(const NBodyCtx* ctx, NBodyState* st)
{
    printf("RUNNING OPENCL TREECODE NBODY APPLICATION\n");
    printf("THE GPU TREE SIZE IS: %d\n", st->effNBody);
    CLInfo* ci = st->ci;   
    cl_int err;
    cl_uint i;
    cl_command_queue q = st->ci->queue;
    gpuData gData;
    initGPUDataArrays(st, &gData);
    fillGPUDataOnlyBodies(st, &gData);
    int n = st->effNBody;

    //BEGIN TIMING
    struct timeval start[6], end[6];
    // sleep(1);
    writeGPUBuffers(st, &gData);


    //HANDLE RUNNING BOUNDING BOX HERE:
    gettimeofday(&start[5], NULL);
    while(st->step < ctx->nStep){
        
        gettimeofday(&start[0], NULL);
        err = nbBoundingBox(st, CL_TRUE);
        if (err != CL_SUCCESS)
        {
            mwPerrorCL(err, "Error executing bounding box kernel");
            return NBODY_CL_ERROR;
        }
        gettimeofday(&end[0], NULL);

        gettimeofday(&start[1], NULL);
        //RUN TREE CONSTRUCTION KERNEL:
        err = nbEncodeTree(st, CL_TRUE);
        if(err != CL_SUCCESS){
            mwPerrorCL(err, "Error executing tree construction kernel");
            return NBODY_CL_ERROR;
        }
        gettimeofday(&end[1], NULL);

        gettimeofday(&start[2], NULL);
        err = nbBitonicMortonSort(st, CL_TRUE);
        if(err != CL_SUCCESS){
            mwPerrorCL(err, "Error executing morton sorting kernel");
            return NBODY_CL_ERROR;
        }
        gettimeofday(&end[2], NULL);

        gettimeofday(&start[3], NULL);
        err = nbConstructTree(st, CL_TRUE);
        if(err != CL_SUCCESS){
            mwPerrorCL(err, "Error executing tree construction kernel");
            return NBODY_CL_ERROR;
        }
        gettimeofday(&end[3], NULL);

        // printDebugStatus(ctx, st, &gData);
        // printf("STEP: %d\n", st->step);
        // readGPUBuffers(st, &gData);

        gettimeofday(&start[4], NULL);
        err = nbForceCalculationTreecode(st, CL_TRUE);
        if(err != CL_SUCCESS){
            mwPerrorCL(err, "Error executing force calculation kernel");
            return NBODY_CL_ERROR;
        }
        gettimeofday(&end[4], NULL);

        if(st->step < ctx->nStep - 1){
            err = nbClearBuffers(st, CL_TRUE);
            if(err != CL_SUCCESS){
                mwPerrorCL(err, "Error executing buffer clearing kernel");
                return NBODY_CL_ERROR;
            }
        }
        ++st->step;
    }
    gettimeofday(&end[5], NULL);
    readGPUBuffers(st, &gData);
    // printDebugStatus(ctx, st, &gData);

    printf("==============================\n");
    printf("BOUNDING BOX EXECUTION TIME:\n");
    // printf("%.4f ms\n", (endT - startT) * 1000);
    printf("%.4f ms\n", (((real)end[0].tv_sec + (real)end[0].tv_usec * (1.0/1000000)) - ((real)start[0].tv_sec + (real)start[0].tv_usec * (1.0/1000000))) * 1000);
    printf("==============================\n");
    fflush(NULL);
    
    printf("==============================\n");
    printf("MORTON CODE GENERATION TIME:\n");
    // printf("%.4f ms\n", (endT - startT) * 1000);
    printf("%.4f ms\n", (((real)end[1].tv_sec + (real)end[1].tv_usec * (1.0/1000000)) - ((real)start[1].tv_sec + (real)start[1].tv_usec * (1.0/1000000))) * 1000);
    printf("==============================\n");
    fflush(NULL);

    printf("==============================\n");
    printf("MORTON CODE SORT EXECUTION TIME:\n");
    // printf("%.4f ms\n", (endT - startT) * 1000);
    printf("%.4f ms\n", (((real)end[2].tv_sec + (real)end[2].tv_usec * (1.0/1000000)) - ((real)start[2].tv_sec + (real)start[2].tv_usec * (1.0/1000000))) * 1000);
    printf("==============================\n");
    fflush(NULL);

    printf("==============================\n");
    printf("TREE CONSTRUCTION EXECUTION TIME:\n");
    // printf("%.4f ms\n", (endT - startT) * 1000);
    printf("%.4f ms\n", (((real)end[3].tv_sec + (real)end[3].tv_usec * (1.0/1000000)) - ((real)start[3].tv_sec + (real)start[3].tv_usec * (1.0/1000000))) * 1000);
    printf("==============================\n");
    fflush(NULL);

    printf("==============================\n");
    printf("FORCE CALCULATION EXECUTION TIME:\n");
    // printf("%.4f ms\n", (endT - startT) * 1000);
    printf("%.4f ms\n", (((real)end[4].tv_sec + (real)end[4].tv_usec * (1.0/1000000)) - ((real)start[4].tv_sec + (real)start[4].tv_usec * (1.0/1000000))) * 1000);
    printf("==============================\n");
    fflush(NULL);

    printf("==============================\n");
    printf("TOTAL EXECUTION TIME:\n");
    // printf("%.4f ms\n", (endT - startT) * 1000);
    printf("%.4f ms\n", (((real)end[5].tv_sec + (real)end[5].tv_usec * (1.0/1000000)) - ((real)start[5].tv_sec + (real)start[5].tv_usec * (1.0/1000000))) * 1000);
    printf("==============================\n");
    fflush(NULL);

}

NBodyStatus nbStepSystemCL(const NBodyCtx* ctx, NBodyState* st)
{
    //THIS FUNCTION IS TO RUN ONLY ONE STEP OF THE OCL FUNCTION
    return NBODY_SUCCESS;
}

NBodyStatus nbStripBodiesSoA(NBodyState* st, gpuData* gData){ //Function to strip bodies out of GPU Tree
  printf("STRIPPING BODIES FROM BUFFER STRUCTURE\n");
  int n = st->effNBody;
  for(int i = 0; i < n; ++i){
    if(i < st->nbody){
        st->bodytab[i].bodynode.pos.x = gData->pos[0][i];
        st->bodytab[i].bodynode.pos.y = gData->pos[1][i];
        st->bodytab[i].bodynode.pos.z = gData->pos[2][i];

        st->bodytab[i].bodynode.bodyID = i;
        st->bodytab[i].vel.x = gData->vel[0][i];
        st->bodytab[i].vel.y = gData->vel[1][i];
        st->bodytab[i].vel.z = gData->vel[2][i];

        st->bodytab[i].bodynode.mass = gData->mass[i];
    }
  }
}

NBodyStatus nbRunSystemCL(const NBodyCtx* ctx, NBodyState* st)
{
    if(ctx->criterion != Exact){
      nbRunSystemCLTreecode(ctx, st);
    }
    else{
      nbRunSystemCLExact(ctx, st);
    }
    return nbWriteFinalCheckpoint(ctx, st);
}

