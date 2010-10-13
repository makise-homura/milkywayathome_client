/* Copyright 2010 Matthew Arsenault, Travis Desell, Boleslaw
Szymanski, Heidi Newberg, Carlos Varela, Malik Magdon-Ismail and
Rensselaer Polytechnic Institute.

This file is part of Milkway@Home.

Milkyway@Home is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Milkyway@Home is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Milkyway@Home.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "milkyway_util.h"
#include "show_cl_types.h"
#include "milkyway_cl.h"
#include "setup_cl.h"
#include "separation_cl_buffers.h"
#include "separation_cl_defs.h"

#if SEPARATION_INLINE_KERNEL
  #include "integral_kernel.h"
#endif /* SEPARATION_INLINE_KERNEL */


/* Only sets the constant arguments, not the outputs which we double buffer */
static inline cl_int separationSetKernelArgs(CLInfo* ci, SeparationCLMem* cm)
{
    cl_int err = CL_SUCCESS;

    /* The constant arguments */
    err |= clSetKernelArg(ci->kern, 2, sizeof(cl_mem), &cm->ap);
    err |= clSetKernelArg(ci->kern, 3, sizeof(cl_mem), &cm->ia);
    err |= clSetKernelArg(ci->kern, 4, sizeof(cl_mem), &cm->sc);
    err |= clSetKernelArg(ci->kern, 5, sizeof(cl_mem), &cm->rc);
    err |= clSetKernelArg(ci->kern, 6, sizeof(cl_mem), &cm->rPts);

    if (err != CL_SUCCESS)
    {
        warn("Error setting kernel arguments: %s\n", showCLInt(err));
        return err;
    }

    return CL_SUCCESS;
}

#if DOUBLEPREC
  #define DOUBLEPREC_DEF_STRING "-D DOUBLEPREC=1 "
#else
  #define DOUBLEPREC_DEF_STRING "-D DOUBLEPREC=0 -cl-single-precision-constant "
#endif /* DOUBLEPREC */

#if SEPARATION_INLINE_KERNEL

char* findKernelSrc()
{
    return integral_kernel_src;
}

void freeKernelSrc(char* src)
{
  #pragma unused(src)
}

#else

/* Reading from the file is more convenient for actually working on
 * it. Inlining is more useful for releasing when we don't want to
 * deal with the hassle of distributing more files. */
char* findKernelSrc()
{
    char* kernelSrc = NULL;
    kernelSrc = mwReadFile("../kernels/integrals.cl");
    if (!kernelSrc)
        warn("Failed to read kernel file\n");

    return kernelSrc;
}

void freeKernelSrc(char* src)
{
    free(src);
}

#endif

cl_int setupSeparationCL(const ASTRONOMY_PARAMETERS* ap,
                         const INTEGRAL_AREA* ia,
                         const STREAM_CONSTANTS* sc,
                         const STREAM_GAUSS* sg,
                         const CLRequest* clr,
                         CLInfo* ci,
                         SeparationCLMem* cm)
{
    cl_int err;
    //char* compileDefs;
    char* kernelSrc;
    char* extraDefs;
    char* cwd;

    cwd = getcwd(NULL, 0);
    asprintf(&extraDefs, DOUBLEPREC_DEF_STRING
                         //"-D__ATI_CL__ "
                        "-cl-strict-aliasing "
                        "-cl-finite-math-only "
                        "-I%s/../include "
                        "-I%s/../../include "
                        "-I%s/../../milkyway/include ",
                        cwd, cwd, cwd);

    kernelSrc = findKernelSrc();
    if (!kernelSrc)
    {
        warn("Failed to read CL kernel source\n");
        return -1;
    }

    //compileDefs = separationCLDefs(ap, extraDefs);
    err = mwSetupCL(ci, clr, "mu_sum_kernel", &kernelSrc, 1, extraDefs);

    freeKernelSrc(kernelSrc);
    free(cwd);
    free(extraDefs);
    //free(compileDefs);

    if (err != CL_SUCCESS)
    {
        fail("Failed to setup OpenCL device: %s\n", showCLInt(err));
        return err;
    }

    err = createSeparationBuffers(ap, ia, sc, sg, ci, cm);
    if (err != CL_SUCCESS)
    {
        fail("Failed to create CL buffers: %s\n", showCLInt(err));
        return err;
    }

    err = separationSetKernelArgs(ci, cm);
    if (err != CL_SUCCESS)
    {
        fail("Failed to set integral kernel arguments: %s\n", showCLInt(err));
        return err;
    }

    return CL_SUCCESS;
}

