/*
 * Copyright (c) 2010-2011 Matthew Arsenault
 * Copyright (c) 2010-2011 Rensselaer Polytechnic Institute
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

#include <stdio.h>
#include "nbody_plain.h"
#include "nbody_shmem.h"
#include "nbody_curses.h"
#include "nbody_defaults.h"
#include "nbody_util.h"
#include "nbody_checkpoint.h"
#include "nbody_grav.h"

#ifdef NBODY_BLENDER_OUTPUT
  #include "blender_visualizer.h"
#endif

static void nbReportProgress(const NBodyCtx* ctx, NBodyState* st)
{
    real frac = (real) st->step / (real) ctx->nStep;

    mw_fraction_done(frac);

    if (st->reportProgress)
    {
        mw_mvprintw(0, 0,
                    "Running: %f / %f (%f%%)\n",
                    frac * ctx->timeEvolve,
                    ctx->timeEvolve,
                    100.0 * frac
            );

        mw_refresh();
    }
}

static NBodyStatus nbCheckpoint(const NBodyCtx* ctx, NBodyState* st)
{
    if (nbTimeToCheckpoint(ctx, st))
    {
        if (nbWriteCheckpoint(ctx, st))
        {
            return NBODY_CHECKPOINT_ERROR;
        }

        mw_checkpoint_completed();
    }

    return NBODY_SUCCESS;
}

/* Advance velocity by half a timestep */
static inline void bodyAdvanceVel(Body* p, const mwvector a, const real dtHalf)
{
    mwvector dv;

    dv = mw_mulvs(a, dtHalf);   /* get velocity increment */
    mw_incaddv(Vel(p), dv);     /* advance v by 1/2 step */
}

/* Advance body position by 1 timestep */
static inline void bodyAdvancePos(Body* p, const real dt)
{
    mwvector dr;

    dr = mw_mulvs(Vel(p), dt);  /* get position increment */
    mw_incaddv(Pos(p), dr);     /* advance r by 1 step */
}

static inline void advancePosVel(NBodyState* st, const int nbody, const real dt)
{
    int i;
    real dtHalf = 0.5 * dt;
    Body* bodies = mw_assume_aligned(st->bodytab, 16);
    const mwvector* accs = mw_assume_aligned(st->acctab, 16);

    /*For writing to a file*/
        #ifndef fp 
            FILE* fr;
        #endif
        fr = fopen("ramping_pos_after_ramp","a+"); 
    /*End file creation code*/  

  #ifdef _OPENMP
    #pragma omp parallel for private(i) shared(bodies, accs) schedule(dynamic, 4096 / sizeof(accs[0]))
  #endif
    for (i = 0; i < nbody; ++i)
    {

        mwvector velocity = st->bodytab[i].vel;
        mwvector position = st->bodytab[i].bodynode.pos;
        bodyAdvanceVel(&bodies[i], accs[i], dtHalf);
        bodyAdvancePos(&bodies[i], dt);

        /*if(!st->ramping) // This is test code to check the orbit AFTER ramping
        {
            fprintf(fr, "%f %f %f %f %f %f\n"
                                     , position.x
                                     , position.y 
                                     , position.z
                                     , velocity.x
                                     , velocity.y
                                     , velocity.z);
        }*/
        if(st->step == 0)
        {
            fprintf(fr, "%f %f %f %f %f %f\n"
                                , position.x
                                , position.y
                                , position.z);
        }
    }
    fclose(fr);
}

static inline void advanceVelocities(NBodyState* st, const int nbody, const real dt)
{
    int i;
    real dtHalf = 0.5 * dt;
    Body* bodies = mw_assume_aligned(st->bodytab, 16);
    const mwvector* accs = mw_assume_aligned(st->acctab, 16);

  #ifdef _OPENMP
    #pragma omp parallel for private(i) schedule(dynamic, 4096 / sizeof(accs[0]))
  #endif
    for (i = 0; i < nbody; ++i)      /* loop over all bodies */
    {
        bodyAdvanceVel(&bodies[i], accs[i], dtHalf);
    }
}

/* stepSystem: advance N-body system one time-step. */
NBodyStatus nbStepSystemPlain(const NBodyCtx* ctx, NBodyState* st)
{


    static mwvector initial_vel = ZERO_VECTOR;

    if(mw_length(initial_vel) == 0)
    {
        initial_vel = mw_addv(initial_vel, nbCenterOfMom(st));
    }
    /*#ifndef initial_vel /*so we keep memory of the center of Momentum
        static mwvector initial_vel;
        mw_printf("%f %f %f\n", initial_vel.x, initial_vel.y, initial_vel.z);
        initial_vel = nbCenterOfMom(st);
    #endif*/

    NBodyStatus rc;
    const real dt = ctx->timestep;

    advancePosVel(st, st->nbody, dt);

    rc = nbGravMap(ctx, st);
    advanceVelocities(st, st->nbody, dt);

    st->step++;
    #ifdef NBODY_BLENDER_OUTPUT
        blenderPrintBodies(st, ctx);
        printf("Frame: %d\n", (int)(st->step));
    #endif

    /* we implement two checks here, one to check that "ramping"
    is still true and that we are within the ramping timstep period.
    If we are not, ramping will still be true, so we must change it
    to false, so the code will not execute ramping code, and reset the
    timestep to 0*/
    if ( st->ramping && (st->step < (ctx->ramp * ctx->nStep) ) )
    {
        subtractMassMomentumCenters(ctx, st);
    }

    else if (st-> ramping && st->step == 0)
    {
        st->ramping = 0;
    }

    else if(st->ramping)// && st->step >= ctx->ramp * ctx->nStep) 
    {
        resetVelocities(ctx, st, initial_vel);
        st->ramping = 0; /*FIXME: later, change this to mwbool*/
        st->step = 0;
    }
    return rc;
}

NBodyStatus nbRunSystemPlain(const NBodyCtx* ctx, NBodyState* st)
{
    NBodyStatus rc = NBODY_SUCCESS;

    rc |= nbGravMap(ctx, st); /* Calculate accelerations for 1st step this episode */
    if (nbStatusIsFatal(rc))
        return rc;

    #ifdef NBODY_BLENDER_OUTPUT
        if(mkdir("./frames", S_IRWXU | S_IRWXG) < 0)
        {
          return NBODY_ERROR;
        }
        deleteOldFiles(st);
        mwvector startCmPos;
        mwvector perpendicularCmPos;
        mwvector nextCmPos;
        nbFindCenterOfMass(&startCmPos, st);
        perpendicularCmPos=startCmPos;
    #endif

    while (st->step < ctx->nStep)
    {
        #ifdef NBODY_BLENDER_OUTPUT
            nbFindCenterOfMass(&nextCmPos, st);
            blenderPossiblyChangePerpendicularCmPos(&nextCmPos,&perpendicularCmPos,&startCmPos);
        #endif
        rc |= nbStepSystemPlain(ctx, st);
        if (nbStatusIsFatal(rc))   /* advance N-body system */
            return rc;

        rc |= nbCheckpoint(ctx, st);
        if (nbStatusIsFatal(rc))
            return rc;

        /* We report the progress at step + 1. 0 is the original
           center of mass. */
        nbReportProgress(ctx, st);
        nbUpdateDisplayedBodies(ctx, st);
    }
    
    #ifdef NBODY_BLENDER_OUTPUT
        blenderPrintMisc(st, ctx, startCmPos, perpendicularCmPos);
    #endif

    return nbWriteFinalCheckpoint(ctx, st);
}