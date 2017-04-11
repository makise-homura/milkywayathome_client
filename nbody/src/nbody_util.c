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

#include "nbody_util.h"
#include "milkyway_math.h"

/* Correct timestep so an integer number of steps covers the exact
 * evolution time */
real nbCorrectTimestep(real timeEvolve, real dt)
{
    real nStep = mw_ceil(timeEvolve / dt);
    return timeEvolve / nStep;
}

mwvector nbCenterOfMass(const NBodyState* st)
{
    int i;
    const Body* b;
    int nbody = st->nbody;
    mwvector cm = ZERO_VECTOR;
    mwvector tmp;
    Kahan mass;
    Kahan pos[3];

    CLEAR_KAHAN(mass);
    memset(pos, 0, sizeof(pos));

    for (i = 0; i < nbody; ++i)
    {
        b = &st->bodytab[i];

        tmp = mw_mulvs(Pos(b), Mass(b));

        KAHAN_ADD(pos[0], tmp.x);
        KAHAN_ADD(pos[1], tmp.y);
        KAHAN_ADD(pos[2], tmp.z);
        KAHAN_ADD(mass, Mass(b));
    }

    X(cm) = pos[0].sum / mass.sum;
    Y(cm) = pos[1].sum / mass.sum;
    Z(cm) = pos[2].sum / mass.sum;
    W(cm) = mass.sum;

    return cm;
}

mwvector nbCenterOfMom(const NBodyState* st)
{
    int i;
    const Body* b;
    int nbody = st->nbody;
    mwvector cm = ZERO_VECTOR;
    mwvector tmp;
    Kahan mass;
    Kahan pos[3];

    CLEAR_KAHAN(mass);
    memset(pos, 0, sizeof(pos));

    for (i = 0; i < nbody; ++i)
    {
        b = &st->bodytab[i];
        tmp = mw_mulvs(Vel(b), Mass(b));

        KAHAN_ADD(pos[0], tmp.x);
        KAHAN_ADD(pos[1], tmp.y);
        KAHAN_ADD(pos[2], tmp.z);
        KAHAN_ADD(mass, Mass(b));
    }

    X(cm) = pos[0].sum / mass.sum;
    Y(cm) = pos[1].sum / mass.sum;
    Z(cm) = pos[2].sum / mass.sum;
    W(cm) = mass.sum;

    return cm;
}

void subtractMassMomentumCenters(const NBodyCtx* ctx, NBodyState* st)
{
/* This function calculates the center of mass and momentum
of a dwarf galaxy, and subtract each value from each body in the n-body
simulation. This function is only called when the timestep is negative,
meaning that we are in the "ramping" phase of the dwarf evolution */


    mwvector CenterMass;
    mwvector CenterMom;
    const Body* b;
    int nbody = st->nbody;
    
    CenterMass = nbCenterOfMass(st);
    CenterMom = nbCenterOfMom(st);

    /*We want to keep track of where the starting position was*/

    // begining_pos and begining_mom are set as static variables so that
    // time the function is called they hold their previous value, so
    // we can hold memory of the begining mass and momentum centers
    // throughout the duration of the simulation.
    static mwvector begining_pos = ZERO_VECTOR;
    static mwvector begining_mom = ZERO_VECTOR;

    /* This will only execute the first time this function is called
    meaning that the vectors will take the values of the initial
    centers of mass and momentum*/
    if(mw_length(begining_pos) == 0 && mw_length(begining_mom) == 0)
    {
        begining_pos = mw_addv(begining_pos, CenterMass);
        begining_mom = mw_addv(begining_mom, CenterMom);
    }


    mwvector position_change = mw_subv(CenterMass, begining_pos);
    mwvector mom_change = mw_subv(CenterMom, begining_mom);
    
    for ( int i = 0; i<nbody; ++i)
    {
        b = &st->bodytab[i]; // find out how to actually alter positions
        mwvector velocity = st->bodytab[i].vel;
        mwvector position = st->bodytab[i].bodynode.pos;
        
        
        st->bodytab[i].vel = mw_subv(velocity,mom_change);
        st->bodytab[i].bodynode.pos = mw_subv(position,position_change);
        //printf("%f %f %f\n", velocity.x, velocity.y,velocity.z);
    }
    

}

void resetVelocities(const NBodyCtx* ctx, NBodyState* st, mwvector CenterMom)
{
    const Body* b;
    int nbody = st->nbody;
    //mw_printf("Reseting Velocities \n");
    for ( int i = 0; i < nbody; ++i)
    {
        mwvector velocity = st->bodytab[i].vel;
        st->bodytab[i].vel = mw_addv(velocity,CenterMom);
    }
}
    

static inline real log8(real x)
{
    return mw_log(x) / mw_log(8.0);
}

/* The estimate formula has the unfortunate property of being negative
   for small n.  This will be the most negative. Add this as an extra
   boost to prevent negative flops estimates.
 */
static real worstFlops(real cQ, real d, real f)
{
    real a = mw_pow(2.0, 3.0 - 3.0 * d / cQ);
    real b = (cQ - d) * mw_log(8.0);
    real c = cQ * mw_log(mw_pow(8.0, 1.0 - d / cQ));

    return -a * sqr(f) * (cQ + b - c) / (M_E * mw_log(8.0));
}

/* Estimate number of operations based on formula derived in
   "A Practical Comparison of N-Body Algorithms" (Blelloc, Narlikar 1995)

   Should be more accurate for more uniform distributions.  Does not
   include the flops from the external potential. However, the effect
   of the potential actually reduces the total number of flops by
   tearing apart the system in general.

   Does not account for newer opening criteria.
 */
real nbEstimateNumberFlops(const NBodyCtx* ctx, int nbody)
{
    real quadTerm, baseTerm;

    real n = (real) nbody;
    real nSteps = ctx->timeEvolve / ctx->timestep;

    /* Cost of interaction for a cell using a quadrupole moment. */
    const real cQ = ctx->useQuad ? 50.0 : 0;

    /* Cost of a direct interaction */
    const real d = 13;

    /* Based on BH86 opening criterion. */
    real f = 28.0 * M_PI / (3.0 * cube(ctx->theta));

    /* FIXME: Don't be lazy and try rederiving for these. It should be
     * some number larger than for BH86. Somewhere I remember
     * something saying about 3x more operations for SW93
     */
    if (ctx->criterion != BH86)
    {
        f *= 3.0;
    }

    quadTerm = cQ * n * f * (log8(n / f) - 1.0);
    baseTerm = d * n * f;

    /* Total flops is then this times the number of timesteps */
    return nSteps * (quadTerm + baseTerm - worstFlops(cQ, d, f));
}

/* These estimates seem to sometimes work OK but very often not */
real nbEstimateTime(const NBodyCtx* ctx, int nbody, real flops)
{
    /* Spends < ~5% of the time in tree construction. Spends about
     * half the time in tree traversal / memory access as actually
     * calculating forces. */
    const real factor = 2.05;

    /* Not 100% efficient. Bullshit number */
    const real efficiency = 0.95;

    real nflop = nbEstimateNumberFlops(ctx, nbody);

    return factor * nflop / (efficiency * flops);
}

void nbReportTreeIncest(const NBodyCtx* ctx, NBodyState* st)
{
    if (!st->treeIncest)   /* don't repeat warning */
    {
        st->treeIncest = TRUE;

        if (!ctx->quietErrors) /* Avoid massive printing of tests causing incest */
        {
            if (ctx->allowIncest)
            {
                mw_printf("[tree-incest detected at step %u / %u (%f%%)]\n",
                          st->step,
                          ctx->nStep,
                          100.0 * (real) st->step / (real) ctx->nStep
                    );
            }
            else
            {
                mw_printf("tree-incest detected (fatal) at step %u / %u (%f%%)\n",
                          st->step,
                          ctx->nStep,
                          100.0 * (real) st->step / (real) ctx->nStep
                    );
            }
        }
    }
}