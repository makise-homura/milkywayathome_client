/*
  Copyright (c) 1993, 2001 Joshua E. Barnes, Honolulu, HI.
  Copyright 2010 Matthew Arsenault, Travis Desell, Boleslaw
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

#include <stdlib.h>
#include "nbody.h"
#include "nbody_params.h"
#include "calc_params.h"
#include "nbody_priv.h"
#include "milkyway_util.h"
#include "nbody_step.h"
#include "grav.h"
#include "orbitintegrator.h"
#include "show.h"

/* make test model */
static void generateModel(NBodyCtx* ctx, unsigned int modelIdx, bodyptr bodies)
{
    DwarfModel* model;
    dsfmt_t dsfmtState;

    model = &ctx->models[modelIdx];
    dsfmt_init_gen_rand(&dsfmtState, ctx->seed);

    switch (model->type)
    {
        #if 0
        case DwarfModelPlummer:
            generatePlummer(&dsfmtState,
                            bodies,
                            model->nbody,
                            &model->initialConditions,
                            model->mass,
                            model->scale_radius,
                            model->ignoreFinal);
            break;
        case DwarfModelKing:
        case DwarfModelDehnen:
        case InvalidDwarfModel:
            fail("Trying to run with invalid dwarf model\n");
        default:
            fail("Unsupported model: %d", model->type);
            #endif
    }
}

static void generateAllModels(NBodyCtx* ctx, bodyptr bodies)
{
    unsigned int i, bodyIdx;

    for (i = 0, bodyIdx = 0; i < ctx->modelNum; ++i)
    {
        generateModel(ctx, i, &bodies[bodyIdx]);
        bodyIdx += ctx->models[i].nbody;
    }
}

static void initState(NBodyCtx* ctx, NBodyState* st)
{
    mw_report("Starting nbody system\n");

    st->tree.rsize = ctx->tree_rsize;
    st->tnow       = 0.0;            /* reset elapsed model time */

    st->bodytab = (bodyptr) mwMalloc(ctx->nbody * sizeof(body));
    st->acctab  = (mwvector*) mwMalloc(ctx->nbody * sizeof(mwvector));

    generateAllModels(ctx, st->bodytab);

  #if NBODY_OPENCL
    setupNBodyCL(ctx, st);
  #endif /* NBODY_OPENCL */

    /* CHECKME: Why does makeTree get used twice for the first step? */
    /* Take 1st step */
  #if !NBODY_OPENCL
    gravMap(ctx, st);
  #else
    gravMapCL(ctx, st);
  #endif /* !NBODY_OPENCL */

}

static void startRun(NBodyCtx* ctx, NBodyState* st)
{
    mw_report("Starting fresh nbody run\n");
    reverseModelOrbits(ctx);
    initState(ctx, st);
}

#if BOINC_APPLICATION
static inline void nbodyCheckpoint(const NBodyCtx* ctx, const NBodyState* st)
{
    if (boinc_time_to_checkpoint())
    {
        if (writeCheckpoint(ctx, st))
            fail("Failed to write checkpoint\n");

        boinc_checkpoint_completed();
    }

    boinc_fraction_done(st->tnow / ctx->time_evolve);
}
#endif /* BOINC_APPLICATION */

static void runSystem(const NBodyCtx* ctx, NBodyState* st)
{
    const real tstop = ctx->time_evolve - ctx->timestep / 1024.0;

    while (st->tnow < tstop)
    {
        stepSystem(ctx, st);   /* advance N-body system */
      #if BOINC_APPLICATION
        nbodyCheckpoint(ctx, st);
      #endif

      #if PERIODIC_OUTPUT
        st->outputTime = (st->outputTime + 1) % ctx->freqOut;
        if (st->outputTime == 0)
            outputBodyPositionBin(ctx, st);
      #endif /* PERIODIC_OUTPUT */
    }

  #if BOINC_APPLICATION
    mw_report("Making final checkpoint\n");
    if (writeCheckpoint(ctx, st))
        fail("Failed to write final checkpoint\n");
  #endif

}

static void endRun(NBodyCtx* ctx, NBodyState* st, const real chisq)
{
    finalOutput(ctx, st, chisq);
    nbodyCtxDestroy(ctx);     /* finish up output */
    nbodyStateDestroy(st);
}

#if BOINC_APPLICATION

/* Setup the run, taking care of checkpointing things when using BOINC */
static int setupRun(NBodyCtx* ctx, NBodyState* st)
{
    /* If the checkpoint exists, try to use it */
    if (boinc_file_exists(ctx->cp_resolved))
    {
        mw_report("Checkpoint exists. Attempting to resume from it.\n");
        if (readCheckpoint(ctx, st))
        {
            mw_report("Failed to read checkpoint\n");
            nbodyStateDestroy(st);
            return 1;
        }
        else
        {
            mw_report("Successfully read checkpoint\n");
            /* We restored the useful state. Now still need to create
             * the workspace where new accelerations are
             * calculated. */
            st->acctab  = (mwvector*) mwMalloc(ctx->nbody * sizeof(mwvector));
          #if !NBODY_OPENCL
            gravMap(ctx, st);
          #else
            gravMapCL(ctx, st);
          #endif /* !NBODY_OPENCL */
        }
    }
    else   /* Otherwise, just start a fresh run */
    {
        startRun(ctx, st);
    }

    return 0;
}

#else

/* When not using BOINC, we don't need to deal with the checkpointing */
static int setupRun(NBodyCtx* ctx, NBodyState* st)
{
    startRun(ctx, st);
    return 0;
}

#endif /* BOINC_APPLICATION */

/* Set context fields read from command line flags */
static inline void nbodySetCtxFromFlags(NBodyCtx* ctx, const NBodyFlags* nbf)
{
    ctx->outputCartesian = nbf->outputCartesian;
    ctx->outputBodies    = nbf->printBodies;
    ctx->outputHistogram = nbf->printHistogram;
    ctx->outfilename     = nbf->outFileName;
    ctx->histogram       = nbf->histogramFileName;
    ctx->histout         = nbf->histoutFileName;
    ctx->cp_filename     = nbf->checkpointFileName;
}

static void verifyFile(const NBodyCtx* ctx, const HistogramParams* hp, int rc)
{
    printNBodyCtx(ctx);
    printHistogramParams(hp);

    if (rc)
        warn("File failed\n");
    else
        warn("File is OK\n");
    mw_finish(rc);
}

/* Takes parsed json and run the simulation, using outFileName for
 * output. */
void runNBodySimulation(json_object* obj,            /* The main configuration */
                        const FitParams* fitParams,  /* For server's arguments */
                        const NBodyFlags* nbf)       /* Misc. parameters to control output */
{
    NBodyCtx ctx  = EMPTY_CTX;
    NBodyState st = EMPTY_STATE;
    HistogramParams histParams;

    real chisq;
    double ts = 0.0, te = 0.0;
    int rc = 0;

    rc |= nbodyGetParamsFromJSON(&ctx, &histParams, obj);
    if (rc && !nbf->verifyOnly)   /* Fail right away, unless we are diagnosing file problems */
        fail("Failed to read input parameters file\n");

    rc |= setCtxConsts(&ctx, fitParams, nbf->setSeed);

    if (nbf->verifyOnly)
        verifyFile(&ctx, &histParams, rc);

    if (rc)
        fail("Failed to read input parameters file\n");

    nbodySetCtxFromFlags(&ctx, nbf);

  #if BOINC_APPLICATION
    if (resolveCheckpoint(&ctx))
        fail("Failed to resolve checkpoint\n");
  #endif /* BOINC_APPLICATION */

    if (initOutput(&ctx))
        fail("Failed to open output files\n");

    if (setupRun(&ctx, &st))
        fail("Failed to setup run\n");

    if (nbf->printTiming)     /* Time the body of the calculation */
        ts = mwGetTime();

    runSystem(&ctx, &st);
    mw_report("Simulation complete\n");

    if (nbf->printTiming)
    {
        te = mwGetTime();
        printf("<run_time> %g </run_time>\n", te - ts);
    }

    /* Get the likelihood */
    chisq = nbodyChisq(&ctx, &st, &histParams);
    if (isnan(chisq))
        warn("Failed to calculate chisq\n");

    endRun(&ctx, &st, chisq);
}

