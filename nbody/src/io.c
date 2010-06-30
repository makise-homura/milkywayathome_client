/* ************************************************************************** */
/* IO.C: I/O routines for export version of hierarchical N-body code. */
/* Public routines: inputdata(), initoutput(), stopoutput(), output(). */
/* */
/* Copyright (c) 1993 by Joshua E. Barnes, Honolulu, HI. */
/* It's free because it's yours. */
/* ************************************************************************** */

#include <string.h>
#include "nbody_priv.h"

#if BOINC_APPLICATION
  #include <boinc_api.h>
#endif

void initoutput(NBodyCtx* ctx)
{
    if (ctx->outfilename)                       /* output file specified? */
    {
        ctx->outfile = nbody_fopen(ctx->outfilename, "w");           /* setup output FILE* */
        if (ctx->outfile == NULL)
            fail("initoutput: cannot open file %s\n", ctx->outfilename);
    }
    else
        ctx->outfile = stdout;
}

/* Low-level input and output operations. */

static void out_2vectors(FILE* str, vector vec1, vector vec2)
{
    fprintf(str, " %21.14E %21.14E %21.14E %21.14E %21.14E %21.14E\n", vec1[0], vec1[1], vec1[2], vec2[0], vec2[1], vec2[2]);
}


static const char hdr[] = "mwnbody";

/* Should be given the same context as the dump */
inline static void readDump(const NBodyCtx* ctx, NBodyState* st, FILE* f)
{
    int nbody;

    char buf[sizeof(hdr)];

    fread(buf, sizeof(char), sizeof(hdr), f);
    fread(&nbody, sizeof(int), 1, f);

    /* TODO: Better checking of things */

    printf("read header = %s\n", hdr);
    printf("read nbody = %d\n", nbody);

    if (strcmp(hdr, buf))
        fail("Didn't find header for checkpoint file.\n");

    if (ctx->model.nbody != nbody)
        fail("Number of bodies in checkpoint file does not match number expected by context.\n");

    st->bodytab = allocate(ctx->model.nbody * sizeof(body));
    fread(&st->tout, sizeof(real), 1, f);
    fread(&st->tnow, sizeof(real), 1, f);
    fread(&st->nstep, sizeof(real), 1, f);
    fread(st->bodytab, sizeof(body), nbody, f);
}

inline static void dumpBodies(const NBodyCtx* ctx, const NBodyState* st, FILE* f)
{
    /* TODO: Error checking */
    /* TODO: I think the other things are unnecessary and can go away */
    fwrite(hdr, 1, sizeof(hdr), f);
    fwrite(&ctx->model.nbody, sizeof(int), 1, f);
    fwrite(&st->tout, sizeof(real), 1, f);
    fwrite(&st->tnow, sizeof(real), 1, f);
    fwrite(&st->nstep, sizeof(st->nstep), 1, f);
    fwrite((const void*) st->bodytab, sizeof(body), ctx->model.nbody, f);
}

void nbody_boinc_output(const NBodyCtx* ctx, NBodyState* st)
{
    FILE* f;
    /* TODO: Check for failure on write */
    if (boinc_time_to_checkpoint())
    {
        f = nbody_fopen("nbody_checkpoint", "wb");
        if (!f)
            fail("Failed to open checkpoint file for reading\n");

        dumpBodies(ctx, st, f);
        boinc_checkpoint_completed();
        fclose(f);
    }

    boinc_fraction_done(st->tnow / ctx->model.time_dwarf);
}

void nbody_boinc_read_checkpoint(const NBodyCtx* ctx, NBodyState* st)
{
    FILE* f;
    /* TODO: Check for failure on write */
    //if (boinc_time_to_checkpoint())
    f = nbody_fopen("nbody_checkpoint", "rb");
    if (!f)
        fail("Failed to open checkpoint file for reading\n");

    readDump(ctx, st, f);
    fclose(f);
}

inline static void cartesianToLbr(vectorptr restrict lbR, const vectorptr restrict r)
{
    lbR[0] = r2d(ratan2(r[1], r[0]));
    lbR[1] = r2d(ratan2(r[2], rsqrt((r[0]) * (r[0]) + r[1] * r[1])));
    lbR[2] = rsqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

    if (lbR[0] < 0)
        lbR[0] += 360.0;
}

/* OUTPUT: compute diagnostics and output data. */
void output(const NBodyCtx* ctx, const NBodyState* st)
{
    bodyptr p;
    vector lbR;
    const bodyptr endp = st->bodytab + ctx->model.nbody;

    for (p = st->bodytab; p < endp; p++)
    {
      #ifndef OUTPUT_CARTESIAN
        cartesianToLbr(lbR, Pos(p));
        out_2vectors(ctx->outfile, lbR, Vel(p));
      #else
        /* Probably useful for making movies and such */
        out_2vectors(ctx->outfile, Pos(p), Vel(p));
      #endif /* OUTPUT_CARTESIAN */
    }

    fflush(ctx->outfile);             /* drain output buffer */
}


/* A bunch of boilerplate for debug printing */

const char* showBool(const bool x)
{
    switch (x)
    {
        case FALSE:
            return "false";
        case TRUE:
            return "true";
        default:
            return "invalid boolean (but true)";
    }
}

const char* showCriterionT(const criterion_t x)
{
    static const char* table[] =
        {
            [EXACT]        = "exact",
            [NEWCRITERION] = "new criterion",
            [BH86]         = "bh86",
            [SW93]         = "sw93"
        };

    if (x > SW93)
        return "invalid criterion_t";
    else
        return table[x];

}

const char* showSphericalT(const spherical_t x)
{
    static const char* table[] =
        {
            [SphericalPotential]        = "SphericalPotential",
        };

    if (x > SphericalPotential)
        return "invalid spherical_t";
    else
        return table[x];
}

const char* showDiskT(const disk_t x)
{
    static const char* table[] =
        {
            [MiyamotoNagaiDisk] = "MiyamotoNagaiDisk",
            [ExponentialDisk]   = "ExponentialDisk"
        };

    if (x > ExponentialDisk)
        return "invalid disk_t";
    else
        return table[x];
}

const char* showHaloT(const halo_t x)
{
    static const char* table[] =
        {
            [LogarithmicHalo] = "LogarithmicHalo",
            [NFWHalo]         = "NFWHalo",
            [TriaxialHalo]    = "TriaxialHalo"
        };

    if (x > TriaxialHalo)
        return "invalid halo_t";
    else
        return table[x];
}

const char* showDwarfModelT(const dwarf_model_t x)
{
    static const char* table[] =
        {
            [DwarfModelPlummer] = "DwarfModelPlummer",
            [DwarfModelKing]    = "DwarfModelKing",
            [DwarfModelDehnen]  = "DwarfModelDehnen"
        };

    if (x > DwarfModelDehnen)
        return "invalid dwarf_model_t";
    else
        return table[x];
}

char* showSpherical(const Spherical* s)
{
    char* buf;

    if (0 > asprintf(&buf,
                     "{\n"
                     "      type  = %s\n"
                     "      mass  = %g\n"
                     "      scale = %g\n"
                     "    };\n",
                     showSphericalT(s->type),
                     s->mass,
                     s->scale))
    {
        fail("asprintf() failed\n");
    }

    return buf;
}

char* showHalo(const Halo* h)
{
    char* buf;

    if (0 > asprintf(&buf,
                     "{ \n"
                     "      type         = %s\n"
                     "      vhalo        = %g\n"
                     "      scale_length = %g\n"
                     "      flattenX     = %g\n"
                     "      flattenY     = %g\n"
                     "      flattenZ     = %g\n"
                     "      triaxAngle   = %g\n"
                     "    };\n",
                     showHaloT(h->type),
                     h->vhalo,
                     h->scale_length,
                     h->flattenX,
                     h->flattenY,
                     h->flattenZ,
                     h->triaxAngle))
    {
        fail("asprintf() failed\n");
    }

    return buf;
}

char* showDisk(const Disk* d)
{
    char* buf;

    if (0 > asprintf(&buf,
                     "{ \n"
                     "      type         = %s\n"
                     "      mass         = %g\n"
                     "      scale_length = %g\n"
                     "      scale_height = %g\n"
                     "    };\n",
                     showDiskT(d->type),
                     d->mass,
                     d->scale_length,
                     d->scale_height))
    {
        fail("asprintf() failed\n");
    }

    return buf;
}

/* For debugging. Need to make this go away for release since it uses
 * GNU extensions */
char* showPotential(const Potential* p)
{
    int rc;
    char* buf;
    char* sphBuf;
    char* diskBuf;
    char* haloBuf;

    sphBuf  = showSpherical(&p->sphere[0]);
    diskBuf = showDisk(&p->disk);
    haloBuf = showHalo(&p->halo);

    rc = asprintf(&buf,
                  "{\n"
                  "    sphere = %s\n"
                  "    disk = %s\n"
                  "    halo = %s\n"
                  "    rings  = { unused pointer %p }\n"
                  "  };\n",
                  sphBuf,
                  diskBuf,
                  haloBuf,
                  p->rings);

    if (rc < 0)
        fail("asprintf() failed\n");

    free(sphBuf);
    free(diskBuf);
    free(haloBuf);

    return buf;
}

char* showDwarfModel(const DwarfModel* d)
{
    char* buf;

    if (0 > asprintf(&buf,
                     "{ \n"
                     "      type           = %s\n"
                     "      nbody          = %d\n"
                     "      mass           = %g\n"
                     "      scale_radius   = %g\n"
                     "      timestep       = %g\n"
                     "      orbit_timestep = %g\n"
                     "      time_dwarf     = %g\n"
                     "      time_orbit     = %g\n"
                     "      eps            = %g\n"
                     "    };\n",
                     showDwarfModelT(d->type),
                     d->nbody,
                     d->mass,
                     d->scale_radius,
                     d->timestep,
                     d->orbit_timestep,
                     d->time_orbit,
                     d->time_dwarf,
                     d->eps))
    {
        fail("asprintf() failed\n");
    }

    return buf;
}

char* showInitialConditions(const InitialConditions* ic)
{
    char* buf;
    if (0 > asprintf(&buf,
                     "initial-conditions = { \n"
                     "  useGalC    = %s\n"
                     "  useRadians = %s\n"
                     "  sunGCDist  = %g\n"
                     "  position   = { %g, %g, %g }\n"
                     "  velocity   = { %g, %g, %g }\n"
                     "};\n",
                     showBool(ic->useGalC),
                     showBool(ic->useRadians),
                     ic->sunGCDist,
                     ic->position[0],
                     ic->position[1],
                     ic->position[2],
                     ic->velocity[0],
                     ic->velocity[1],
                     ic->velocity[2]))
    {
        fail("asprintf() failed\n");
    }

    return buf;
}

char* showContext(const NBodyCtx* ctx)
{
    char* buf;
    char* potBuf;
    char* modelBuf;

    potBuf   = showPotential(&ctx->pot);
    modelBuf = showDwarfModel(&ctx->model);

    if (0 > asprintf(&buf,
                     "ctx = { \n"
                     "  pot = %s\n"
                     "  model = %s\n"
                     "  headline    = %s\n"
                     "  outfilename = %s\n"
                     "  outfile     = %p\n"
                     "  criterion   = %s\n"
                     "  usequad     = %s\n"
                     "  allowIncest = %s\n"
                     "  seed        = %ld\n"
                     "  tree_rsize  = %g\n"
                     "  theta       = %g\n"
                     "  freq        = %g\n"
                     "  freqout     = %g\n"
                     "};\n",
                     potBuf,
                     modelBuf,
                     ctx->headline,
                     ctx->outfilename,
                     ctx->outfile,
                     showCriterionT(ctx->criterion),
                     showBool(ctx->usequad),
                     showBool(ctx->allowIncest),
                     ctx->seed,
                     ctx->tree_rsize,
                     ctx->theta,
                     ctx->freq,
                     ctx->freqout))
    {
        fail("asprintf() failed\n");
    }

    free(potBuf);
    free(modelBuf);

    return buf;
}

void printContext(const NBodyCtx* ctx)
{
    char* buf = showContext(ctx);
    puts(buf);
    free(buf);
}

void printInitialConditions(const InitialConditions* ic)
{
    char* buf = showInitialConditions(ic);
    puts(buf);
    free(buf);
}

char* showVector(const vector v)
{
    char* buf;

    if (asprintf(&buf, "{ %g, %g, %g }", v[0], v[1], v[2]) < 0)
        fail ("asprintf() failed\n");

    return buf;

}

