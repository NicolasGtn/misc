
/* --------------- */
/* --- mutil.c --- */
/* --------------- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nrdef.h"
#include "nrutil.h"
#include "vnrdef.h"
#include "vnrutil.h"

#include "mutil.h"

/*
#include <stdint.h>
#include <string.h> // memset
#include <mach/mach_time.h>
//Framework Kernel
double chrono(void)
{
    static int first_time = 1;
    static double conversion = 0.0;
    
    static mach_timebase_info_data_t timebase;
    static kern_return_t err;
    
    uint64_t t = mach_absolute_time();
    
    if(first_time) {
        memset( &timebase, 0, sizeof( timebase ));
        err = mach_timebase_info( &timebase );
        //if( 0 != err ) return err;
        if(err != 0) return 0.0;
        conversion = 1e-9 * (double) timebase.numer / (double) timebase.denom;
        //printf("numer = %d denom = %d\n", timebase.numer, timebase.denom);
        first_time = 0;
    }
    return conversion * (double) t;
}
*/
double chrono(void)
{
    //return dtime();
    return 0.0;
}

/* ----------- */
/* --- CPP --- */
/* ----------- */
/* --------------------- */
double cpp(double t, int n)
/* --------------------- */
{
    double c = n;
    double cpp = t * FREQ / c;
    return cpp;
}
/* ---------------- */
/* --- routines --- */
/* ---------------- */

// ----------------
uint8 ui8rand(void)
// ----------------
{
    static uint32 x = 0;
    x = (11 * x + 13) % 17;
    return (uint8) x;
}
// ------------------
uint32 ui32rand(void)
// ------------------
{
    static uint32 x = 0;
    x = (11 * x + 13) % 17;
    return (uint32) x;
}
// -----------------
uint32 f32rand(void)
// -----------------
{
    static float32 x = 0.0f;
    x = (float32) fmod(11 * x + 13, 17);
    return x;
}
// -----------------
uint32 f32srand(void)
// -----------------
{
    // version signee
    static float32 x = 0.0f;
    
    x = (float32) fmod(11 * x + 13, 17) - 8;
    
    return x;
}
/* --------------------------------------- */
void rand_ui8vector(uint8 *X, int i0, int i1)
/* --------------------------------------- */
{
    int i;
    
    for(i=i0; i<=i1; i++) {
        X[i] = ui8rand();
    }
}
/* ----------------------------------------- */
void rand_ui32vector(uint32 *X, int i0, int i1)
/* ----------------------------------------- */
{
    int i;
    
    for(i=i0; i<=i1; i++) {
        X[i] = ui32rand();
    }
}
/* ----------------------------------------- */
void rand_f32vector(float32 *X, int i0, int i1)
/* ----------------------------------------- */
{
    int i;
    
    for(i=i0; i<=i1; i++) {
        X[i] = f32rand();
    }
}
// ---------------------------------------------
void srand_f32vector(float32 *X, int i0, int i1)
// ---------------------------------------------
{
    int i;
    
    for(i=i0; i<=i1; i++) {
        X[i] = f32srand();
    }
}
/* --------------- */
int getIter(int size)
/* --------------- */
{
    if(size <   32) return 1024;
    if(size <   64) return 64 ;
    if(size <  128) return 8;
    if(size <  256) return 2;
    if(size <  512) return 2;
    if(size < 1024) return 2;
    if(size < 2048) return 1;
    return 2;
}
/* ----------------- */
int getIterAV(int size)
/* ----------------- */
{
    return 3*getIter(size);
}
/* --------------------------------- */
float32 gauss(float32 sigma, float32 x)
/* --------------------------------- */
{
    float32 pi = 3.1415927;
    float32 y;
    y = 1.0f / (sqrt(2*pi)*sigma) * exp(-(x*x)/(2*sigma*sigma));
    return y;
}
