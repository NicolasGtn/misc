/* -------------------- */
/* --- test_simd1.c --- */
/* -------------------- */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "nrdef.h"
#include "nrutil.h"

#include "vnrdef.h"
#include "vnrutil.h"

#include "mutil.h"

#include "mymacro.h"
#include "simd_macro.h"
#include "simd1.h"

/* ========================== */
/* === Fonctions de tests === */
/* ========================== */

// -------------------------------
void test_add_dot_vf32vector(void)
// -------------------------------
{
    int n = 4*4;
    int card; // cardinal of vector type

    int si0, si1; // scalar indices
    int vi0, vi1; // vector indices
    int mi0, mi1; // memory (bounded) indices

    vfloat32 *vX1, *vX2, *vY, d;

    puts("------------------------------");
    puts("-- test_add_dot_vf32vector ---");
    puts("------------------------------");

    // ------------------------- //
    // -- calculs des indices -- //
    // ------------------------- //

    card = card_vfloat32();

    si0 = 0;
    si1 = n-1;
    s2v1D(si0, si1, card, &vi0, &vi1);
    v2m1D(vi0, vi1, card, &mi0, &mi1);
    
    // ------------------------------------------- //
    // -- allocation des tableaux 1D vectoriels -- //
    // ------------------------------------------- //

    vX1 = vf32vector(vi0, vi1);
    vX2 = vf32vector(vi0, vi1);
    vY  = vf32vector(vi0, vi1);

    // ---------- //
    // -- init -- //
    // ---------- //

    init_vf32vector_param(vX1, vi0, vi1, 1, 1);
    init_vf32vector_param(vX2, vi0, vi1, 1, 2);
    zero_vf32vector(vY, vi0, vi1);

    // --------------- //
    // -- affichage -- //
    // --------------- //

    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vX1, si0, si1, "%4.0f", "sX1");

    // affichage par bloc SIMD: appel de la fonction SIMD
    display_vf32vector(vX1, vi0, vi1, "%4.0f", "vX1"); puts("");

    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vX2, si0, si1, "%4.0f", "sX2");

    // affichage par bloc SIMD: appel de la fonction SIMD
    display_vf32vector(vX2, vi0, vi1, "%4.0f", "vX2"); puts("");

    // ------------ //
    // -- calcul -- //
    // ------------ //

    add_vf32vector(vX1, vX2, n/card, vY);
    d = dot_vf32vector(vX1, vX2, n/card);

    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vY, si0, si1, "%4.0f", "sY");

    // affichage par bloc SIMD: appel de la fonction SIMD
    display_vf32vector(vY, vi0, vi1, "%4.0f", "vY"); puts("");

    display_vfloat32(d, "%6.0f ", "dot"); puts("");

    // ---------- //
    // -- free -- //
    // ---------- //

    free_vf32vector(vX1, vi0, vi1);
    free_vf32vector(vX2, vi0, vi1);
    free_vf32vector(vY,  vi0, vi1);
}
// ---------------------------
void test_sum_vf32vector(void)
// ---------------------------
{
    int n = 4*4;
    int card; // cardinal of vector type

    int si0, si1; // scalar indices
    int vi0, vi1; // vector indices
    int mi0, mi1; // memory (bounded) indices

    vfloat32 *vX, *vY3, *vY5;

    puts("--------------------------");
    puts("-- test_sum_vf32vector ---");
    puts("--------------------------");

    // ------------------------- //
    // -- calculs des indices -- //
    // ------------------------- //

    card = card_vfloat32();

    si0 = 0;
    si1 = n-1;
    s2v1D(si0, si1, card, &vi0, &vi1);
    v2m1D(vi0, vi1, card, &mi0, &mi1);
    
    // ------------------------------------------- //
    // -- allocation des tableaux 1D vectoriels -- //
    // ------------------------------------------- //

    vX  = vf32vector(vi0, vi1+2);
    vY3 = vf32vector(vi0, vi1);
    vY5 = vf32vector(vi0, vi1);

    // ---------- //
    // -- init -- //
    // ---------- //

    init_vf32vector_param(vX, vi0+1, vi1+1, 1, 1);
    zero_vf32vector(vX, vi0, vi0);
    zero_vf32vector(vX, vi1+2, vi1+2);
    zero_vf32vector(vY3, vi0, vi1);
    zero_vf32vector(vY5, vi0, vi1);

    // --------------- //
    // -- affichage -- //
    // --------------- //

    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vX, si0, si1+8, "%4.0f", "sX1");

    // affichage par bloc SIMD: appel de la fonction SIMD
    //display_vf32vector(vX, vi0, vi1, "%4.0f", "vX"); puts("");

    // ------------ //
    // -- calcul -- //
    // ------------ //

    sum3_vf32vector(vX, n/card, vY3);
    sum5_vf32vector(vX, n/card, vY5);

    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vY3, si0, si1, "%4.0f", "sY3");

    // affichage par bloc SIMD: appel de la fonction SIMD
    //display_vf32vector(vY3, vi0, vi1, "%4.0f", "vY3"); puts("");

    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vY5, si0, si1, "%4.0f", "sY5");

    // affichage par bloc SIMD: appel de la fonction SIMD
    //display_vf32vector(vY5, vi0, vi1, "%4.0f", "vY5"); puts("");

    // ---------- //
    // -- free -- //
    // ---------- //

    free_vf32vector(vX,  vi0, vi1);
    free_vf32vector(vY3, vi0, vi1);
    free_vf32vector(vY5, vi0, vi1);
}
// ---------------------------
void test_min_vf32vector(void)
// ---------------------------
{
    int n = 4*4;
    int card; // cardinal of vector type

    int si0, si1; // scalar indices
    int vi0, vi1; // vector indices
    int mi0, mi1; // memory (bounded) indices

    vfloat32 *vX, *vY3, *vY5;

    puts("--------------------------");
    puts("-- test_min_vf32vector ---");
    puts("--------------------------");

    // ------------------------- //
    // -- calculs des indices -- //
    // ------------------------- //

    card = card_vfloat32();

    si0 = 0;
    si1 = n-1;
    s2v1D(si0, si1, card, &vi0, &vi1);
    v2m1D(vi0, vi1, card, &mi0, &mi1);
    
    // ------------------------------------------- //
    // -- allocation des tableaux 1D vectoriels -- //
    // ------------------------------------------- //
    vX  = vf32vector(vi0, vi1+2);
    vY3 = vf32vector(vi0, vi1);
    vY5 = vf32vector(vi0, vi1);

    // ---------- //
    // -- init -- //
    // ---------- //

    init_vf32vector_param(vX, vi0+1, vi1+1, 1, 2);
    init_vf32vector_param(vX, vi0, vi0, FLT_MAX, FLT_MAX);
    init_vf32vector_param(vX, vi1+2, vi1+2, FLT_MAX, FLT_MAX);
    rand_f32vector((float32*) vX, si0+4, si1+4); // generateur de nombres aleatoires (congruence lineaire)
    zero_vf32vector(vY3, vi0, vi1);
    zero_vf32vector(vY5, vi0, vi1);

    // --------------- //
    // -- affichage -- //
    // --------------- //

    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vX, si0+4, si1+4, "%4.0f", "sX1");

    // affichage par bloc SIMD: appel de la fonction SIMD
    //display_vf32vector(vX, vi0, vi1, "%4.0f", "vX"); puts("");

    // ------------ //
    // -- calcul -- //
    // ------------ //

    //min3_vf32vector(vX, vi0, vi1, vY3);
    //min5_vf32vector(vX, vi0, vi1, vY5);
    
    min3_vf32vector(vX, n/card, vY3);
    min5_vf32vector(vX, n/card, vY5);

    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vY3, si0, si1, "%4.0f", "sY3");
    display_f32vector((float32*) vY5, si0, si1, "%4.0f", "sY5");

    // ---------- //
    // -- free -- //
    // ---------- //

    free_vf32vector(vX,  vi0, vi1);
    free_vf32vector(vY3, vi0, vi1);
    free_vf32vector(vY5, vi0, vi1);
}
// ---------------------------
void test_cond_vf32ector(void)
// ---------------------------
{
    int b = 1;
    int n;
    int card; // cardinal of vector type
    
    int si0, si1; // scalar indices
    int vi0, vi1; // vector indices
    int mi0, mi1; // memory (bounded) indices
    
    int si0b, si1b; // scalar indices with border
    int vi0b, vi1b; // vector indices with border
    int mi0b, mi1b; // memory (bounded) indices  with border
    
    
    float32 *sX;
    vfloat32 *vX, *vY;
    vfloat32 avg;
    
    char* format = "%6.2f ";

    // chronometrie
    int iter, niter = 4;
    int run, nrun = 5;
    double t0, t1, dt, tmin, t;
    double cycles;
    
    puts("====================");
    puts("=== test 1D Cond ===");
    puts("====================");
    
    // ------------------------- //
    // -- calculs des indices -- //
    // ------------------------- //
    
    n=16;
    
    printf("n = %d\n", n);
    
    card = card_vfloat32();
    
    si0 = 0;
    si1 = n-1;
    
    s2v1D(si0, si1, card, &vi0, &vi1);
    v2m1D(vi0, vi1, card, &mi0, &mi1);
    
    si0b = si0-b;
    si1b = si1+b;
    
    s2v1D(si0b, si1b, card, &vi0b, &vi1b);
    v2m1D(vi0b, vi1b, card, &mi0b, &mi1b);
    
    // ------------------------------------------- //
    // -- allocation des tableaux 1D vectoriels -- //
    // ------------------------------------------- //
    
    vX = vf32vector (vi0b, vi1b);
    vY = vf32vector (vi0,  vi1);
    sX = (float32*) vX; // wrapper scalaire

    printf("si0 = %d\nsi1 = %d\nsi0b = %d\nsi1b = %d\n", si0, si1, si0b,si1b);
    // ---------- //
    // -- init -- //
    // ---------- //
    
    zero_vf32vector(vX, vi0b, vi1b);
    rand_f32vector((float32*)vX, 0, n-1);
    
    // forcage a zero pour cas particulier
    sX[3] = 0.0f;
    sX[4] = 0.0f;
    sX[5] = 0.0f;
    
    // --------------- //
    // -- affichage -- //
    // --------------- //
    
    // affichage classique sur une ligne: appel de la fonction scalaire
    display_f32vector((float32*) vX, si0, si1, "%6.0f ", "sX1");
    
    // affichage par bloc SIMD: appel de la fonction SIMD
    display_vf32vector(vX, vi0, vi1, "%6.0f ", "vX1");
    puts("");
    
    puts("----------------------");
    puts("--- addition3 cond ---");
    puts("----------------------");
    
    zero_vf32vector(vY, vi0b, vi1b);
    positive_add3_vf32vector(vX, n/card, vY);
    display_f32vector((float32*) vY, si0, si1, "%6.1f", "add3");
    
    puts("---------------------");
    puts("--- average3 cond ---");
    puts("---------------------");
    
    zero_vf32vector(vY, vi0b, vi1b);
    positive_avg3_vf32vector(vX, n/card, vY);
    display_f32vector((float32*) vY, si0, si1, "%6.1f", "avg3");
    
    puts("-----------------------------");
    puts("--- global reduction cond ---");
    puts("-----------------------------");
    
    zero_vf32vector(vX, vi0b, vi1b);
    avg = positive_avg_vf32vector(vX, n/card);
    display_vfloat32(avg, "%6.1f", "avg3"); puts("");
    
    // ---------- //
    // -- free -- //
    // ---------- //
    
    free_vf32vector(vX, vi0b, vi1b);
}
// ------------------
void test_simd1(void)
// ------------------
{
    puts("================");
    puts("== test_simd1 ==");
    puts("================");
    
    //test_add_dot_vf32vector();
    //test_sum_vf32vector();
    //test_min_vf32vector();
    test_cond_vf32ector();
}
