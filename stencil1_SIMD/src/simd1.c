/* --------------- */
/* --- simd1.c --- */
/* --------------- */

#include <stdio.h>
#include <stdlib.h>

#include "nrdef.h"
#include "nrutil.h"

#include "vnrdef.h"
#include "vnrutil.h"

#include "mutil.h"

#include "mymacro.h"
#include "simd_macro.h"
#include "simd1.h"

// ------------------
void test_macro(void)
// ------------------
{
    vfloat32 a, b, c, d, e;
    vfloat32 au, cu; // unaligned vector
    vfloat32 a3, a5; // add3 add5
    puts("------------------");
    puts("--- test_macro ---");
    puts("------------------"); puts("");
    
    puts("vec_left et vec_right");
    
    // setr simule le fonctionnement d'un load avec permutation des blocs grace au "r" = reverse
    a = _mm_setr_ps(1,   2,  3,  4);
    b = _mm_setr_ps(5,   6,  7,  8);
    c = _mm_setr_ps(9,  10, 11, 12);
    d = _mm_setr_ps(13, 14, 15, 16);
    e = _mm_setr_ps(17, 18, 19, 20);
    
    display_vfloat32(a, "%4.0f", "a "); puts("");
    display_vfloat32(b, "%4.0f", "b "); puts("");
    display_vfloat32(c, "%4.0f", "c "); puts("\n");
    
    au = vec_left1(a, b);
    cu = vec_right1(b, c);
    display_vfloat32(au, "%4.0f", "au"); puts("");
    display_vfloat32(cu, "%4.0f", "cu"); puts("\n");
    
    puts("vec_add3 et vec_add5");
    a3 = vec_add3(a, b, c);
    a5 = vec_add5(a, b, c, d, e);
    
    display_vfloat32(a3, "%4.0f", "a3"); puts("");
    display_vfloat32(a5, "%4.0f", "a5"); puts("\n");
}
// -------------------------------------------------------------------
void add_vf32vector(vfloat32 *vX1, vfloat32 *vX2, int n, vfloat32 *vY)
// -------------------------------------------------------------------
{
    vfloat32 x1, x2, y;

    for(int i=0; i<n; i++) {
        
        x1 = _mm_load_ps((float32*) &vX1[i]);
        x2 = _mm_load_ps((float32*) &vX2[i]);
        
        y = _mm_add_ps(x1, x2);
        
        DEBUG(display_vfloat32(x1, "%4.0f", "x1 =")); DEBUG(puts(""));
        DEBUG(display_vfloat32(x2, "%4.0f", "x2 =")); DEBUG(puts(""));
        DEBUG(display_vfloat32(y,  "%4.0f", "y  =")); DEBUG(puts(""));
        DEBUG(display_vfloat32(y,  "%4.0f", "y  =")); DEBUG(puts(""));
        
        _mm_store_ps((float*) &vY[i], y);
        
        DEBUG(puts("-------------------"));
    }
}
// ---------------------------------------------------------
vfloat32 dot_vf32vector(vfloat32 *vX1, vfloat32 *vX2, int n)
// ---------------------------------------------------------
{
    vfloat32 z[4];
    vfloat32 x1, x2, p, s;

    for(int i=0; i<n; i++) {   
        x1 = _mm_load_ps((float32*) &vX1[i]);
        x2 = _mm_load_ps((float32*) &vX2[i]);
        p = _mm_mul_ps(x1, x2);                         //p = [a,b,c,d]
        s = _mm_shuffle_ps(p, p, _MM_SHUFFLE(2,1,0,3)); //s = [b,c,d,a]
        s = _mm_add_ps(p, s);                           //s = [a+b,b+c,c+d,d+a]
        p = _mm_shuffle_ps(s, s, _MM_SHUFFLE(1,0,3,2)); //p = [c+d,a+d,a+b,b+c]
        _mm_store_ps(&z[i], _mm_add_ps(p, s));          //z = [a+b+c+d, ...]
    }
    p = _mm_shuffle_ps(z[0], z[1], _MM_SHUFFLE(3,2,3,2));
    s = _mm_shuffle_ps(z[2], z[3], _MM_SHUFFLE(3,2,3,2));
    s = _mm_shuffle_ps(p, s, _MM_SHUFFLE(3,1,3,1));

    return s; // attention il faut retourner un registre SIMD et non un scalaire

}
// ----------------------------------------------------
void sum3_vf32vector(vfloat32 *vX, int n, vfloat32 *vY)
// ----------------------------------------------------
{
    vfloat32 prev, curr, next;
    vfloat32 a,b;
    for(int i=1; i<=n; i++) {
        prev = _mm_load_ps((float32*) &vX[i-1]);
        curr = _mm_load_ps((float32*) &vX[i]);
        next = _mm_load_ps((float32*) &vX[i+1]);

        a = _mm_shuffle_ps(prev, curr, _MM_SHUFFLE(1,0,3,2));
        a = _mm_shuffle_ps(a, curr, _MM_SHUFFLE(2,1,2,1));

        b = _mm_shuffle_ps(curr, next, _MM_SHUFFLE(1,0,3,2));
        b = _mm_shuffle_ps(curr, b, _MM_SHUFFLE(2,1,2,1));

        curr = _mm_add_ps(curr, a);
        curr = _mm_add_ps(curr, b);

        _mm_store_ps((float*) &vY[i-1], curr);
    }
}
// ----------------------------------------------------
void sum5_vf32vector(vfloat32 *vX, int n, vfloat32 *vY)
// ----------------------------------------------------
{
    vfloat32 prev, curr, next;
    vfloat32 a,b,c,d;
    for(int i=1; i<=n; i++) {
        prev = _mm_load_ps((float32*) &vX[i-1]);
        curr = _mm_load_ps((float32*) &vX[i]);
        next = _mm_load_ps((float32*) &vX[i+1]);

        a = _mm_shuffle_ps(prev, curr, _MM_SHUFFLE(1,0,3,2));
        b = _mm_shuffle_ps(a, curr, _MM_SHUFFLE(2,1,2,1));

        c = _mm_shuffle_ps(curr, next, _MM_SHUFFLE(1,0,3,2));
        d = _mm_shuffle_ps(curr, c, _MM_SHUFFLE(2,1,2,1));

        a = _mm_add_ps(a, b);
        c = _mm_add_ps(c, d);
        a = _mm_add_ps(a,c);
        curr = _mm_add_ps(curr, a);

        _mm_store_ps((float*) &vY[i-1], curr);
    }
}
// ----------------------------------------------------
void min3_vf32vector(vfloat32 *vX, int n, vfloat32 *vY)
// ----------------------------------------------------
{
    vfloat32 prev, curr, next;
    vfloat32 a,b;
    for(int i=1; i<=n; i++) {
        prev = _mm_load_ps((float32*) &vX[i-1]);
        curr = _mm_load_ps((float32*) &vX[i]);
        next = _mm_load_ps((float32*) &vX[i+1]);

        a = _mm_shuffle_ps(prev, curr, _MM_SHUFFLE(1,0,3,2));
        a = _mm_shuffle_ps(a, curr, _MM_SHUFFLE(2,1,2,1));

        b = _mm_shuffle_ps(curr, next, _MM_SHUFFLE(1,0,3,2));
        b = _mm_shuffle_ps(curr, b, _MM_SHUFFLE(2,1,2,1));

        curr = _mm_min_ps(curr, a);
        curr = _mm_min_ps(curr, b);

        _mm_store_ps((float*) &vY[i-1], curr);
    }
}
// ----------------------------------------------------
void min5_vf32vector(vfloat32 *vX, int n, vfloat32 *vY)
// ----------------------------------------------------
{
    vfloat32 prev, curr, next;
    vfloat32 a,b,c,d;
    for(int i=1; i<=n; i++) {
        prev = _mm_load_ps((float32*) &vX[i-1]);
        curr = _mm_load_ps((float32*) &vX[i]);
        next = _mm_load_ps((float32*) &vX[i+1]);

        a = _mm_shuffle_ps(prev, curr, _MM_SHUFFLE(1,0,3,2));
        b = _mm_shuffle_ps(a, curr, _MM_SHUFFLE(2,1,2,1));

        c = _mm_shuffle_ps(curr, next, _MM_SHUFFLE(1,0,3,2));
        d = _mm_shuffle_ps(curr, c, _MM_SHUFFLE(2,1,2,1));

        a = _mm_min_ps(a, b);
        c = _mm_min_ps(c, d);
        a = _mm_min_ps(a,c);
        curr = _mm_min_ps(curr, a);

        _mm_store_ps((float*) &vY[i-1], curr);
    }
}
// -------------------------------------------------------------
void positive_add3_vf32vector(vfloat32 *vX, int n, vfloat32 *vY)
// -------------------------------------------------------------
{
    vfloat32 x, y , a, *vT;

    //vT = vf32vector (-1, 16);

    for(int i=1; i<=n; i++) {
        x = _mm_load_ps((float32*) &vX[i]);
        y = _mm_load_ps((float32*) &vY[i]);

        a = _mm_cmplt_ps(y, x);
        a = _mm_and_ps(a, x);
        display_f32vector((float32*) &a, 0, 3, "%6.0f ", "a");
    }
/*
    k1=_mm_set_ps(1,1,1,1);
    k2=_mm_set_ps(-2,-2,-2,-2);
    c=_mm_cmplt_ps(a,b);
    k=_mm_or_ps(_mm_and_ps(c,k2),_mm_andnot_ps(c,k1));
    d=_mm_add_ps(d,k); // incrément*/
}
// -------------------------------------------------------------
void positive_avg3_vf32vector(vfloat32 *vX, int n, vfloat32 *vY)
// -------------------------------------------------------------
{
}
// --------------------------------------------------
vfloat32 positive_avg_vf32vector(vfloat32 *vX, int n)
// --------------------------------------------------
{
    vfloat32 avg3 = _mm_set1_ps(0.0f);
    
    return avg3;

}