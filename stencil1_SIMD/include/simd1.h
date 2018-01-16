/* --------------- */
/* --- simd1.h --- */
/* --------------- */

#ifndef __SIMD1_H__
#define __SIMD1_H__

#pragma message("  include  simd1.h")

#ifdef __cplusplus
#pragma message ("C++")
extern "C" {
#endif

/*
 * Definir ici les macros de calcul de vecteurs non alignes
 * vec_left4_ps(v0,v1)
 * vec_left3_ps(v0,v1)
 * vec_left2_ps(v0,v1)
 * vec_left1_ps(v0,v1)
 * vec_right1_ps(v1,v2)
 * vec_right2_ps(v1,v2)
 * vec_right3_ps(v1,v2)
 * vec_right4_ps(v1,v2)
 */


/*
#define vec_left1_ps(v0, v1)  v0
#define vec_left2_ps(v0, v1)  v0
#define vec_left3_ps(v0, v1)  v0
#define vec_left4_ps(v0, v1)  v0

#define vec_right1_ps(v1, v2) v1
#define vec_right2_ps(v1, v2) v1
#define vec_right3_ps(v1, v2) v1
#define vec_right4_ps(v1, v2) v2
*/
/*
 * Definir ici les macros d'extraction de min
 * vMIN3(x0,x1,x2)
 * vMIN5(x0,x1,x2,x3,x4)
 */

/*
#define vMIN2(x0,x1)         
#define vMIN3(x0,x1,x2)      
#define vMIN4(x0,x1,x2,x3)   
#define vMIN5(x0,x1,x2,x3,x4)
*/

/*
 * ===========================
 * === Liste des fonctions ===
 * ===========================
 */

void test_macro(void);

// addition ponctuelle
void add_vf32vector(vfloat32 *vX1, vfloat32 *vX2, int n, vfloat32 *vY);

// produit scalaire = reduction totale
vfloat32 dot_vf32vector(vfloat32 *vX1, vfloat32 *vX2, int n);

// reduction partielle
void sum3_vf32vector(vfloat32 *vX, int n, vfloat32 *vY);
void sum5_vf32vector(vfloat32 *vX, int n, vfloat32 *vY);

void min3_vf32vector(vfloat32 *vX, int n, vfloat32 *vY);
void min5_vf32vector(vfloat32 *vX, int n, vfloat32 *vY);

void positive_add3_vf32vector(vfloat32 *vX, int n, vfloat32 *vY);
void positive_avg3_vf32vector(vfloat32 *vX, int n, vfloat32 *vY);

// reduction totale
vfloat32 positive_avg_vf32vector(vfloat32 *vX, int n);

    
#ifdef __cplusplus
}
#endif

#endif /* __SIMD1_H__ */
