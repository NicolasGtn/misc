/* -------------------- */
/* --- test_simd1.h --- */
/* -------------------- */

#ifndef __TEST_SIMD1_H__
#define __TEST_SIMD1_H__

#ifdef __cplusplus
#pragma message ("C++")
extern "C" {
#endif

void test_add_dot_vf32vector(void);
void test_sum_vf32vector(void);
void test_min_vf32vector(void);
void test_cond_vf32ector(void);

void test_simd1(void);
    
#ifdef __cplusplus
}
#endif

#endif /* __TEST_SIMD1_H__ */
