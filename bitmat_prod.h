#ifndef _BITMAT_PROD_H_
#define _BITMAT_PROD_H_

#include <stdint.h>
#include <emmintrin.h>
#include <immintrin.h>

static inline
__m128i bitmat_prod_accu_64x128_M4R_sse( __m128i r0 , const uint64_t * mat4R , uint64_t a )
{
	const __m128i * mat128 = (const __m128i*)mat4R;
	while( a ) {
		r0 ^= _mm_load_si128( mat128 + (a&0xf) );
		mat128 += 16;
		a >>= 4;
	}
	return r0;
}

static inline
void bitmatrix_prod_64x128_4R_sse( uint8_t * r , const uint64_t * mat4R , uint64_t a )
{
	__m128i r0 = _mm_setzero_si128();
	r0 = bitmat_prod_accu_64x128_M4R_sse( r0 , mat4R , a );
	_mm_store_si128( (__m128i *) r , r0 );
}

static inline
void bitmatrix_prod_128x128_4R_sse( uint8_t * r , const uint64_t * mat4R , const uint8_t *a )
{
	__m128i r0 = _mm_setzero_si128();
	const uint64_t *a64 = (const uint64_t*)a;
	r0 = bitmat_prod_accu_64x128_M4R_sse( r0 , mat4R , a64[0] );
	r0 = bitmat_prod_accu_64x128_M4R_sse( r0 , mat4R+2*256 , a64[1] );
	_mm_store_si128( (__m128i *) r , r0 );
}



static inline
__m256i bitmat_prod_accu_64x256_M4R_avx( __m256i r0 , const uint64_t * mat4R , uint64_t a )
{
	const __m256i * mat256 = (const __m256i*)mat4R;
	while( a ) {
		r0 ^= _mm256_load_si256( mat256 + (a&0xf) );
		mat256 += 16;
		a >>= 4;
	}
	return r0;
}


static inline
void bitmatrix_prod_128x256_4R_avx( uint8_t * r , const uint64_t * mat4R , const uint64_t *a )
{
	__m256i r0 = _mm256_setzero_si256();
	r0 = bitmat_prod_accu_64x256_M4R_avx( r0 , mat4R , a[0] );
	r0 = bitmat_prod_accu_64x256_M4R_avx( r0 , mat4R+16*16*4 , a[1] );
	_mm256_store_si256( (__m256i *) r , r0 );
}

static inline
void bitmatrix_prod_256x256_4R_avx( uint8_t * r , const uint64_t * mat4R , const uint64_t *a )
{
	__m256i r0 = _mm256_setzero_si256();
	r0 = bitmat_prod_accu_64x256_M4R_avx( r0 , mat4R , a[0] );
	r0 = bitmat_prod_accu_64x256_M4R_avx( r0 , mat4R+16*16*4 , a[1] );
	r0 = bitmat_prod_accu_64x256_M4R_avx( r0 , mat4R+16*16*4*2 , a[2] );
	r0 = bitmat_prod_accu_64x256_M4R_avx( r0 , mat4R+16*16*4*3 , a[3] );
	_mm256_store_si256( (__m256i *) r , r0 );
}


#endif
