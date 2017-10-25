/*
Copyright (C) 2017 Ming-Shing Chen

This file is part of BitPolyMul.

BitPolyMul is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BitPolyMul is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with BitPolyMul.  If not, see <http://www.gnu.org/licenses/>.
*/

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
__m256i bitmat_prod_128x128_x2_4R_sse( const uint64_t * mat4R , __m256i a )
{
	uint64_t a64[4] __attribute__((aligned(32)));
	_mm256_store_si256( (__m256i*) a64 , a );

	__m128i r0 = _mm_setzero_si128();
	__m128i r1 = _mm_setzero_si128();
	r0 = bitmat_prod_accu_64x128_M4R_sse( r0 , mat4R , a64[0] );
	r1 = bitmat_prod_accu_64x128_M4R_sse( r1 , mat4R , a64[2] );
	r0 = bitmat_prod_accu_64x128_M4R_sse( r0 , mat4R+2*256 , a64[1] );
	r1 = bitmat_prod_accu_64x128_M4R_sse( r1 , mat4R+2*256 , a64[3] );

	__m256i r = _mm256_castsi128_si256( r0 );
	return _mm256_inserti128_si256( r , r1 , 1 );
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


static inline
void bitmatrix_prod_tri256x256_4R_avx( uint8_t * r , const uint64_t * mat4R_128 , const uint64_t * mat4R_256h , const uint64_t *a )
{
	__m128i r0_128 = _mm_setzero_si128();
	r0_128 = bitmat_prod_accu_64x128_M4R_sse( r0_128 , mat4R_128 , a[0] );
	r0_128 = bitmat_prod_accu_64x128_M4R_sse( r0_128 , mat4R_128+2*256 , a[1] );

	__m256i r0 = _mm256_inserti128_si256( _mm256_setzero_si256() , r0_128 , 0 );
	r0 = bitmat_prod_accu_64x256_M4R_avx( r0 , mat4R_256h , a[2] );
	r0 = bitmat_prod_accu_64x256_M4R_avx( r0 , mat4R_256h+16*16*4 , a[3] );
	_mm256_store_si256( (__m256i *) r , r0 );
}




static inline
__m128i bitmat_prod_accu_64x128_M6R_sse( __m128i r0 , const uint64_t * mat4R , uint64_t a )
{
	const __m128i * mat128 = (const __m128i*)mat4R;
	while( a ) {
		r0 ^= _mm_load_si128( mat128 + (a&0x3f) );
		mat128 += 64;
		a >>= 6;
	}
	return r0;
}

static inline
void bitmatrix_prod_64x128_6R_sse( uint8_t * r , const uint64_t * mat4R , uint64_t a )
{
	__m128i r0 = _mm_setzero_si128();
	r0 = bitmat_prod_accu_64x128_M6R_sse( r0 , mat4R , a );
	_mm_store_si128( (__m128i *) r , r0 );
}

static inline
void bitmatrix_prod_128x128_6R_sse( uint8_t * r , const uint64_t * mat4R , const uint8_t *a )
{
	__m128i r0 = _mm_setzero_si128();
	const uint64_t *a64 = (const uint64_t*)a;
	r0 = bitmat_prod_accu_64x128_M6R_sse( r0 , mat4R , a64[0] );
	r0 = bitmat_prod_accu_64x128_M6R_sse( r0 , mat4R+2*64*11 , a64[1] );
	_mm_store_si128( (__m128i *) r , r0 );
}




static inline
__m128i bitmat_prod_accu_64x128_M8R_sse( __m128i r0 , const uint64_t * mat4R , uint64_t a )
{
	const __m128i * mat128 = (const __m128i*)mat4R;
	while( a ) {
		r0 ^= _mm_load_si128( mat128 + (a&0xff) );
		mat128 += 256;
		a >>= 8;
	}
	return r0;
}

static inline
void bitmatrix_prod_64x128_8R_sse( uint8_t * r , const uint64_t * mat4R , uint64_t a )
{
	__m128i r0 = _mm_setzero_si128();
	r0 = bitmat_prod_accu_64x128_M8R_sse( r0 , mat4R , a );
	_mm_store_si128( (__m128i *) r , r0 );
}

static inline
void bitmatrix_prod_128x128_8R_sse( uint8_t * r , const uint64_t * mat4R , const uint8_t *a )
{
	__m128i r0 = _mm_setzero_si128();
	const uint64_t *a64 = (const uint64_t*)a;
	r0 = bitmat_prod_accu_64x128_M8R_sse( r0 , mat4R , a64[0] );
	r0 = bitmat_prod_accu_64x128_M8R_sse( r0 , mat4R+2*256*8 , a64[1] );
	_mm_store_si128( (__m128i *) r , r0 );
}



#endif
