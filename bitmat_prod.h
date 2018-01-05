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



#include "transpose.h"


static inline
void bitmatrix_prod_128x128_4R_b32_avx2( uint8_t * r32 , const uint64_t * matB4R , const uint8_t *a32 )
{
	uint8_t t32[32*16] __attribute__((aligned(32)));
	tr_16x16_b2_avx2( t32 , a32 );

	__m256i *r = (__m256i*) r32;
	const __m256i * tab = (const __m256i*) & matB4R[0];
	__m256i _0xf = _mm256_set1_epi8(0xf);

	for(unsigned i=0;i<16;i++) r[i] = _mm256_setzero_si256();

for(unsigned k=0;k<16;k+=8) {
	for(unsigned i=0;i<16;i++) {
		__m256i temp = _mm256_load_si256( (__m256i*)(t32 + i*32) );
		__m256i low1_low0 = temp&_0xf;
		__m256i high1_high0 = _mm256_srli_epi16( _mm256_andnot_si256( _0xf ,  temp ) , 4 );

		__m256i high1_low0 = _mm256_permute2x128_si256( low1_low0 , high1_high0 , 0x30 );
		__m256i low1_high0 = _mm256_permute2x128_si256( low1_low0 , high1_high0 , 0x12);

		for(unsigned j=k;j<k+8;j++) {
			//__m256i _tab = _mm256_load_si256( &tab[i*16+j] );
			__m256i _tab = tab[i*16+j];
			__m256i tab_r = _mm256_permute4x64_epi64( _tab , 0x4e );
			r[j] ^= _mm256_shuffle_epi8( _tab , high1_low0 ) ^ _mm256_shuffle_epi8( tab_r , low1_high0 );
		}
	}
}

	tr_16x16_b2_avx2( r32 , r32 );
}




static inline
void bitmatrix_prod_128x128_4R_b32_opt_avx2( uint8_t * r32 , const uint64_t * matB4R , const uint8_t *a32 )
{
	uint8_t t32[32*16] __attribute__((aligned(32)));
	tr_16x16_b2_avx2( t32 , a32 );

	__m256i *r = (__m256i*) r32;
	const __m256i * tab = (const __m256i*) & matB4R[0];
	__m256i _0xf = _mm256_set1_epi8(0xf);

	for(unsigned i=0;i<8;i++) r[i] = _mm256_setzero_si256();
	for(unsigned i=0;i<16;i++) {
		__m256i temp = _mm256_load_si256( (__m256i*)(t32 + i*32) );
		__m256i low1_low0 = temp&_0xf;
		__m256i high1_high0 = _mm256_srli_epi16( _mm256_andnot_si256( _0xf ,  temp ) , 4 );

		for(unsigned j=0;j<8;j++) {
			__m256i tab_0 = tab[i*16+(j<<1)];
			__m256i tab_1 = tab[i*16+(j<<1)+1];
			r[j] ^= _mm256_shuffle_epi8( tab_0 , low1_low0 ) ^ _mm256_shuffle_epi8( tab_1 , high1_high0 );
		}
	}
	tab += 16*16;
	r += 8;
	for(unsigned i=0;i<8;i++) r[i] = _mm256_setzero_si256();
	for(unsigned i=0;i<16;i++) {
		__m256i temp = _mm256_load_si256( (__m256i*)(t32 + i*32) );
		__m256i low1_low0 = temp&_0xf;
		__m256i high1_high0 = _mm256_srli_epi16( _mm256_andnot_si256( _0xf ,  temp ) , 4 );

		for(unsigned j=0;j<8;j++) {
			__m256i tab_0 = tab[i*16+(j<<1)];
			__m256i tab_1 = tab[i*16+(j<<1)+1];
			r[j] ^= _mm256_shuffle_epi8( tab_0 , low1_low0 ) ^ _mm256_shuffle_epi8( tab_1 , high1_high0 );
		}
	}

	tr_16x16_b2_avx2( r32 , r32 );
}




static inline
void bitmatrix_prod_64x128_4R_b32_opt_avx2( uint8_t * r128_32 , const uint64_t * matB4R , const uint8_t *a64_32 )
{
	uint8_t t32[32*8] __attribute__((aligned(32)));
	/// 0x10,0x11,0x12,.....0x2f
	tr_8x8_b4_avx2( t32 , a64_32 , 32 );
	/// bitsliced: 0x10,0x14,0x18,0x1c, 0x20,0x24,0x28,0x2c, 0x11,0x15,0x19,0x1d, 0x21,0x25,0x29,0x2d, |
	///            0x12,0x16,0x1a,0x1e, 0x22,0x26,0x2a,0x2e, 0x13,0x17,0x1b,0x1f, 0x23,0x27,0x2b,0x2f,

	__m256i *r = (__m256i*) r128_32;
	const __m256i * tab = (const __m256i*) & matB4R[0];
	__m256i _0xf = _mm256_set1_epi8(0xf);

	for(unsigned i=0;i<8;i++) r[i] = _mm256_setzero_si256();
	for(unsigned i=0;i<8;i++) { ///
		__m256i temp = _mm256_load_si256( (__m256i*)(t32 + i*32) );
		__m256i low1_low0 = temp&_0xf;
		__m256i high1_high0 = _mm256_srli_epi16( _mm256_andnot_si256( _0xf ,  temp ) , 4 );

		for(unsigned j=0;j<8;j++) {
			__m256i tab_0 = tab[i*16+(j<<1)];
			__m256i tab_1 = tab[i*16+(j<<1)+1];
			r[j] ^= _mm256_shuffle_epi8( tab_0 , low1_low0 ) ^ _mm256_shuffle_epi8( tab_1 , high1_high0 );
		}
	}
	tab += 16*16;
	r += 8;
	for(unsigned i=0;i<8;i++) r[i] = _mm256_setzero_si256();
	for(unsigned i=0;i<8;i++) { ///
		__m256i temp = _mm256_load_si256( (__m256i*)(t32 + i*32) );
		__m256i low1_low0 = temp&_0xf;
		__m256i high1_high0 = _mm256_srli_epi16( _mm256_andnot_si256( _0xf ,  temp ) , 4 );

		for(unsigned j=0;j<8;j++) {
			__m256i tab_0 = tab[i*16+(j<<1)];
			__m256i tab_1 = tab[i*16+(j<<1)+1];
			r[j] ^= _mm256_shuffle_epi8( tab_0 , low1_low0 ) ^ _mm256_shuffle_epi8( tab_1 , high1_high0 );
		}
	}

	/// bitsliced: 0x10,0x14,0x18,0x1c, 0x20,0x24,0x28,0x2c, 0x11,0x15,0x19,0x1d, 0x21,0x25,0x29,0x2d, |
	///            0x12,0x16,0x1a,0x1e, 0x22,0x26,0x2a,0x2e, 0x13,0x17,0x1b,0x1f, 0x23,0x27,0x2b,0x2f,
	tr_16x16_b2_avx2( r128_32 , r128_32 );
	/// 0x10,0x12,0x14,....,0x2e,0x11,0x13,0x15,....,0x2d,0x2f
}




#endif
