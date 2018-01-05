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


#include <stdint.h>

#include "gfext_aesni.h"

#include "bitmat_prod.h"

#include "byte_inline_func.h"

#include "config_profile.h"

#include "string.h"


/////////////////////////////////////////////////
///
/// pclmulqdq version
///
//////////////////////////////////////////////////////

#include "gf2128_cantor_iso.h"





static inline
void butterfly( __m128i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	uint8_t ska_iso[16] __attribute__((aligned(32)));
	bitmatrix_prod_64x128_8R_sse( ska_iso , gfCantorto2128_8R , ska );
	__m128i a = _mm_load_si128( (__m128i*) ska_iso );
	a ^= extra_a;

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[i] ^= _gf2ext128_mul_sse( poly[unit_2+i] , a );
		poly[unit_2+i] ^= poly[i];
	}

}


static inline
void i_butterfly( __m128i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	uint8_t ska_iso[16] __attribute__((aligned(32)));
	bitmatrix_prod_64x128_8R_sse( ska_iso , gfCantorto2128_8R , ska );
	__m128i a = _mm_load_si128( (__m128i*) ska_iso );
	a ^= extra_a;

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		poly[i] ^= _gf2ext128_mul_sse( poly[unit_2+i] , a );
	}

}




static inline
void butterfly_avx2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	uint8_t ska_iso[16] __attribute__((aligned(32)));
	bitmatrix_prod_64x128_8R_sse( ska_iso , gfCantorto2128_8R , ska );
	__m128i a = _mm_load_si128( (__m128i*) ska_iso );
	a ^= extra_a;

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[i] ^= _gf2ext128_mul_2x1_avx2( poly[unit_2+i] , a );
		poly[unit_2+i] ^= poly[i];
	}

}


static inline
void i_butterfly_avx2( __m256i * poly , unsigned unit , unsigned ska , __m128i extra_a )
{
	uint8_t ska_iso[16] __attribute__((aligned(32)));
	bitmatrix_prod_64x128_8R_sse( ska_iso , gfCantorto2128_8R , ska );
	__m128i a = _mm_load_si128( (__m128i*) ska_iso );
	a ^= extra_a;

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		poly[i] ^= _gf2ext128_mul_2x1_avx2( poly[unit_2+i] , a );
	}

}


/////////////////////////////////////////////////////



void btfy( uint64_t * fx , unsigned n_fx , unsigned scalar_a )
{

	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );
	unsigned n_terms = n_fx;

	unsigned i=log_n;

	__m256i * poly256 = (__m256i*) &fx[0];
	for( ; i>1 ; i--) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a - k > 0 ) ? _mm_load_si128( (__m128i*) (&gfCantorto2128[2*(scalar_a-k -1)]) ) : _mm_setzero_si128();

		for(unsigned j=0;j<num;j++) {
			butterfly_avx2( poly256 + j*unit , unit , j<<1 , extra_a );
		}
	}
	__m128i * poly128 = (__m128i*) &fx[0];
	if( i>0 ) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a - k > 0 ) ? _mm_load_si128( (__m128i*) (&gfCantorto2128[2*(scalar_a-k -1)]) ) : _mm_setzero_si128();

		for(unsigned j=0;j<num;j++) {
			butterfly( poly128 + j*unit , unit , j<<1 , extra_a );
		}
	}
}


void i_btfy( uint64_t * fx , unsigned n_fx , unsigned scalar_a )
{
	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );

	unsigned n_terms = n_fx;

	unsigned i=1;
	__m128i *poly128 = (__m128i*) &fx[0];
	for( ; i < 2; i++) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a - k > 0 ) ? _mm_load_si128( (__m128i*) (&gfCantorto2128[2*(scalar_a-k -1)]) ) : _mm_setzero_si128();

		for(unsigned j=0;j<num;j++) {
			i_butterfly( poly128 + j*unit , unit , j<<1 , extra_a );
		}
	}

	__m256i *poly256 = (__m256i*) &fx[0];
	for( ; i <= log_n; i++) {
		unsigned unit = (1<<(i-1));
		unsigned num = (n_terms>>1) / unit;

		unsigned k = i-1;
		__m128i extra_a = (scalar_a - k > 0 ) ? _mm_load_si128( (__m128i*) (&gfCantorto2128[2*(scalar_a-k -1)]) ) : _mm_setzero_si128();

		for(unsigned j=0;j<num;j++) {
			i_butterfly_avx2( poly256 + j*unit , unit , j<<1 , extra_a );
		}
	}

}






