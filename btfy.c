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


static
void butterfly( __m128i * poly , unsigned unit , unsigned ska , unsigned extra_a )
{
	uint8_t ska_iso[16] __attribute__((aligned(32)));
	bitmatrix_prod_64x128_8R_sse( ska_iso , gfCantorto2128_8R , ska );
	if( extra_a ) {
		__m128i a = _mm_load_si128( (__m128i*) ska_iso );
		a ^= _mm_load_si128( (__m128i*) (&gfCantorto2128[2*(extra_a-1)]) );
		_mm_store_si128( (__m128i*) ska_iso , a );
	}

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		__m128i r;
		gf2ext128_mul_sse( (uint8_t*)&r , (uint8_t*)&poly[unit_2+i] , ska_iso );
		poly[i] ^= r;
		poly[unit_2+i] ^= poly[i];
	}

}


static
void i_butterfly( __m128i * poly , unsigned unit , unsigned ska , unsigned extra_a )
{
	uint8_t ska_iso[16] __attribute__((aligned(32)));
	bitmatrix_prod_64x128_8R_sse( ska_iso , gfCantorto2128_8R , ska );
	if( extra_a ) {
		__m128i a = _mm_load_si128( (__m128i*) ska_iso );
		a ^= _mm_load_si128( (__m128i*) (&gfCantorto2128[2*(extra_a-1)]) );
		_mm_store_si128( (__m128i*) ska_iso , a );
	}

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		__m128i r;
		gf2ext128_mul_sse( (uint8_t*)&r , (uint8_t*)&poly[unit_2+i] , ska_iso );
		poly[i] ^= r;
	}

}


/////////////////////////////////////////////////////



void btfy( uint64_t * fx , unsigned n_fx , unsigned scalar_a )
{

	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );
	unsigned n_terms = n_fx;

	__m128i * poly = (__m128i*) &fx[0];

	for(unsigned i=log_n; i>0 ; i--) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		unsigned extra_a = (scalar_a > k)? scalar_a - k : 0;

		for(unsigned j=0;j<num;j++) {
			butterfly( poly + j*unit , unit , j<<1 , extra_a );
		}
	}
}


void i_btfy( uint64_t * fx , unsigned n_fx , unsigned scalar_a )
{
	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );

	__m128i *poly = (__m128i*) &fx[0];
	unsigned n_terms = n_fx;

	for(unsigned i=1; i <= log_n; i++) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		unsigned k = i-1;
		unsigned extra_a = (scalar_a > k)? scalar_a - k : 0;

		for(unsigned j=0;j<num;j++) {
			i_butterfly( poly + j*unit , unit , j<<1 , extra_a );
		}
	}
}








/////////////////////////////////////////////////
///
/// truncated FFT,  7 layers
///
//////////////////////////////////////////////////////

#include "trunc_btfy_tab.h"

#include "transpose.h"

#include "bc_tab.h"



static inline
void bit_bc_64x2_div( __m256i * x )
{
	for(unsigned i=0;i<64;i++) x[64+i] = _mm256_srli_si256( x[i] , 8 );
	for(int i=64-1;i>=0;i--) {
		x[1+i] ^= x[64+i];
		x[4+i] ^= x[64+i];
		x[16+i] ^= x[64+i];
	}
	for(unsigned i=0;i<64;i++) x[i] = _mm256_unpacklo_epi64( x[i] , x[64+i] );

	for(unsigned i=0;i<64;i+=64) {
		__m256i * pi = x + i;
		for(int j=32-1;j>=0;j--) {
			pi[1+j] ^= pi[32+j];
			pi[2+j] ^= pi[32+j];
			pi[16+j] ^= pi[32+j];
		}
	}
	for(unsigned i=0;i<64;i+=32) {
		__m256i * pi = x + i;
		for(int j=16-1;j>=0;j--) {
			pi[1+j] ^= pi[16+j];
		}
	}
	for(unsigned i=0;i<64;i+=16) {
		__m256i * pi = x + i;
		for(int j=8-1;j>=0;j--) {
			pi[1+j] ^= pi[8+j];
			pi[2+j] ^= pi[8+j];
			pi[4+j] ^= pi[8+j];
		}
	}
	for(unsigned i=0;i<64;i+=8) {
		__m256i * pi = x + i;
		pi[4] ^= pi[7];
		pi[3] ^= pi[6];
		pi[2] ^= pi[5];
		pi[1] ^= pi[4];
	}

	for(unsigned i=0;i<64;i+=4) {
		__m256i * pi = x + i;
		pi[2] ^= pi[3];
		pi[1] ^= pi[2];
	}

}



void encode_half_inp( uint64_t * rfx , const uint64_t * fx , unsigned n_fx_128b )
{
	if(128*2 > n_fx_128b) { printf("unsupported number of terms.\n"); exit(-1); }

	__m256i temp[128];
	__m128i * temp128 = (__m128i*) temp;
	uint64_t * temp64 = (uint64_t *)temp;
	const __m256i * fx_256 = (const __m256i*) fx;
	__m128i * rfx_128 = (__m128i*) rfx;
	unsigned n_fx_256b = n_fx_128b/2;
	unsigned num = n_fx_256b/128;

	for(unsigned i=0;i < num; i ++ ){
		for(unsigned j=0;j<64;j++) {
			temp[j] = fx_256[i + j*num];
			temp[j] = div_s7( temp[j] );
		}

		tr_bit_64x64_b4_avx2( (uint8_t*)(temp128) , (const uint8_t *)temp );
		bit_bc_64x2_div( temp );
		// truncated FFT
		for(unsigned j=0;j<64;j++) bitmatrix_prod_64x128_8R_sse( (uint8_t*)(rfx_128 + i*256+j) , beta_mul_80_m8r , temp64[j*4] );
		for(unsigned j=0;j<64;j++) bitmatrix_prod_64x128_8R_sse( (uint8_t*)(rfx_128 + i*256+64+j) , beta_mul_80_m8r , temp64[j*4+1] );
		for(unsigned j=0;j<64;j++) bitmatrix_prod_64x128_8R_sse( (uint8_t*)(rfx_128 + i*256+128+j) , beta_mul_80_m8r , temp64[j*4+2] );
		for(unsigned j=0;j<64;j++) bitmatrix_prod_64x128_8R_sse( (uint8_t*)(rfx_128 + i*256+128+64+j) , beta_mul_80_m8r , temp64[j*4+3] );
	}
}


static inline
void bit_bc_div( __m256i * x )
{
	for(int i=64-1;i>=0;i--) {
		x[1+i] ^= x[64+i];
		x[4+i] ^= x[64+i];
		x[16+i] ^= x[64+i];
	}
	for(unsigned i=0;i<128;i+=64) {
		__m256i * pi = x + i;
		for(int j=32-1;j>=0;j--) {
			pi[1+j] ^= pi[32+j];
			pi[2+j] ^= pi[32+j];
			pi[16+j] ^= pi[32+j];
		}
	}
	for(unsigned i=0;i<128;i+=32) {
		__m256i * pi = x + i;
		for(int j=16-1;j>=0;j--) {
			pi[1+j] ^= pi[16+j];
		}
	}
	for(unsigned i=0;i<128;i+=16) {
		__m256i * pi = x + i;
		for(int j=8-1;j>=0;j--) {
			pi[1+j] ^= pi[8+j];
			pi[2+j] ^= pi[8+j];
			pi[4+j] ^= pi[8+j];
		}
	}
	for(unsigned i=0;i<128;i+=8) {
		__m256i * pi = x + i;
		pi[4] ^= pi[7];
		pi[3] ^= pi[6];
		pi[2] ^= pi[5];
		pi[1] ^= pi[4];
	}

	for(unsigned i=0;i<128;i+=4) {
		__m256i * pi = x + i;
		pi[2] ^= pi[3];
		pi[1] ^= pi[2];
	}

}

void encode( uint64_t * rfx , const uint64_t * fx , unsigned n_fx_128b )
{
	if(128*2 > n_fx_128b) { printf("unsupported number of terms.\n"); exit(-1); }

	__m256i temp[128];
	__m128i * temp128 = (__m128i*) temp;
	const __m256i * fx_256 = (const __m256i*) fx;
	__m128i * rfx_128 = (__m128i*) rfx;
	unsigned n_fx_256b = n_fx_128b/2;
	unsigned num = n_fx_256b/128;

	for(unsigned i=0;i < num; i ++ ){
		for(unsigned j=0;j<128;j++) {
			temp[j] = fx_256[i + j*num];
			temp[j] = div_s7( temp[j] );
		}

		tr_bit_128x128_b2_avx2( (uint8_t*)(temp128) , (const uint8_t *)temp );
		bit_bc_div( temp );
		// truncated FFT
		for(unsigned j=0;j<128;j++) bitmatrix_prod_128x128_4R_sse( (uint8_t*)(rfx_128 + i*256+j) , beta_mul_80_m4r , (const uint8_t*)(temp128 + j*2) );
		for(unsigned j=0;j<128;j++) bitmatrix_prod_128x128_4R_sse( (uint8_t*)(rfx_128 + i*256+128+j) , beta_mul_80_m4r , (const uint8_t*)(temp128 + j*2+1) );
	}
}



static inline
void bit_bc_exp( __m256i * x )
{
	for(unsigned i=0;i<128;i+=4) {
		__m256i * pi = x + i;
		pi[1] ^= pi[2];
		pi[2] ^= pi[3];
	}
	for(unsigned i=0;i<128;i+=8) {
		__m256i * pi = x + i;
		pi[1] ^= pi[4];
		pi[2] ^= pi[5];
		pi[3] ^= pi[6];
		pi[4] ^= pi[7];
	}
	for(unsigned i=0;i<128;i+=16) {
		__m256i * pi = x + i;
		for(unsigned j=0;j<8;j++) {
			pi[1+j] ^= pi[8+j];
			pi[2+j] ^= pi[8+j];
			pi[4+j] ^= pi[8+j];
		}
	}
	for(unsigned i=0;i<128;i+=32) {
		__m256i * pi = x + i;
		for(unsigned j=0;j<16;j++) {
			pi[1+j] ^= pi[16+j];
		}
	}
	for(unsigned i=0;i<128;i+=64) {
		__m256i * pi = x + i;
		for(unsigned j=0;j<32;j++) {
			pi[1+j] ^= pi[32+j];
			pi[2+j] ^= pi[32+j];
			pi[16+j] ^= pi[32+j];
		}
	}
	for(unsigned i=0;i<64;i++) {
		x[1+i] ^= x[64+i];
		x[4+i] ^= x[64+i];
		x[16+i] ^= x[64+i];
	}
}


void decode( uint64_t * rfx , const uint64_t * fx , unsigned n_fx_128b )
{
	if(128*2 > n_fx_128b) { printf("unsupported number of terms.\n"); exit(-1); }

	const __m128i * fx_128 = (__m128i*) fx;
	__m256i * rfx_256 = (__m256i*) rfx;
	unsigned n_fx_256b = n_fx_128b/2;
	unsigned num = n_fx_256b/128;
	__m256i temp[128];
	__m128i * temp128 = (__m128i*) temp;

	for(unsigned i=0;i < num; i ++ ){
		/// truncated iFFT here.
		for(unsigned j=0;j<128;j++) bitmatrix_prod_128x128_8R_sse( (uint8_t*)(temp128 + j*2) , i_beta_mul_80_m8r , (const uint8_t*)(fx_128 + i*256+j) );
		for(unsigned j=0;j<128;j++) bitmatrix_prod_128x128_8R_sse( (uint8_t*)(temp128 + j*2+1) , i_beta_mul_80_m8r , (const uint8_t*)(fx_128 + i*256+128+j) );

		bit_bc_exp( temp );

		tr_bit_128x128_b2_avx2( (uint8_t*)temp , (const uint8_t *)temp128 );

		for(unsigned j=0;j<128;j++) {
			temp[j] = exp_s7( temp[j] );
			rfx_256[i+j*num] = temp[j];
		}
	}
}

