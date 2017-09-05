/*
Copyright (C) 2007 Ming-Shing Chen

This file is part of BitPolyMul.

BitPolyMul is free software: you can redistribute it and/or modify
it under the terms of the Lesser GNU General Public License as published by
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

#include "gftower.h"

#include "gf2128_tower_iso.h"

#include "bitmat_prod.h"

#include "ska.h"

#include "byte_inline_func.h"

#include "config_profile.h"

#include "string.h"


/////////////////////////////////////////////////
///
/// pclmulqdq version
///
//////////////////////////////////////////////////////

#include "gf2128_cantor_iso.h"

//#define _SIMPLE_TOWER_

static
void butterfly_0( __m128i * poly , unsigned unit )
{
	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
	}
}


static
void butterfly( __m128i * poly , unsigned unit , unsigned ska )
{
	uint8_t ska_iso[16] __attribute__((aligned(32)));
#ifdef _SIMPLE_TOWER_
	bitmatrix_prod_64x128_4R_sse( ska_iso , gfTowerto2128_4R , ska );
#else
	bitmatrix_prod_64x128_4R_sse( ska_iso , gfCantorto2128_4R , ska );
#endif
	//__m128i a = _mm_load_si128( (__m128i*) ska_iso );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		__m128i r;
		gf2ext128_mul_sse( (uint8_t*)&r , (uint8_t*)&poly[unit_2+i] , ska_iso );
		poly[i] ^= r;
		poly[unit_2+i] ^= poly[i];
	}

}


static
void i_butterfly( __m128i * poly , unsigned unit , unsigned ska )
{
	uint8_t ska_iso[16] __attribute__((aligned(32)));
#ifdef _SIMPLE_TOWER_
	bitmatrix_prod_64x128_4R_sse( ska_iso , gfTowerto2128_4R , ska );
#else
	bitmatrix_prod_64x128_4R_sse( ska_iso , gfCantorto2128_4R , ska );
#endif
	//__m128i a = _mm_load_si128( (__m128i*) ska_iso );

	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		__m128i r;
		gf2ext128_mul_sse( (uint8_t*)&r , (uint8_t*)&poly[unit_2+i] , ska_iso );
		poly[i] ^= r;
	}

}


/////////////////////////////////////////////////////



void butterfly_net_half_inp_clmul( uint64_t * fx , unsigned n_fx )
{
	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );

	unsigned n_terms = n_fx;

	__m128i * poly = (__m128i*) &fx[0];

	/// first layer
	memcpy( poly + (n_terms/2) , poly , 8*n_terms );

	for(unsigned i=log_n-1; i>0 ; i--) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		butterfly_0( poly , unit );
		for(unsigned j=1;j<num;j++) {
#ifdef _SIMPLE_TOWER_
			butterfly( poly + j*unit , unit , get_s_k_a( i-1 , j ) );
#else
			butterfly( poly + j*unit , unit , get_s_k_a_cantor( i-1 , j*unit ) );
#endif
		}
	}
}



void i_butterfly_net_clmul( uint64_t * fx , unsigned n_fx )
{
	if( 1 >= n_fx ) return;

	unsigned log_n = __builtin_ctz( n_fx );

	__m128i *poly = (__m128i*) &fx[0];
	unsigned n_terms = n_fx;

	for(unsigned i=1; i <= log_n; i++) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		butterfly_0( poly , unit );
		for(unsigned j=1;j<num;j++) {
#ifdef _SIMPLE_TOWER_
			i_butterfly( poly + j*unit , unit , get_s_k_a( i-1 , j ) );
#else
			i_butterfly( poly + j*unit , unit , get_s_k_a_cantor( i-1 , j*unit ) );
#endif
		}
	}
}









////////////////////////////////////////////////////
///
/// vpshufb version
///
/////////////////////////////////////////////////

#define _SHUFFLE_BYTE_AVX2_

#ifdef _SHUFFLE_BYTE_AVX2_

#include "gf16_tabs.h"
#include <emmintrin.h>  /// avx2



static
void bs_butterfly_0_avx2( __m128i * poly128 , unsigned unit128 )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;
	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
	}
}

static
void bs_butterfly_gf256_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );

	for(unsigned i=0;i<unit_2;i++) {
		poly[i] ^= bs_gf256_mul_avx2( poly[unit_2+i] , multab_0 , ml );
		poly[unit_2+i] ^= poly[i];
	}
}

static
void bs_butterfly_gf65536_12bit_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	__m256i multab0 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(ska&0xff)) );
	__m256i multab_0l = _mm256_permute2x128_si256( multab0 , multab0 , 0 );
	__m256i multab_0h = _mm256_permute2x128_si256( multab0 , multab0 , 0x11 );
	__m256i multab1 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(ska>>8)) );
	__m256i multab_1l = _mm256_permute2x128_si256( multab1 , multab1 , 0 );
	__m256i multab_1h = _mm256_permute2x128_si256( multab1 , multab1 , 0x11 );
	__m256i multab_01l = multab_0l^multab_1l;
	__m256i multab_01h = multab_0h^multab_1h;
	__m256i multab80 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(0x80)) );
	__m256i multab_80l = _mm256_permute2x128_si256( multab80 , multab80 , 0 );
	__m256i multab_80h = _mm256_permute2x128_si256( multab80 , multab80 , 0x11 );

	for(unsigned i=0;i<unit_2;i+=2) {
#if 1
		__m256i src_0l = poly[unit_2+i] & ml;
		__m256i src_0h = _mm256_srli_epi16(poly[unit_2+i],4) & ml;
		__m256i src_1l = poly[unit_2+i+1] & ml;
		__m256i src_1h = _mm256_srli_epi16(poly[unit_2+i+1],4) & ml;

		__m256i ab0 = _mm256_shuffle_epi8(multab_0l,src_0l)^_mm256_shuffle_epi8(multab_0h,src_0h);
		__m256i ab2_l = _mm256_shuffle_epi8(multab_1l,src_1l);
		__m256i ab2_h = _mm256_shuffle_epi8(multab_1l,src_1h);
		__m256i ab1 = _mm256_shuffle_epi8(multab_01l,src_1l^src_0l)^_mm256_shuffle_epi8(multab_01h,src_1h^src_0h)^ab0;
		__m256i ab2r = _mm256_shuffle_epi8(multab_80l,ab2_l)^_mm256_shuffle_epi8(multab_80h,ab2_h);

		poly[i] ^= ab0^ab2r;
		poly[i+1] ^= ab1;
		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];
#else
		struct ymm_x2 ab = bs_gf65536_12bit_mul_avx2( poly[unit_2+i] , poly[unit_2+i+1] , multab_0l , multab_0h , multab_1l , multab_01l , multab_01h ,
			multab_80l , multab_80h , ml );

		poly[i] ^= ab.ymm0;
		poly[i+1] ^= ab.ymm1;
		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];
#endif
	}
}

static
void bs_butterfly_gf65536_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );
	struct ymm_x2 multab_1 = get_multab_avx2( ska>>8 );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );
	struct ymm_x2 multab_01;
	multab_01.ymm0 = multab_0.ymm0 ^ multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1 ^ multab_1.ymm1;

	for(unsigned i=0;i<unit_2;i+=2) {

		struct ymm_x2 ab = bs_gf65536_mul_avx2( poly[unit_2+i] , poly[unit_2+i+1] , multab_0 , multab_1 , multab_01 , multab_80 , ml );

		poly[i] ^= ab.ymm0;
		poly[i+1] ^= ab.ymm1;
		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];

	}
}


static
void bs_butterfly_gf232_20bit_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	__m256i multab0 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(ska&0xff)) );
	__m256i multab_0l = _mm256_permute2x128_si256( multab0 , multab0 , 0 );
	__m256i multab_0h = _mm256_permute2x128_si256( multab0 , multab0 , 0x11 );
	__m256i multab1 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*((ska>>8)&0xff)) );
	__m256i multab_1l = _mm256_permute2x128_si256( multab1 , multab1 , 0 );
	__m256i multab_1h = _mm256_permute2x128_si256( multab1 , multab1 , 0x11 );
	__m256i multab80 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(0x80)) );
	__m256i multab_80l = _mm256_permute2x128_si256( multab80 , multab80 , 0 );
	__m256i multab_80h = _mm256_permute2x128_si256( multab80 , multab80 , 0x11 );

	__m256i multab2 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(ska>>16)) );
	__m256i multab_2l = _mm256_permute2x128_si256( multab2 , multab2 , 0 );

	for(unsigned i=0;i<unit_2;i+=4) {
		struct ymm_x4 ab = bs_gf232_20bit_mul_avx2( poly[unit_2+i] , poly[unit_2+i+1] , poly[unit_2+i+2]  , poly[unit_2+i+3] ,
			multab_0l , multab_0h , multab_1l , multab_1h , multab_2l ,
			multab_80l , multab_80h , ml );

		poly[i] ^= ab.ymm0;
		poly[i+1] ^= ab.ymm1;
		poly[i+2] ^= ab.ymm2;
		poly[i+3] ^= ab.ymm3;

		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];
		poly[unit_2+i+2] ^= poly[i+2];
		poly[unit_2+i+3] ^= poly[i+3];
	}
}


static
void bs_butterfly_gf232_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );
	struct ymm_x2 multab_1 = get_multab_avx2( ska>>8 );
	struct ymm_x2 multab_2 = get_multab_avx2( ska>>16 );
	struct ymm_x2 multab_3 = get_multab_avx2( ska>>24 );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );

	for(unsigned i=0;i<unit_2;i+=4) {
		struct ymm_x4 ab = bs_gf232_mul_avx2( *(struct ymm_x4 *)(&poly[unit_2+i]) , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );

		poly[i] ^= ab.ymm0;
		poly[i+1] ^= ab.ymm1;
		poly[i+2] ^= ab.ymm2;
		poly[i+3] ^= ab.ymm3;

		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];
		poly[unit_2+i+2] ^= poly[i+2];
		poly[unit_2+i+3] ^= poly[i+3];
	}
}


static
void bs_butterfly_avx2( __m128i * poly , unsigned unit , unsigned ska )
{
	//if( 65536*16 <= ska ) { printf("ska out of range.\n"); exit(-1); }
	if( 256 > ska ) bs_butterfly_gf256_avx2( poly , unit , ska );
	else if( 4096 > ska ) bs_butterfly_gf65536_12bit_avx2( poly , unit , ska );
	else if( 65536 > ska ) bs_butterfly_gf65536_avx2( poly , unit , ska );
	else if(65536*16 > ska ) bs_butterfly_gf232_20bit_avx2( poly , unit , ska );
	else bs_butterfly_gf232_avx2( poly , unit , ska );
}

///////////////////////////////


static
void bs_butterfly_gf256_l1_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 , unsigned ska2 , unsigned ska3 )
{
	__m128i * poly = (__m128i *) poly128;
	//unsigned unit = 2;
	//unsigned unit_2 = 1;

	__m128i ml = _mm_load_si128( (__m128i*) __mask_low );
	__m128i m32_16 = _mm_load_si128( (__m128i*) __mask_32bit_low );
	__m128i m16_8 = _mm_load_si128( (__m128i*) __mask_16bit_low );

	struct xmm_x2 multab_0 = get_multab_sse( ska0 );
	__m128i ab_xmm0 = bs_gf256_mul_sse( poly[1] , multab_0 , ml );
	__m128i ab_xmm1 = bs_gf256_mul_sse( poly[3] , multab_0 , ml );
	__m128i ab_xmm2 = bs_gf256_mul_sse( poly[5] , multab_0 , ml );
	__m128i ab_xmm3 = bs_gf256_mul_sse( poly[7] , multab_0 , ml );
	poly[0] ^= ab_xmm0 & m32_16 & m16_8;
	poly[2] ^= ab_xmm1 & m32_16 & m16_8;
	poly[4] ^= ab_xmm2 & m32_16 & m16_8;
	poly[6] ^= ab_xmm3 & m32_16 & m16_8;

	multab_0 = get_multab_sse( ska1 );
	ab_xmm0 = bs_gf256_mul_sse( poly[1] , multab_0 , ml );
	ab_xmm1 = bs_gf256_mul_sse( poly[3] , multab_0 , ml );
	ab_xmm2 = bs_gf256_mul_sse( poly[5] , multab_0 , ml );
	ab_xmm3 = bs_gf256_mul_sse( poly[7] , multab_0 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , ab_xmm0 & m32_16 );
	poly[2] ^= _mm_andnot_si128( m16_8 , ab_xmm1 & m32_16 );
	poly[4] ^= _mm_andnot_si128( m16_8 , ab_xmm2 & m32_16 );
	poly[6] ^= _mm_andnot_si128( m16_8 , ab_xmm3 & m32_16 );

	multab_0 = get_multab_sse( ska2 );
	ab_xmm0 = bs_gf256_mul_sse( poly[1] , multab_0 , ml );
	ab_xmm1 = bs_gf256_mul_sse( poly[3] , multab_0 , ml );
	ab_xmm2 = bs_gf256_mul_sse( poly[5] , multab_0 , ml );
	ab_xmm3 = bs_gf256_mul_sse( poly[7] , multab_0 , ml );
	poly[0] ^= _mm_andnot_si128( m32_16 , ab_xmm0 & m16_8 );
	poly[2] ^= _mm_andnot_si128( m32_16 , ab_xmm1 & m16_8 );
	poly[4] ^= _mm_andnot_si128( m32_16 , ab_xmm2 & m16_8 );
	poly[6] ^= _mm_andnot_si128( m32_16 , ab_xmm3 & m16_8 );

	multab_0 = get_multab_sse( ska3 );
	ab_xmm0 = bs_gf256_mul_sse( poly[1] , multab_0 , ml );
	ab_xmm1 = bs_gf256_mul_sse( poly[3] , multab_0 , ml );
	ab_xmm2 = bs_gf256_mul_sse( poly[5] , multab_0 , ml );
	ab_xmm3 = bs_gf256_mul_sse( poly[7] , multab_0 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab_xmm0 ) );
	poly[2] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab_xmm1 ) );
	poly[4] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab_xmm2 ) );
	poly[6] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab_xmm3 ) );

	poly[1] ^= poly[0];
	poly[3] ^= poly[2];
	poly[5] ^= poly[4];
	poly[7] ^= poly[6];
}

static
void bs_butterfly_gf65536_l1_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 , unsigned ska2 , unsigned ska3 )
{
	__m128i * poly = (__m128i *) poly128;
	//unsigned unit = 2;
	//unsigned unit_2 = 1;

	__m128i ml = _mm_load_si128( (__m128i*) __mask_low );
	__m128i m32_16 = _mm_load_si128( (__m128i*) __mask_32bit_low );
	__m128i m16_8 = _mm_load_si128( (__m128i*) __mask_16bit_low );

	struct xmm_x2 multab_80 = get_multab_sse( 0x80 );

	struct xmm_x2 multab_0 = get_multab_sse( ska0 );
	struct xmm_x2 multab_1 = get_multab_sse( ska0>>8 );
	struct xmm_x2 multab_01;
	multab_01.xmm0 = multab_0.xmm0^multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1^multab_1.xmm1;
	struct xmm_x2 ab0 = bs_gf65536_mul_sse( poly[1] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	struct xmm_x2 ab1 = bs_gf65536_mul_sse( poly[5] , poly[7] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= ab0.xmm0 & m32_16 & m16_8;
	poly[2] ^= ab0.xmm1 & m32_16 & m16_8;
	poly[4] ^= ab1.xmm0 & m32_16 & m16_8;
	poly[6] ^= ab1.xmm1 & m32_16 & m16_8;

	multab_0 = get_multab_sse( ska1 );
	multab_1 = get_multab_sse( ska1>>8 );
	multab_01.xmm0 = multab_0.xmm0^multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1^multab_1.xmm1;
	ab0 = bs_gf65536_mul_sse( poly[1] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_sse( poly[5] , poly[7] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , ab0.xmm0 & m32_16 );
	poly[2] ^= _mm_andnot_si128( m16_8 , ab0.xmm1 & m32_16 );
	poly[4] ^= _mm_andnot_si128( m16_8 , ab1.xmm0 & m32_16 );
	poly[6] ^= _mm_andnot_si128( m16_8 , ab1.xmm1 & m32_16 );

	multab_0 = get_multab_sse( ska2 );
	multab_1 = get_multab_sse( ska2>>8 );
	multab_01.xmm0 = multab_0.xmm0^multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1^multab_1.xmm1;
	ab0 = bs_gf65536_mul_sse( poly[1] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_sse( poly[5] , poly[7] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m32_16 , ab0.xmm0 & m16_8 );
	poly[2] ^= _mm_andnot_si128( m32_16 , ab0.xmm1 & m16_8 );
	poly[4] ^= _mm_andnot_si128( m32_16 , ab1.xmm0 & m16_8 );
	poly[6] ^= _mm_andnot_si128( m32_16 , ab1.xmm1 & m16_8 );

	multab_0 = get_multab_sse( ska3 );
	multab_1 = get_multab_sse( ska3>>8 );
	multab_01.xmm0 = multab_0.xmm0^multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1^multab_1.xmm1;
	ab0 = bs_gf65536_mul_sse( poly[1] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_sse( poly[5] , poly[7] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab0.xmm0 ) );
	poly[2] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab0.xmm1 ) );
	poly[4] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab1.xmm0 ) );
	poly[6] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab1.xmm1 ) );

	poly[1] ^= poly[0];
	poly[3] ^= poly[2];
	poly[5] ^= poly[4];
	poly[7] ^= poly[6];
}

static
void bs_butterfly_gf232_l1_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 , unsigned ska2 , unsigned ska3 )
{
	__m128i * poly = (__m128i *) poly128;
	//unsigned unit = 2;
	//unsigned unit_2 = 1;

	__m128i ml = _mm_load_si128( (__m128i*) __mask_low );
	__m128i m32_16 = _mm_load_si128( (__m128i*) __mask_32bit_low );
	__m128i m16_8 = _mm_load_si128( (__m128i*) __mask_16bit_low );

	struct xmm_x2 multab_80 = get_multab_sse( 0x80 );

	struct xmm_x4 src;
	src.xmm0 = poly[1];
	src.xmm1 = poly[3];
	src.xmm2 = poly[5];
	src.xmm3 = poly[7];

	struct xmm_x2 multab_0 = get_multab_sse( ska0 );
	struct xmm_x2 multab_1 = get_multab_sse( ska0>>8 );
	struct xmm_x2 multab_2 = get_multab_sse( ska0>>16 );
	struct xmm_x2 multab_3 = get_multab_sse( ska0>>24 );
	struct xmm_x4 ab = bs_gf232_mul_sse( src , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	poly[0] ^= ab.xmm0 & m32_16 & m16_8;
	poly[2] ^= ab.xmm1 & m32_16 & m16_8;
	poly[4] ^= ab.xmm2 & m32_16 & m16_8;
	poly[6] ^= ab.xmm3 & m32_16 & m16_8;

	multab_0 = get_multab_sse( ska1 );
	multab_1 = get_multab_sse( ska1>>8 );
	multab_2 = get_multab_sse( ska1>>16 );
	multab_3 = get_multab_sse( ska1>>24 );
	ab = bs_gf232_mul_sse( src , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , ab.xmm0 & m32_16 );
	poly[2] ^= _mm_andnot_si128( m16_8 , ab.xmm1 & m32_16 );
	poly[4] ^= _mm_andnot_si128( m16_8 , ab.xmm2 & m32_16 );
	poly[6] ^= _mm_andnot_si128( m16_8 , ab.xmm3 & m32_16 );

	multab_0 = get_multab_sse( ska2 );
	multab_1 = get_multab_sse( ska2>>8 );
	multab_2 = get_multab_sse( ska2>>16 );
	multab_3 = get_multab_sse( ska2>>24 );
	ab = bs_gf232_mul_sse( src , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m32_16 , ab.xmm0 & m16_8 );
	poly[2] ^= _mm_andnot_si128( m32_16 , ab.xmm1 & m16_8 );
	poly[4] ^= _mm_andnot_si128( m32_16 , ab.xmm2 & m16_8 );
	poly[6] ^= _mm_andnot_si128( m32_16 , ab.xmm3 & m16_8 );

	multab_0 = get_multab_sse( ska3 );
	multab_1 = get_multab_sse( ska3>>8 );
	multab_2 = get_multab_sse( ska3>>16 );
	multab_3 = get_multab_sse( ska3>>24 );
	ab = bs_gf232_mul_sse( src , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab.xmm0 ) );
	poly[2] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab.xmm1 ) );
	poly[4] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab.xmm2 ) );
	poly[6] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab.xmm3 ) );

	poly[1] ^= poly[0];
	poly[3] ^= poly[2];
	poly[5] ^= poly[4];
	poly[7] ^= poly[6];
}


static
void bs_butterfly_l1_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 , unsigned ska2 , unsigned ska3 )
{
	if( ska3 < 256 ) bs_butterfly_gf256_l1_avx2( poly128 , ska0 , ska1 , ska2 , ska3 );
	else if( ska3 < 65536 ) bs_butterfly_gf65536_l1_avx2( poly128 , ska0 , ska1 , ska2 , ska3 );
	else bs_butterfly_gf232_l1_avx2( poly128 , ska0 , ska1 , ska2 , ska3 );
}

////////////////////////////////////////////

static
void bs_butterfly_gf256_l2_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 4;
	//unsigned unit_2 = 2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	__m256i m32_16 = _mm256_load_si256( (__m256i*) __mask_32bit_low );

	struct ymm_x2 multab_0 = get_multab_avx2( ska0 );
	__m256i ab_ymm0 = bs_gf256_mul_avx2( poly[0] , multab_0 , ml );
	__m256i ab_ymm1 = bs_gf256_mul_avx2( poly[1] , multab_0 , ml );
	__m256i ab_ymm2 = bs_gf256_mul_avx2( poly[2] , multab_0 , ml );
	__m256i ab_ymm3 = bs_gf256_mul_avx2( poly[3] , multab_0 , ml );
	poly[0] ^= _mm256_srli_epi16( ab_ymm0 & m32_16 , 8 );
	poly[1] ^= _mm256_srli_epi16( ab_ymm1 & m32_16 , 8 );
	poly[2] ^= _mm256_srli_epi16( ab_ymm2 & m32_16 , 8 );
	poly[3] ^= _mm256_srli_epi16( ab_ymm3 & m32_16 , 8 );

	multab_0 = get_multab_avx2( ska1 );
	ab_ymm0 = bs_gf256_mul_avx2( poly[0] , multab_0 , ml );
	ab_ymm1 = bs_gf256_mul_avx2( poly[1] , multab_0 , ml );
	ab_ymm2 = bs_gf256_mul_avx2( poly[2] , multab_0 , ml );
	ab_ymm3 = bs_gf256_mul_avx2( poly[3] , multab_0 , ml );
	poly[0] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab_ymm0) , 8 );
	poly[1] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab_ymm1) , 8 );
	poly[2] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab_ymm2) , 8 );
	poly[3] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab_ymm3) , 8 );

	poly[0] ^= _mm256_slli_epi16( poly[0] , 8 );
	poly[1] ^= _mm256_slli_epi16( poly[1] , 8 );
	poly[2] ^= _mm256_slli_epi16( poly[2] , 8 );
	poly[3] ^= _mm256_slli_epi16( poly[3] , 8 );
}


static
void bs_butterfly_gf65536_l2_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 4;
	//unsigned unit_2 = 2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );

	__m256i m32_16 = _mm256_load_si256( (__m256i*) __mask_32bit_low );

	struct ymm_x2 multab_0 = get_multab_avx2( ska0 );
	struct ymm_x2 multab_1 = get_multab_avx2( ska0>>8 );
	struct ymm_x2 ab0,ab1,multab_01;

	multab_01.ymm0 = multab_0.ymm0^multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1^multab_1.ymm1;
	ab0 = bs_gf65536_mul_avx2( poly[0] , poly[1] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_avx2( poly[2] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm256_srli_epi16( ab0.ymm0 & m32_16 , 8 );
	poly[1] ^= _mm256_srli_epi16( ab0.ymm1 & m32_16 , 8 );
	poly[2] ^= _mm256_srli_epi16( ab1.ymm0 & m32_16 , 8 );
	poly[3] ^= _mm256_srli_epi16( ab1.ymm1 & m32_16 , 8 );

	multab_0 = get_multab_avx2( ska1 );
	multab_1 = get_multab_avx2( ska1>>8 );
	multab_01.ymm0 = multab_0.ymm0^multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1^multab_1.ymm1;
	ab0 = bs_gf65536_mul_avx2( poly[0] , poly[1] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_avx2( poly[2] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab0.ymm0) , 8 );
	poly[1] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab0.ymm1) , 8 );
	poly[2] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab1.ymm0) , 8 );
	poly[3] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab1.ymm1) , 8 );

	poly[0] ^= _mm256_slli_epi16( poly[0] , 8 );
	poly[1] ^= _mm256_slli_epi16( poly[1] , 8 );
	poly[2] ^= _mm256_slli_epi16( poly[2] , 8 );
	poly[3] ^= _mm256_slli_epi16( poly[3] , 8 );
}

static
void bs_butterfly_gf232_l2_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 4;
	//unsigned unit_2 = 2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );

	__m256i m32_16 = _mm256_load_si256( (__m256i*) __mask_32bit_low );

	struct ymm_x2 multab_0 = get_multab_avx2( ska0 );
	struct ymm_x2 multab_1 = get_multab_avx2( ska0>>8 );
	struct ymm_x2 multab_2 = get_multab_avx2( ska0>>16 );
	struct ymm_x2 multab_3 = get_multab_avx2( ska0>>24 );

	struct ymm_x4 ab = bs_gf232_mul_avx2( *(struct ymm_x4 *)poly , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );

	poly[0] ^= _mm256_srli_epi16( ab.ymm0 & m32_16 , 8 );
	poly[1] ^= _mm256_srli_epi16( ab.ymm1 & m32_16 , 8 );
	poly[2] ^= _mm256_srli_epi16( ab.ymm2 & m32_16 , 8 );
	poly[3] ^= _mm256_srli_epi16( ab.ymm3 & m32_16 , 8 );

	multab_0 = get_multab_avx2( ska1 );
	multab_1 = get_multab_avx2( ska1>>8 );
	multab_2 = get_multab_avx2( ska1>>16 );
	multab_3 = get_multab_avx2( ska1>>24 );

	ab = bs_gf232_mul_avx2( *(struct ymm_x4 *)poly , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );

	poly[0] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab.ymm0) , 8 );
	poly[1] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab.ymm1) , 8 );
	poly[2] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab.ymm2) , 8 );
	poly[3] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab.ymm3) , 8 );

	poly[0] ^= _mm256_slli_epi16( poly[0] , 8 );
	poly[1] ^= _mm256_slli_epi16( poly[1] , 8 );
	poly[2] ^= _mm256_slli_epi16( poly[2] , 8 );
	poly[3] ^= _mm256_slli_epi16( poly[3] , 8 );
}

static
void bs_butterfly_l2_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 )
{
	if( ska1 < 256 ) bs_butterfly_gf256_l2_avx2( poly128 , ska0 , ska1 );
	else if( ska1 < 65536 ) bs_butterfly_gf65536_l2_avx2( poly128 , ska0 , ska1 );
	else bs_butterfly_gf232_l2_avx2( poly128 , ska0 , ska1 );
}

/////////////////////////////////////////////////////

static
void bs_butterfly_gf256_l3_avx2( __m128i * poly128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 8;
	//unsigned unit_2 = 4;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );

	__m256i ab_ymm0 = bs_gf256_mul_avx2( poly[0] , multab_0 , ml );
	__m256i ab_ymm1 = bs_gf256_mul_avx2( poly[1] , multab_0 , ml );
	__m256i ab_ymm2 = bs_gf256_mul_avx2( poly[2] , multab_0 , ml );
	__m256i ab_ymm3 = bs_gf256_mul_avx2( poly[3] , multab_0 , ml );

	poly[0] ^= _mm256_srli_epi32( ab_ymm0 , 16 );
	poly[1] ^= _mm256_srli_epi32( ab_ymm1 , 16 );
	poly[2] ^= _mm256_srli_epi32( ab_ymm2 , 16 );
	poly[3] ^= _mm256_srli_epi32( ab_ymm3 , 16 );

	poly[0] ^= _mm256_slli_epi32( poly[0] , 16 );
	poly[1] ^= _mm256_slli_epi32( poly[1] , 16 );
	poly[2] ^= _mm256_slli_epi32( poly[2] , 16 );
	poly[3] ^= _mm256_slli_epi32( poly[3] , 16 );
}

static
void bs_butterfly_gf65536_l3_avx2( __m128i * poly128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 8;
	//unsigned unit_2 = 4;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );
	struct ymm_x2 multab_1 = get_multab_avx2( ska>>8 );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );
	struct ymm_x2 multab_01;
	multab_01.ymm0 = multab_0.ymm0^multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1^multab_1.ymm1;

	struct ymm_x2 ab0 = bs_gf65536_mul_avx2( poly[0] , poly[1] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	struct ymm_x2 ab1 = bs_gf65536_mul_avx2( poly[2] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );

	poly[0] ^= _mm256_srli_epi32( ab0.ymm0 , 16 );
	poly[1] ^= _mm256_srli_epi32( ab0.ymm1 , 16 );
	poly[2] ^= _mm256_srli_epi32( ab1.ymm0 , 16 );
	poly[3] ^= _mm256_srli_epi32( ab1.ymm1 , 16 );

	poly[0] ^= _mm256_slli_epi32( poly[0] , 16 );
	poly[1] ^= _mm256_slli_epi32( poly[1] , 16 );
	poly[2] ^= _mm256_slli_epi32( poly[2] , 16 );
	poly[3] ^= _mm256_slli_epi32( poly[3] , 16 );
}

static
void bs_butterfly_gf232_l3_avx2( __m128i * poly128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 8;
	//unsigned unit_2 = 4;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );
	struct ymm_x2 multab_1 = get_multab_avx2( ska>>8 );
	struct ymm_x2 multab_2 = get_multab_avx2( ska>>16 );
	struct ymm_x2 multab_3 = get_multab_avx2( ska>>24 );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );

	struct ymm_x4 ab = bs_gf232_mul_avx2( *(struct ymm_x4 *)poly , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );

	poly[0] ^= _mm256_srli_epi32( ab.ymm0 , 16 );
	poly[1] ^= _mm256_srli_epi32( ab.ymm1 , 16 );
	poly[2] ^= _mm256_srli_epi32( ab.ymm2 , 16 );
	poly[3] ^= _mm256_srli_epi32( ab.ymm3 , 16 );

	poly[0] ^= _mm256_slli_epi32( poly[0] , 16 );
	poly[1] ^= _mm256_slli_epi32( poly[1] , 16 );
	poly[2] ^= _mm256_slli_epi32( poly[2] , 16 );
	poly[3] ^= _mm256_slli_epi32( poly[3] , 16 );
}

static
void bs_butterfly_l3_avx2( __m128i * poly128 , unsigned ska )
{
	if( ska < 256 ) bs_butterfly_gf256_l3_avx2( poly128 , ska );
	else if( ska < 65536 ) bs_butterfly_gf65536_l3_avx2( poly128 , ska );
	else bs_butterfly_gf232_l3_avx2( poly128 , ska );
}

///////////////////////////////////////////////////////////////
///
/// inverse butterfly
///
/////////////////////////////

static
void bs_i_butterfly_gf256_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );

	for(unsigned i=0;i<unit_2;i++) {
		poly[unit_2+i] ^= poly[i];
		poly[i] ^= bs_gf256_mul_avx2( poly[unit_2+i] , multab_0 , ml );
	}
}

static
void bs_i_butterfly_gf65536_12bit_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	__m256i multab0 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(ska&0xff)) );
	__m256i multab_0l = _mm256_permute2x128_si256( multab0 , multab0 , 0 );
	__m256i multab_0h = _mm256_permute2x128_si256( multab0 , multab0 , 0x11 );
	__m256i multab1 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(ska>>8)) );
	__m256i multab_1l = _mm256_permute2x128_si256( multab1 , multab1 , 0 );
	__m256i multab_1h = _mm256_permute2x128_si256( multab1 , multab1 , 0x11 );
	__m256i multab_01l = multab_0l^multab_1l;
	__m256i multab_01h = multab_0h^multab_1h;
	__m256i multab80 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(0x80)) );
	__m256i multab_80l = _mm256_permute2x128_si256( multab80 , multab80 , 0 );
	__m256i multab_80h = _mm256_permute2x128_si256( multab80 , multab80 , 0x11 );

	for(unsigned i=0;i<unit_2;i+=2) {
		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];
#if 1
		__m256i src_0l = poly[unit_2+i] & ml;
		__m256i src_0h = _mm256_srli_epi16(poly[unit_2+i],4) & ml;
		__m256i src_1l = poly[unit_2+i+1] & ml;
		__m256i src_1h = _mm256_srli_epi16(poly[unit_2+i+1],4) & ml;

		__m256i ab0 = _mm256_shuffle_epi8(multab_0l,src_0l)^_mm256_shuffle_epi8(multab_0h,src_0h);
		__m256i ab2_l = _mm256_shuffle_epi8(multab_1l,src_1l);
		__m256i ab2_h = _mm256_shuffle_epi8(multab_1l,src_1h);
		__m256i ab1 = _mm256_shuffle_epi8(multab_01l,src_1l^src_0l)^_mm256_shuffle_epi8(multab_01h,src_1h^src_0h)^ab0;
		__m256i ab2r = _mm256_shuffle_epi8(multab_80l,ab2_l)^_mm256_shuffle_epi8(multab_80h,ab2_h);

		poly[i] ^= ab0^ab2r;
		poly[i+1] ^= ab1;
#else
		struct ymm_x2 ab = bs_gf65536_12bit_mul_avx2( poly[unit_2+i] , poly[unit_2+i+1] , multab_0l , multab_0h , multab_1l , multab_01l , multab_01h ,
			multab_80l , multab_80h , ml );
		poly[i] ^= ab.ymm0;
		poly[i+1] ^= ab.ymm1;
#endif
	}
}

static
void bs_i_butterfly_gf65536_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );
	struct ymm_x2 multab_1 = get_multab_avx2( ska>>8 );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );
	struct ymm_x2 multab_01;
	multab_01.ymm0 = multab_0.ymm0 ^ multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1 ^ multab_1.ymm1;

	for(unsigned i=0;i<unit_2;i+=2) {
		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];

		struct ymm_x2 ab = bs_gf65536_mul_avx2( poly[unit_2+i] , poly[unit_2+i+1] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
		poly[i] ^= ab.ymm0;
		poly[i+1] ^= ab.ymm1;
	}
}


static
void bs_i_butterfly_gf232_20bit_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	__m256i multab0 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(ska&0xff)) );
	__m256i multab_0l = _mm256_permute2x128_si256( multab0 , multab0 , 0 );
	__m256i multab_0h = _mm256_permute2x128_si256( multab0 , multab0 , 0x11 );
	__m256i multab1 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*((ska>>8)&0xff)) );
	__m256i multab_1l = _mm256_permute2x128_si256( multab1 , multab1 , 0 );
	__m256i multab_1h = _mm256_permute2x128_si256( multab1 , multab1 , 0x11 );
	__m256i multab80 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(0x80)) );
	__m256i multab_80l = _mm256_permute2x128_si256( multab80 , multab80 , 0 );
	__m256i multab_80h = _mm256_permute2x128_si256( multab80 , multab80 , 0x11 );

	__m256i multab2 = _mm256_load_si256( (__m256i*) (__gf256_mul+32*(ska>>16)) );
	__m256i multab_2l = _mm256_permute2x128_si256( multab2 , multab2 , 0 );

	for(unsigned i=0;i<unit_2;i+=4) {
		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];
		poly[unit_2+i+2] ^= poly[i+2];
		poly[unit_2+i+3] ^= poly[i+3];

		struct ymm_x4 ab = bs_gf232_20bit_mul_avx2( poly[unit_2+i] , poly[unit_2+i+1] , poly[unit_2+i+2]  , poly[unit_2+i+3] ,
			multab_0l , multab_0h , multab_1l , multab_1h , multab_2l ,
			multab_80l , multab_80h , ml );

		poly[i] ^= ab.ymm0;
		poly[i+1] ^= ab.ymm1;
		poly[i+2] ^= ab.ymm2;
		poly[i+3] ^= ab.ymm3;
	}
}

static
void bs_i_butterfly_gf232_avx2( __m128i * poly128 , unsigned unit128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	unsigned unit= unit128/2;
	unsigned unit_2= unit/2;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );
	struct ymm_x2 multab_1 = get_multab_avx2( ska>>8 );
	struct ymm_x2 multab_2 = get_multab_avx2( ska>>16 );
	struct ymm_x2 multab_3 = get_multab_avx2( ska>>24 );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );

	for(unsigned i=0;i<unit_2;i+=4) {
		poly[unit_2+i] ^= poly[i];
		poly[unit_2+i+1] ^= poly[i+1];
		poly[unit_2+i+2] ^= poly[i+2];
		poly[unit_2+i+3] ^= poly[i+3];

		struct ymm_x4 ab = bs_gf232_mul_avx2( *(struct ymm_x4 *)(&poly[unit_2+i]) , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );

		poly[i] ^= ab.ymm0;
		poly[i+1] ^= ab.ymm1;
		poly[i+2] ^= ab.ymm2;
		poly[i+3] ^= ab.ymm3;
	}
}


static
void bs_i_butterfly_avx2( __m128i * poly , unsigned unit , unsigned ska )
{
	//if( 65536*16 <= ska ) { printf("ska out of range.\n"); exit(-1); }
	if( 256 > ska ) bs_i_butterfly_gf256_avx2( poly , unit , ska );
	else if( 4096 > ska ) bs_i_butterfly_gf65536_12bit_avx2( poly , unit , ska );
	else if( 65536 > ska) bs_i_butterfly_gf65536_avx2( poly , unit , ska );
	else if( 65536*16 > ska ) bs_i_butterfly_gf232_20bit_avx2( poly , unit , ska );
	else bs_i_butterfly_gf232_avx2( poly , unit , ska );
}



//////////////////////////////////////////////

static
void bs_i_butterfly_gf256_l1_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 , unsigned ska2 , unsigned ska3 )
{
	__m128i * poly = (__m128i *) poly128;
	//unsigned unit = 2;
	//unsigned unit_2 = 1;

	__m128i ml = _mm_load_si128( (__m128i*) __mask_low );
	__m128i m32_16 = _mm_load_si128( (__m128i*) __mask_32bit_low );
	__m128i m16_8 = _mm_load_si128( (__m128i*) __mask_16bit_low );

	poly[1] ^= poly[0];
	poly[3] ^= poly[2];
	poly[5] ^= poly[4];
	poly[7] ^= poly[6];

	struct xmm_x2 multab_0 = get_multab_sse( ska0 );
	__m128i ab_xmm0 = bs_gf256_mul_sse( poly[1] , multab_0 , ml );
	__m128i ab_xmm1 = bs_gf256_mul_sse( poly[3] , multab_0 , ml );
	__m128i ab_xmm2 = bs_gf256_mul_sse( poly[5] , multab_0 , ml );
	__m128i ab_xmm3 = bs_gf256_mul_sse( poly[7] , multab_0 , ml );
	poly[0] ^= ab_xmm0 & m32_16 & m16_8;
	poly[2] ^= ab_xmm1 & m32_16 & m16_8;
	poly[4] ^= ab_xmm2 & m32_16 & m16_8;
	poly[6] ^= ab_xmm3 & m32_16 & m16_8;

	multab_0 = get_multab_sse( ska1 );
	ab_xmm0 = bs_gf256_mul_sse( poly[1] , multab_0 , ml );
	ab_xmm1 = bs_gf256_mul_sse( poly[3] , multab_0 , ml );
	ab_xmm2 = bs_gf256_mul_sse( poly[5] , multab_0 , ml );
	ab_xmm3 = bs_gf256_mul_sse( poly[7] , multab_0 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , ab_xmm0 & m32_16 );
	poly[2] ^= _mm_andnot_si128( m16_8 , ab_xmm1 & m32_16 );
	poly[4] ^= _mm_andnot_si128( m16_8 , ab_xmm2 & m32_16 );
	poly[6] ^= _mm_andnot_si128( m16_8 , ab_xmm3 & m32_16 );

	multab_0 = get_multab_sse( ska2 );
	ab_xmm0 = bs_gf256_mul_sse( poly[1] , multab_0 , ml );
	ab_xmm1 = bs_gf256_mul_sse( poly[3] , multab_0 , ml );
	ab_xmm2 = bs_gf256_mul_sse( poly[5] , multab_0 , ml );
	ab_xmm3 = bs_gf256_mul_sse( poly[7] , multab_0 , ml );
	poly[0] ^= _mm_andnot_si128( m32_16 , ab_xmm0 & m16_8 );
	poly[2] ^= _mm_andnot_si128( m32_16 , ab_xmm1 & m16_8 );
	poly[4] ^= _mm_andnot_si128( m32_16 , ab_xmm2 & m16_8 );
	poly[6] ^= _mm_andnot_si128( m32_16 , ab_xmm3 & m16_8 );

	multab_0 = get_multab_sse( ska3 );
	ab_xmm0 = bs_gf256_mul_sse( poly[1] , multab_0 , ml );
	ab_xmm1 = bs_gf256_mul_sse( poly[3] , multab_0 , ml );
	ab_xmm2 = bs_gf256_mul_sse( poly[5] , multab_0 , ml );
	ab_xmm3 = bs_gf256_mul_sse( poly[7] , multab_0 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab_xmm0 ) );
	poly[2] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab_xmm1 ) );
	poly[4] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab_xmm2 ) );
	poly[6] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab_xmm3 ) );

}

static
void bs_i_butterfly_gf65536_l1_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 , unsigned ska2 , unsigned ska3 )
{
	__m128i * poly = (__m128i *) poly128;
	//unsigned unit = 2;
	//unsigned unit_2 = 1;

	__m128i ml = _mm_load_si128( (__m128i*) __mask_low );
	__m128i m32_16 = _mm_load_si128( (__m128i*) __mask_32bit_low );
	__m128i m16_8 = _mm_load_si128( (__m128i*) __mask_16bit_low );

	poly[1] ^= poly[0];
	poly[3] ^= poly[2];
	poly[5] ^= poly[4];
	poly[7] ^= poly[6];

	struct xmm_x2 multab_80 = get_multab_sse( 0x80 );

	struct xmm_x2 multab_0 = get_multab_sse( ska0 );
	struct xmm_x2 multab_1 = get_multab_sse( ska0>>8 );
	struct xmm_x2 multab_01;
	multab_01.xmm0 = multab_0.xmm0^multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1^multab_1.xmm1;
	struct xmm_x2 ab0 = bs_gf65536_mul_sse( poly[1] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	struct xmm_x2 ab1 = bs_gf65536_mul_sse( poly[5] , poly[7] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= ab0.xmm0 & m32_16 & m16_8;
	poly[2] ^= ab0.xmm1 & m32_16 & m16_8;
	poly[4] ^= ab1.xmm0 & m32_16 & m16_8;
	poly[6] ^= ab1.xmm1 & m32_16 & m16_8;

	multab_0 = get_multab_sse( ska1 );
	multab_1 = get_multab_sse( ska1>>8 );
	multab_01.xmm0 = multab_0.xmm0^multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1^multab_1.xmm1;
	ab0 = bs_gf65536_mul_sse( poly[1] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_sse( poly[5] , poly[7] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , ab0.xmm0 & m32_16 );
	poly[2] ^= _mm_andnot_si128( m16_8 , ab0.xmm1 & m32_16 );
	poly[4] ^= _mm_andnot_si128( m16_8 , ab1.xmm0 & m32_16 );
	poly[6] ^= _mm_andnot_si128( m16_8 , ab1.xmm1 & m32_16 );

	multab_0 = get_multab_sse( ska2 );
	multab_1 = get_multab_sse( ska2>>8 );
	multab_01.xmm0 = multab_0.xmm0^multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1^multab_1.xmm1;
	ab0 = bs_gf65536_mul_sse( poly[1] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_sse( poly[5] , poly[7] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m32_16 , ab0.xmm0 & m16_8 );
	poly[2] ^= _mm_andnot_si128( m32_16 , ab0.xmm1 & m16_8 );
	poly[4] ^= _mm_andnot_si128( m32_16 , ab1.xmm0 & m16_8 );
	poly[6] ^= _mm_andnot_si128( m32_16 , ab1.xmm1 & m16_8 );

	multab_0 = get_multab_sse( ska3 );
	multab_1 = get_multab_sse( ska3>>8 );
	multab_01.xmm0 = multab_0.xmm0^multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1^multab_1.xmm1;
	ab0 = bs_gf65536_mul_sse( poly[1] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_sse( poly[5] , poly[7] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab0.xmm0 ) );
	poly[2] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab0.xmm1 ) );
	poly[4] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab1.xmm0 ) );
	poly[6] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab1.xmm1 ) );

}

static
void bs_i_butterfly_gf232_l1_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 , unsigned ska2 , unsigned ska3 )
{
	__m128i * poly = (__m128i *) poly128;
	//unsigned unit = 2;
	//unsigned unit_2 = 1;

	poly[1] ^= poly[0];
	poly[3] ^= poly[2];
	poly[5] ^= poly[4];
	poly[7] ^= poly[6];

	__m128i ml = _mm_load_si128( (__m128i*) __mask_low );
	__m128i m32_16 = _mm_load_si128( (__m128i*) __mask_32bit_low );
	__m128i m16_8 = _mm_load_si128( (__m128i*) __mask_16bit_low );

	struct xmm_x2 multab_80 = get_multab_sse( 0x80 );

	struct xmm_x4 src;
	src.xmm0 = poly[1];
	src.xmm1 = poly[3];
	src.xmm2 = poly[5];
	src.xmm3 = poly[7];

	struct xmm_x2 multab_0 = get_multab_sse( ska0 );
	struct xmm_x2 multab_1 = get_multab_sse( ska0>>8 );
	struct xmm_x2 multab_2 = get_multab_sse( ska0>>16 );
	struct xmm_x2 multab_3 = get_multab_sse( ska0>>24 );
	struct xmm_x4 ab = bs_gf232_mul_sse( src , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	poly[0] ^= ab.xmm0 & m32_16 & m16_8;
	poly[2] ^= ab.xmm1 & m32_16 & m16_8;
	poly[4] ^= ab.xmm2 & m32_16 & m16_8;
	poly[6] ^= ab.xmm3 & m32_16 & m16_8;

	multab_0 = get_multab_sse( ska1 );
	multab_1 = get_multab_sse( ska1>>8 );
	multab_2 = get_multab_sse( ska1>>16 );
	multab_3 = get_multab_sse( ska1>>24 );
	ab = bs_gf232_mul_sse( src , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , ab.xmm0 & m32_16 );
	poly[2] ^= _mm_andnot_si128( m16_8 , ab.xmm1 & m32_16 );
	poly[4] ^= _mm_andnot_si128( m16_8 , ab.xmm2 & m32_16 );
	poly[6] ^= _mm_andnot_si128( m16_8 , ab.xmm3 & m32_16 );

	multab_0 = get_multab_sse( ska2 );
	multab_1 = get_multab_sse( ska2>>8 );
	multab_2 = get_multab_sse( ska2>>16 );
	multab_3 = get_multab_sse( ska2>>24 );
	ab = bs_gf232_mul_sse( src , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m32_16 , ab.xmm0 & m16_8 );
	poly[2] ^= _mm_andnot_si128( m32_16 , ab.xmm1 & m16_8 );
	poly[4] ^= _mm_andnot_si128( m32_16 , ab.xmm2 & m16_8 );
	poly[6] ^= _mm_andnot_si128( m32_16 , ab.xmm3 & m16_8 );

	multab_0 = get_multab_sse( ska3 );
	multab_1 = get_multab_sse( ska3>>8 );
	multab_2 = get_multab_sse( ska3>>16 );
	multab_3 = get_multab_sse( ska3>>24 );
	ab = bs_gf232_mul_sse( src , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	poly[0] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab.xmm0 ) );
	poly[2] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab.xmm1 ) );
	poly[4] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab.xmm2 ) );
	poly[6] ^= _mm_andnot_si128( m16_8 , _mm_andnot_si128( m32_16 , ab.xmm3 ) );
}


static
void bs_i_butterfly_l1_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 , unsigned ska2 , unsigned ska3 )
{
	if( ska3 < 256 ) bs_i_butterfly_gf256_l1_avx2( poly128 , ska0 , ska1 , ska2 , ska3 );
	else if( ska3 < 65536 ) bs_i_butterfly_gf65536_l1_avx2( poly128 , ska0 , ska1 , ska2 , ska3 );
	else bs_i_butterfly_gf232_l1_avx2( poly128 , ska0 , ska1 , ska2 , ska3 );
}

/////////////////////////////////////////////


static
void bs_i_butterfly_gf256_l2_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 4;
	//unsigned unit_2 = 2;

	poly[0] ^= _mm256_slli_epi16( poly[0] , 8 );
	poly[1] ^= _mm256_slli_epi16( poly[1] , 8 );
	poly[2] ^= _mm256_slli_epi16( poly[2] , 8 );
	poly[3] ^= _mm256_slli_epi16( poly[3] , 8 );

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	__m256i m32_16 = _mm256_load_si256( (__m256i*) __mask_32bit_low );

	struct ymm_x2 multab_0 = get_multab_avx2( ska0 );
	__m256i ab_ymm0 = bs_gf256_mul_avx2( poly[0] , multab_0 , ml );
	__m256i ab_ymm1 = bs_gf256_mul_avx2( poly[1] , multab_0 , ml );
	__m256i ab_ymm2 = bs_gf256_mul_avx2( poly[2] , multab_0 , ml );
	__m256i ab_ymm3 = bs_gf256_mul_avx2( poly[3] , multab_0 , ml );
	poly[0] ^= _mm256_srli_epi16( ab_ymm0 & m32_16 , 8 );
	poly[1] ^= _mm256_srli_epi16( ab_ymm1 & m32_16 , 8 );
	poly[2] ^= _mm256_srli_epi16( ab_ymm2 & m32_16 , 8 );
	poly[3] ^= _mm256_srli_epi16( ab_ymm3 & m32_16 , 8 );

	multab_0 = get_multab_avx2( ska1 );
	ab_ymm0 = bs_gf256_mul_avx2( poly[0] , multab_0 , ml );
	ab_ymm1 = bs_gf256_mul_avx2( poly[1] , multab_0 , ml );
	ab_ymm2 = bs_gf256_mul_avx2( poly[2] , multab_0 , ml );
	ab_ymm3 = bs_gf256_mul_avx2( poly[3] , multab_0 , ml );
	poly[0] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab_ymm0) , 8 );
	poly[1] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab_ymm1) , 8 );
	poly[2] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab_ymm2) , 8 );
	poly[3] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab_ymm3) , 8 );
}


static
void bs_i_butterfly_gf65536_l2_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 4;
	//unsigned unit_2 = 2;

	poly[0] ^= _mm256_slli_epi16( poly[0] , 8 );
	poly[1] ^= _mm256_slli_epi16( poly[1] , 8 );
	poly[2] ^= _mm256_slli_epi16( poly[2] , 8 );
	poly[3] ^= _mm256_slli_epi16( poly[3] , 8 );

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );

	__m256i m32_16 = _mm256_load_si256( (__m256i*) __mask_32bit_low );

	struct ymm_x2 multab_0 = get_multab_avx2( ska0 );
	struct ymm_x2 multab_1 = get_multab_avx2( ska0>>8 );
	struct ymm_x2 ab0,ab1,multab_01;

	multab_01.ymm0 = multab_0.ymm0^multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1^multab_1.ymm1;
	ab0 = bs_gf65536_mul_avx2( poly[0] , poly[1] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_avx2( poly[2] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm256_srli_epi16( ab0.ymm0 & m32_16 , 8 );
	poly[1] ^= _mm256_srli_epi16( ab0.ymm1 & m32_16 , 8 );
	poly[2] ^= _mm256_srli_epi16( ab1.ymm0 & m32_16 , 8 );
	poly[3] ^= _mm256_srli_epi16( ab1.ymm1 & m32_16 , 8 );

	multab_0 = get_multab_avx2( ska1 );
	multab_1 = get_multab_avx2( ska1>>8 );
	multab_01.ymm0 = multab_0.ymm0^multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1^multab_1.ymm1;
	ab0 = bs_gf65536_mul_avx2( poly[0] , poly[1] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	ab1 = bs_gf65536_mul_avx2( poly[2] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	poly[0] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab0.ymm0) , 8 );
	poly[1] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab0.ymm1) , 8 );
	poly[2] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab1.ymm0) , 8 );
	poly[3] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab1.ymm1) , 8 );
}

static
void bs_i_butterfly_gf232_l2_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 4;
	//unsigned unit_2 = 2;

	poly[0] ^= _mm256_slli_epi16( poly[0] , 8 );
	poly[1] ^= _mm256_slli_epi16( poly[1] , 8 );
	poly[2] ^= _mm256_slli_epi16( poly[2] , 8 );
	poly[3] ^= _mm256_slli_epi16( poly[3] , 8 );

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );

	__m256i m32_16 = _mm256_load_si256( (__m256i*) __mask_32bit_low );

	struct ymm_x2 multab_0 = get_multab_avx2( ska0 );
	struct ymm_x2 multab_1 = get_multab_avx2( ska0>>8 );
	struct ymm_x2 multab_2 = get_multab_avx2( ska0>>16 );
	struct ymm_x2 multab_3 = get_multab_avx2( ska0>>24 );

	struct ymm_x4 ab = bs_gf232_mul_avx2( *(struct ymm_x4 *)poly , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );

	poly[0] ^= _mm256_srli_epi16( ab.ymm0 & m32_16 , 8 );
	poly[1] ^= _mm256_srli_epi16( ab.ymm1 & m32_16 , 8 );
	poly[2] ^= _mm256_srli_epi16( ab.ymm2 & m32_16 , 8 );
	poly[3] ^= _mm256_srli_epi16( ab.ymm3 & m32_16 , 8 );

	multab_0 = get_multab_avx2( ska1 );
	multab_1 = get_multab_avx2( ska1>>8 );
	multab_2 = get_multab_avx2( ska1>>16 );
	multab_3 = get_multab_avx2( ska1>>24 );

	ab = bs_gf232_mul_avx2( *(struct ymm_x4 *)poly , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );

	poly[0] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab.ymm0) , 8 );
	poly[1] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab.ymm1) , 8 );
	poly[2] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab.ymm2) , 8 );
	poly[3] ^= _mm256_srli_epi16( _mm256_andnot_si256( m32_16 , ab.ymm3) , 8 );
}

static
void bs_i_butterfly_l2_avx2( __m128i * poly128 , unsigned ska0 , unsigned ska1 )
{
	if( ska1 < 256 ) bs_i_butterfly_gf256_l2_avx2( poly128 , ska0 , ska1 );
	else if( ska1 < 65536 ) bs_i_butterfly_gf65536_l2_avx2( poly128 , ska0 , ska1 );
	else bs_i_butterfly_gf232_l2_avx2( poly128 , ska0 , ska1 );
}


/////////////////////////////////////////////////

static
void bs_i_butterfly_gf256_l3_avx2( __m128i * poly128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 8;
	//unsigned unit_2 = 4;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );

	poly[0] ^= _mm256_slli_epi32( poly[0] , 16 );
	poly[1] ^= _mm256_slli_epi32( poly[1] , 16 );
	poly[2] ^= _mm256_slli_epi32( poly[2] , 16 );
	poly[3] ^= _mm256_slli_epi32( poly[3] , 16 );

	__m256i ab_ymm0 = bs_gf256_mul_avx2( poly[0] , multab_0 , ml );
	__m256i ab_ymm1 = bs_gf256_mul_avx2( poly[1] , multab_0 , ml );
	__m256i ab_ymm2 = bs_gf256_mul_avx2( poly[2] , multab_0 , ml );
	__m256i ab_ymm3 = bs_gf256_mul_avx2( poly[3] , multab_0 , ml );

	poly[0] ^= _mm256_srli_epi32( ab_ymm0 , 16 );
	poly[1] ^= _mm256_srli_epi32( ab_ymm1 , 16 );
	poly[2] ^= _mm256_srli_epi32( ab_ymm2 , 16 );
	poly[3] ^= _mm256_srli_epi32( ab_ymm3 , 16 );

}

static
void bs_i_butterfly_gf65536_l3_avx2( __m128i * poly128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 8;
	//unsigned unit_2 = 4;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );
	struct ymm_x2 multab_1 = get_multab_avx2( ska>>8 );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );
	struct ymm_x2 multab_01;
	multab_01.ymm0 = multab_0.ymm0^multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1^multab_1.ymm1;

	poly[0] ^= _mm256_slli_epi32( poly[0] , 16 );
	poly[1] ^= _mm256_slli_epi32( poly[1] , 16 );
	poly[2] ^= _mm256_slli_epi32( poly[2] , 16 );
	poly[3] ^= _mm256_slli_epi32( poly[3] , 16 );

	struct ymm_x2 ab0 = bs_gf65536_mul_avx2( poly[0] , poly[1] , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	struct ymm_x2 ab1 = bs_gf65536_mul_avx2( poly[2] , poly[3] , multab_0 , multab_1 , multab_01 , multab_80 , ml );

	poly[0] ^= _mm256_srli_epi32( ab0.ymm0 , 16 );
	poly[1] ^= _mm256_srli_epi32( ab0.ymm1 , 16 );
	poly[2] ^= _mm256_srli_epi32( ab1.ymm0 , 16 );
	poly[3] ^= _mm256_srli_epi32( ab1.ymm1 , 16 );
}

static
void bs_i_butterfly_gf232_l3_avx2( __m128i * poly128 , unsigned ska )
{
	__m256i * poly = (__m256i *) poly128;
	//unsigned unit = 8;
	//unsigned unit_2 = 4;

	__m256i ml = _mm256_load_si256( (__m256i*) __mask_low );
	struct ymm_x2 multab_0 = get_multab_avx2( ska );
	struct ymm_x2 multab_1 = get_multab_avx2( ska>>8 );
	struct ymm_x2 multab_2 = get_multab_avx2( ska>>16 );
	struct ymm_x2 multab_3 = get_multab_avx2( ska>>24 );
	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );

	poly[0] ^= _mm256_slli_epi32( poly[0] , 16 );
	poly[1] ^= _mm256_slli_epi32( poly[1] , 16 );
	poly[2] ^= _mm256_slli_epi32( poly[2] , 16 );
	poly[3] ^= _mm256_slli_epi32( poly[3] , 16 );

	struct ymm_x4 ab = bs_gf232_mul_avx2( *(struct ymm_x4 *)poly , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );

	poly[0] ^= _mm256_srli_epi32( ab.ymm0 , 16 );
	poly[1] ^= _mm256_srli_epi32( ab.ymm1 , 16 );
	poly[2] ^= _mm256_srli_epi32( ab.ymm2 , 16 );
	poly[3] ^= _mm256_srli_epi32( ab.ymm3 , 16 );
}

static
void bs_i_butterfly_l3_avx2( __m128i * poly128 , unsigned ska )
{
	if( ska < 256 ) bs_i_butterfly_gf256_l3_avx2( poly128 , ska );
	else if( ska < 65536 ) bs_i_butterfly_gf65536_l3_avx2( poly128 , ska );
	else bs_i_butterfly_gf232_l3_avx2( poly128 , ska );
}


#endif





////////////////////////////////////////////////////////




void butterfly_net_half_inp( uint64_t * fx , unsigned n_fx )
{
	if( 1 >= n_fx ) return;
	if( 16 > n_fx ) { printf("unsupported number of terms.\n"); exit(-1); }

	unsigned log_n = __builtin_ctz( n_fx );
	unsigned n_terms = n_fx;

	__m128i * poly = (__m128i*) &fx[0];

	/// first layer
	memcpy( poly + (n_terms/2) , poly , 8*n_terms );

#ifdef _SHUFFLE_BYTE_AVX2_
	for(unsigned i=log_n-1; i > 3 ; i-- ){
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		bs_butterfly_0_avx2( poly , unit );
		for(unsigned j=1;j<num;j++) {
			bs_butterfly_avx2( poly + j*unit , unit , get_s_k_a( i-1 , j ) );
		}
	}
#endif

	{
		unsigned i = 3;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j++) bs_butterfly_l3_avx2( poly + j*unit , get_s_k_a( i-1 , j ) );
	}
	{
		unsigned i = 2;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j+=2) bs_butterfly_l2_avx2( poly + j*unit , get_s_k_a( i-1 , j ) , get_s_k_a( i-1 , j+1 ) );
	}
	{
		unsigned i = 1;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j+=4) bs_butterfly_l1_avx2( poly + j*unit ,
			get_s_k_a( i-1 , j ) , get_s_k_a( i-1 , j+1 ) , get_s_k_a( i-1 , j+2 ) , get_s_k_a( i-1 , j+3 ) );
	}

}


#ifdef _PROFILE_
#include "benchmark.h"

struct benchmark bm_bm;
struct benchmark bm_mul;
#endif



void i_butterfly_net( uint64_t * fx , unsigned n_fx )
{
	if( 1 >= n_fx ) return;
	if( 8 > n_fx ) { printf("unsupported number of terms.\n"); exit(-1); }

	unsigned log_n = __builtin_ctz( n_fx );

//	printf("n_fx: %d, log_n: %d\n", n_fx , log_n );
//	byte_dump( fx , n_fx ); puts("");

	__m128i *poly = (__m128i*) &fx[0];
	unsigned n_terms = n_fx;

#ifdef _PROFILE_
bm_start(&bm_mul);
#endif
	{
		unsigned i = 1;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j+=4) bs_i_butterfly_l1_avx2( poly + j*unit ,
			get_s_k_a( i-1 , j ) , get_s_k_a( i-1 , j+1 ) , get_s_k_a( i-1 , j+2 ) , get_s_k_a( i-1 , j+3 ) );
	}
	{
		unsigned i = 2;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j+=2) bs_i_butterfly_l2_avx2( poly + j*unit , get_s_k_a( i-1 , j ) , get_s_k_a( i-1 , j+1 ) );
	}
	{
		unsigned i = 3;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j++) bs_i_butterfly_l3_avx2( poly + j*unit , get_s_k_a( i-1 , j ) );
	}
#ifdef _PROFILE_
bm_stop(&bm_mul);
#endif


#ifdef _PROFILE_
bm_start(&bm_bm);
#endif
#ifdef _SHUFFLE_BYTE_AVX2_
	for(unsigned i=4; i <= log_n; i++) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		bs_butterfly_0_avx2( poly , unit );
		for(unsigned j=1;j<num;j++) {
			bs_i_butterfly_avx2( poly + j*unit , unit , get_s_k_a( i-1 , j ) );
		}
	}
#endif
#ifdef _PROFILE_
bm_stop(&bm_bm);
#endif

}





////////////////////////////////////////////////////////////////////////







void butterfly_net_half_inp_256( uint64_t * fx , unsigned n_fx )
{
	if( 1 >= n_fx ) return;
	if( 8 > n_fx ) { printf("unsupported number of terms.\n"); exit(-1); }

	unsigned log_n = __builtin_ctz( n_fx );
	unsigned n_terms = n_fx;

	__m256i * poly = (__m256i*) &fx[0];

	/// first layer
	memcpy( poly + (n_terms/2) , poly , 16*n_terms );

#ifdef _SHUFFLE_BYTE_AVX2_
	for(unsigned i=log_n-1; i > 2 ; i-- ){
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		bs_butterfly_0_avx2( (__m128i*)poly , unit*2 );
		for(unsigned j=1;j<num;j++) {
			bs_butterfly_avx2( (__m128i*)(poly + j*unit) , unit*2 , get_s_k_a( i-1 , j ) );
		}
	}
#endif

	{
		unsigned i = 2;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j++) bs_butterfly_l3_avx2( (__m128i*)(poly + j*unit) , get_s_k_a( i-1 , j ) );
	}
	{
		unsigned i = 1;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j+=2) bs_butterfly_l2_avx2( (__m128i*)(poly + j*unit) , get_s_k_a( i-1 , j ) , get_s_k_a( i-1 , j+1 ) );
	}

}


void i_butterfly_net_256( uint64_t * fx , unsigned n_fx )
{
	if( 1 >= n_fx ) return;
	if( 8 > n_fx ) { printf("unsupported number of terms.\n"); exit(-1); }

	unsigned log_n = __builtin_ctz( n_fx );

//	printf("n_fx: %d, log_n: %d\n", n_fx , log_n );
//	byte_dump( fx , n_fx ); puts("");

	__m256i *poly = (__m256i*) &fx[0];
	unsigned n_terms = n_fx;

#ifdef _PROFILE_
bm_start(&bm_mul);
#endif
	{
		unsigned i = 1;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j+=2) bs_i_butterfly_l2_avx2( (__m128i*)(poly + j*unit) , get_s_k_a( i-1 , j ) , get_s_k_a( i-1 , j+1 ) );
	}
	{
		unsigned i = 2;
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;
		for(unsigned j=0;j<num;j++) bs_i_butterfly_l3_avx2( (__m128i*)(poly + j*unit) , get_s_k_a( i-1 , j ) );
	}
#ifdef _PROFILE_
bm_stop(&bm_mul);
#endif


#ifdef _PROFILE_
bm_start(&bm_bm);
#endif
#ifdef _SHUFFLE_BYTE_AVX2_
	for(unsigned i=3; i <= log_n; i++) {
		unsigned unit = (1<<i);
		unsigned num = n_terms / unit;

		bs_butterfly_0_avx2( (__m128i*)poly , unit*2 );
		for(unsigned j=1;j<num;j++) {
			bs_i_butterfly_avx2( (__m128i*)(poly + j*unit) , unit*2 , get_s_k_a( i-1 , j ) );
		}
	}
#endif
#ifdef _PROFILE_
bm_stop(&bm_bm);
#endif

}




