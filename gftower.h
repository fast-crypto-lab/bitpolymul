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

#ifndef _GF_EXT_TOWER_H_
#define _GF_EXT_TOWER_H_


#include <stdint.h>

#include <immintrin.h>

#include "gf16_tabs.h"


struct xmm_x2 {
	__m128i xmm0;
	__m128i xmm1;
};


static inline
struct xmm_x2 get_multab_sse( unsigned ska ) {
	struct xmm_x2 tab;
//        __m128i multab_l = _mm_load_si128( (__m128i*) (__gf256_mul+32*ska) );
//        __m128i multab_h = _mm_load_si128( (__m128i*) (__gf256_mul+32*ska+16) );
	ska &= 0xff;
	tab.xmm0 = _mm_load_si128( (__m128i*) (__gf256_mul+32*ska) );
	tab.xmm1 = _mm_load_si128( (__m128i*) (__gf256_mul+32*ska+16) );
	return tab;
}


/////////////////////////////////////

/// return seperated high/low nibble
static inline
__m128i bs_gf256_mul_sse_2( __m128i src_l , __m128i src_h , struct xmm_x2 multab ) {
	return _mm_shuffle_epi8(multab.xmm0,src_l)^_mm_shuffle_epi8(multab.xmm1,src_h);
}

static inline
__m128i bs_gf256_mul_sse( __m128i src , struct xmm_x2 multab , __m128i ml  ) {
	__m128i src_l = src & ml;
	__m128i src_h = _mm_srli_epi16( _mm_andnot_si128( ml , src ) ,4 );
	return _mm_shuffle_epi8(multab.xmm0,src_l)^_mm_shuffle_epi8(multab.xmm1,src_h);
}

/////////////////////////////////////


static inline
struct xmm_x2 bs_gf65536_mul_sse( __m128i src_0 , __m128i src_1 ,
		struct xmm_x2 multab_0 , struct xmm_x2 multab_1 , struct xmm_x2 multab_01 ,
		struct xmm_x2 multab_80 , __m128i ml ) {

	__m128i src_0l = src_0 & ml;
	__m128i src_0h = _mm_srli_epi16( src_0 ,4) & ml;
	__m128i src_1l = src_1 & ml;
	__m128i src_1h = _mm_srli_epi16( src_1 ,4) & ml;

	__m128i ab0 = bs_gf256_mul_sse_2( src_0l , src_0h , multab_0 );
	__m128i ab2 = bs_gf256_mul_sse_2( src_1l , src_1h , multab_1 );
	__m128i ab1 = bs_gf256_mul_sse_2( src_0l^src_1l , src_0h^src_1h , multab_01 )^ab0;
	__m128i ab2_h = _mm_srli_epi16(ab2,4)&ml;
	__m128i ab2r = bs_gf256_mul_sse_2( ab2&ml , ab2_h , multab_80 );

	struct xmm_x2 ret;
	ret.xmm0 = ab0^ab2r;
	ret.xmm1 = ab1;
	return ret;
}


static inline
struct xmm_x2 bs_gf65536_mul_0x8000_sse( struct xmm_x2 a , struct xmm_x2 multab_80 , __m128i mask_f ) {
	__m128i a0x80 = bs_gf256_mul_sse( a.xmm0 , multab_80 , mask_f );
	__m128i a1x80 = bs_gf256_mul_sse( a.xmm1 , multab_80 , mask_f );
	struct xmm_x2 ret;
	ret.xmm1 = a0x80^a1x80;
	ret.xmm0 = bs_gf256_mul_sse( a1x80 , multab_80 , mask_f );
	return ret;
}


/////////////////////////////////////

struct xmm_x4 {
	__m128i xmm0;
	__m128i xmm1;
	__m128i xmm2;
	__m128i xmm3;
};

static inline
struct xmm_x4 bs_gf232_mul_sse( struct xmm_x4 src ,
		struct xmm_x2 multab_0 , struct xmm_x2 multab_1 , struct xmm_x2 multab_2 , struct xmm_x2 multab_3 ,
		struct xmm_x2 multab_80 , __m128i ml ) {

	struct xmm_x2 multab_01;
	multab_01.xmm0 = multab_0.xmm0 ^ multab_1.xmm0;
	multab_01.xmm1 = multab_0.xmm1 ^ multab_1.xmm1;
	struct xmm_x2 a0 = bs_gf65536_mul_sse( src.xmm0 , src.xmm1 , multab_0 , multab_1 , multab_01 , multab_80 , ml );

	struct xmm_x2 multab_23;
	multab_23.xmm0 = multab_2.xmm0 ^ multab_3.xmm0;
	multab_23.xmm1 = multab_2.xmm1 ^ multab_3.xmm1;
	struct xmm_x2 a2 = bs_gf65536_mul_sse( src.xmm2 , src.xmm3 , multab_2 , multab_3 , multab_23 , multab_80 , ml );

	struct xmm_x2 multab_0123;
	multab_0123.xmm0 = multab_01.xmm0 ^ multab_23.xmm0;
	multab_0123.xmm1 = multab_01.xmm1 ^ multab_23.xmm1;

	multab_01.xmm0 = multab_0.xmm0 ^ multab_2.xmm0;
	multab_01.xmm1 = multab_0.xmm1 ^ multab_2.xmm1;
	multab_23.xmm0 = multab_1.xmm0 ^ multab_3.xmm0;
	multab_23.xmm1 = multab_1.xmm1 ^ multab_3.xmm1;

	struct xmm_x2 a1 = bs_gf65536_mul_sse( src.xmm0^src.xmm2 , src.xmm1^src.xmm3 ,
		multab_01 , multab_23 , multab_0123 , multab_80 , ml );

	a1.xmm0 ^= a0.xmm0;
	a1.xmm1 ^= a0.xmm1;

	multab_01 = bs_gf65536_mul_0x8000_sse( a2 , multab_80 , ml );
	a0.xmm0 ^= multab_01.xmm0;
	a0.xmm1 ^= multab_01.xmm1;

	struct xmm_x4 ret;
	ret.xmm0 = a0.xmm0;
	ret.xmm1 = a0.xmm1;
	ret.xmm2 = a1.xmm0;
	ret.xmm3 = a1.xmm1;
	return ret;
}

////////////////////////////////////////////////////
///
/// considered bad
///
////////////////////////////////////////////////////

static inline
__m128i gf232_mul_sse( __m128i src ,
		struct xmm_x2 multab_0 , struct xmm_x2 multab_1 , struct xmm_x2 multab_2 , struct xmm_x2 multab_3 ,
		struct xmm_x2 multab_80 , __m128i ml , __m128i m32_8 ) {

	struct xmm_x4 a;
	a.xmm0 = src & m32_8;
	a.xmm1 = _mm_srli_si128(src,1)&m32_8;
	a.xmm2 = _mm_srli_si128(src,2)&m32_8;
	a.xmm3 = _mm_srli_si128(src,3)&m32_8;

	struct xmm_x4 r = bs_gf232_mul_sse( a , multab_0 , multab_1 , multab_2 , multab_3 , multab_80 , ml );
	r.xmm0 ^= _mm_slli_si128(r.xmm1,1) ^ _mm_slli_si128(r.xmm2,2) ^_mm_slli_si128(r.xmm3,3);
	return r.xmm0;
}


static inline
__m128i gf65536_mul_sse( __m128i src ,
		struct xmm_x2 multab_0 , struct xmm_x2 multab_1 , struct xmm_x2 multab_01 ,
		struct xmm_x2 multab_80 , __m128i ml , __m128i m16l ) {

	__m128i src_0 = src & m16l;
	__m128i src_1 = _mm_srli_si128( src , 1 ) & m16l;

	struct xmm_x2 r = bs_gf65536_mul_sse( src_0 , src_1 , multab_0 , multab_1 , multab_01 , multab_80 , ml );
	return r.xmm0^_mm_slli_si128( r.xmm1 , 1 );
}




///////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////
///
///     ymm
///
///////////////////////////////////////////////////////

struct ymm_x2 {
	__m256i ymm0;
	__m256i ymm1;
};

static inline
struct ymm_x2 get_multab_avx2( unsigned ska ) {
	ska &= 0xff;
	__m256i multab = _mm256_load_si256( (__m256i*) (__gf256_mul+32*ska) );
	__m256i multab_l = _mm256_permute2x128_si256( multab , multab , 0 );
	__m256i multab_h = _mm256_permute2x128_si256( multab , multab , 0x11 );
	struct ymm_x2 tab;
	tab.ymm0 = multab_l;
	tab.ymm1 = multab_h;
	return tab;
}


////////////////////////////////////////////////////////


static inline
__m256i bs_gf256_mul_avx2( __m256i src , struct ymm_x2 multab , __m256i mask_f  ) {
	__m256i src_l = src & mask_f;
	__m256i src_h = _mm256_srli_epi16( _mm256_andnot_si256( mask_f , src ) ,4 );
	return _mm256_shuffle_epi8(multab.ymm0,src_l)^_mm256_shuffle_epi8(multab.ymm1,src_h);
}

static inline
__m256i bs_gf256_mul_avx2_2( __m256i src_l , __m256i src_h , struct ymm_x2 multab ) {
	return _mm256_shuffle_epi8(multab.ymm0,src_l)^_mm256_shuffle_epi8(multab.ymm1,src_h);
}

//////////////////////////////////////////////////////


static inline
struct ymm_x2 bs_gf65536_mul_avx2( __m256i src_0 , __m256i src_1 , struct ymm_x2 multab_0 , struct ymm_x2 multab_1 ,
		struct ymm_x2 multab_01 , struct ymm_x2 multab_80 , __m256i ml ) {

	__m256i src_0l = src_0 & ml;
	__m256i src_0h = _mm256_srli_epi16( src_0 ,4) & ml;
	__m256i src_1l = src_1 & ml;
	__m256i src_1h = _mm256_srli_epi16( src_1 ,4) & ml;

	__m256i ab0 = bs_gf256_mul_avx2_2( src_0l , src_0h , multab_0 );
	__m256i ab2 = bs_gf256_mul_avx2_2( src_1l , src_1h , multab_1 );
	__m256i ab1 = bs_gf256_mul_avx2_2( src_0l^src_1l , src_0h^src_1h , multab_01 ) ^ ab0;
	__m256i ab2r = bs_gf256_mul_avx2( ab2 , multab_80 , ml );

	struct ymm_x2 ret;
	ret.ymm0 = ab0^ab2r;
	ret.ymm1 = ab1;
	return ret;
}



static inline
struct ymm_x2 bs_gf65536_mul_0x8000_avx2( struct ymm_x2 a , struct ymm_x2 multab_80 , __m256i mask_f ) {
	__m256i a0x80 = bs_gf256_mul_avx2( a.ymm0 , multab_80 , mask_f );
	__m256i a1x80 = bs_gf256_mul_avx2( a.ymm1 , multab_80 , mask_f );
	struct ymm_x2 ret;
	ret.ymm1 = a0x80^a1x80;
	ret.ymm0 = bs_gf256_mul_avx2( a1x80 , multab_80 , mask_f );
	return ret;
}

/////////////////////////////////////////////////////////////////////

struct ymm_x4 {
	__m256i ymm0;
	__m256i ymm1;
	__m256i ymm2;
	__m256i ymm3;
};

///////////////////////////////////////////////////////////////////////


static inline
struct ymm_x4 bs_gf232_mul_avx2( struct ymm_x4 src , struct ymm_x2 multab_0 , struct ymm_x2 multab_1 ,struct ymm_x2 multab_2 ,struct ymm_x2 multab_3 ,
		struct ymm_x2 multab_80 , __m256i ml ) {
	struct ymm_x2 r0;
	struct ymm_x2 r1;
	struct ymm_x2 r2;

	struct ymm_x2 multab_01;
	multab_01.ymm0 = multab_0.ymm0^multab_1.ymm0;
	multab_01.ymm1 = multab_0.ymm1^multab_1.ymm1;
	r0 = bs_gf65536_mul_avx2( src.ymm0 , src.ymm1 , multab_0 , multab_1 , multab_01 , multab_80 , ml );

	struct ymm_x2 multab_23;
	multab_23.ymm0 = multab_2.ymm0^multab_3.ymm0;
	multab_23.ymm1 = multab_2.ymm1^multab_3.ymm1;
	r2 = bs_gf65536_mul_avx2( src.ymm2 , src.ymm3 , multab_2 , multab_3 , multab_23 , multab_80 , ml );

///// !!!!!!!!! mismatch between contents and name
	multab_01.ymm0 = multab_0.ymm0 ^ multab_2.ymm0;
	multab_01.ymm1 = multab_0.ymm1 ^ multab_2.ymm1;
///// !!!!!!!!! mismatch between contents and name
	multab_23.ymm0 = multab_1.ymm0 ^ multab_3.ymm0;
	multab_23.ymm1 = multab_1.ymm1 ^ multab_3.ymm1;
	struct ymm_x2 multab_0213;
	multab_0213.ymm0 = multab_01.ymm0^multab_23.ymm0;
	multab_0213.ymm1 = multab_01.ymm1^multab_23.ymm1;

	r1 = bs_gf65536_mul_avx2( src.ymm0^src.ymm2 , src.ymm1^src.ymm3 , multab_01 , multab_23 , multab_0213 , multab_80 , ml );

	struct ymm_x4 ret;
	ret.ymm0 = r0.ymm0;
	ret.ymm1 = r0.ymm1;
	ret.ymm2 = r0.ymm0 ^ r1.ymm0;
	ret.ymm3 = r0.ymm1 ^ r1.ymm1;

	struct ymm_x2 rd = bs_gf65536_mul_0x8000_avx2( r2 , multab_80 , ml );
	ret.ymm0 ^= rd.ymm0;
	ret.ymm1 ^= rd.ymm1;
	return ret;
}


///////////////////////////////////////////////////////
///
/// still used
///
///////////////////////////////////////////////////////


static inline
struct ymm_x2 bs_gf65536_12bit_mul_avx2( __m256i src_0 , __m256i src_1 ,  __m256i multab_0l , __m256i multab_0h , __m256i multab_1l ,
		__m256i multab_01l , __m256i multab_01h , __m256i multab_80l , __m256i multab_80h ,  __m256i ml  ) {
	__m256i src_0l = src_0 & ml;
	__m256i src_0h = _mm256_srli_epi16( src_0 ,4) & ml;
	__m256i src_1l = src_1 & ml;
	__m256i src_1h = _mm256_srli_epi16( src_1 ,4) & ml;

	__m256i ab0 = _mm256_shuffle_epi8(multab_0l,src_0l)^_mm256_shuffle_epi8(multab_0h,src_0h);
	__m256i ab2_l = _mm256_shuffle_epi8(multab_1l,src_1l);
	__m256i ab2_h = _mm256_shuffle_epi8(multab_1l,src_1h);
	__m256i ab1 = _mm256_shuffle_epi8(multab_01l,src_1l^src_0l)^_mm256_shuffle_epi8(multab_01h,src_1h^src_0h)^ab0;
	__m256i ab2r = _mm256_shuffle_epi8(multab_80l,ab2_l)^_mm256_shuffle_epi8(multab_80h,ab2_h);

	struct ymm_x2 ret;
	ret.ymm0 = ab0^ab2r;
	ret.ymm1 = ab1;
	return ret;
}


static inline
struct ymm_x4 bs_gf232_20bit_mul_avx2( __m256i src_0 , __m256i src_1 ,  __m256i src_2 , __m256i src_3 ,
		__m256i multab_0l , __m256i multab_0h , __m256i multab_1l , __m256i multab_1h ,
		__m256i multab_2l ,
		__m256i multab_80l , __m256i multab_80h ,  __m256i ml  ) {
	__m256i src_0l = src_0 & ml;
	__m256i src_0h = _mm256_srli_epi16( src_0 ,4) & ml;
	__m256i src_1l = src_1 & ml;
	__m256i src_1h = _mm256_srli_epi16( src_1 ,4) & ml;
	__m256i multab_01l = multab_0l^multab_1l;
	__m256i multab_01h = multab_0h^multab_1h;
	struct ymm_x4 ret;
	{
		__m256i ab0 = _mm256_shuffle_epi8(multab_0l,src_0l)^_mm256_shuffle_epi8(multab_0h,src_0h);
		__m256i ab2 = _mm256_shuffle_epi8(multab_1l,src_1l)^_mm256_shuffle_epi8(multab_1h,src_1h);
		__m256i ab1 = _mm256_shuffle_epi8(multab_01l,src_1l^src_0l)^_mm256_shuffle_epi8(multab_01h,src_1h^src_0h)^ab0;
		__m256i ab2_h = _mm256_srli_epi16(ab2,4)&ml;
		__m256i ab2r = _mm256_shuffle_epi8(multab_80l,ab2&ml)^_mm256_shuffle_epi8(multab_80h,ab2_h);
	ret.ymm0 = ab0^ab2r;
	ret.ymm1 = ab1;
	}

	__m256i src_2l = src_2 & ml;
	__m256i src_2h = _mm256_srli_epi16( src_2 ,4) & ml;
	__m256i src_3l = src_3 & ml;
	__m256i src_3h = _mm256_srli_epi16( src_3 ,4) & ml;
	{
		__m256i ab0 = _mm256_shuffle_epi8(multab_0l,src_2l)^_mm256_shuffle_epi8(multab_0h,src_2h);
		__m256i ab2 = _mm256_shuffle_epi8(multab_1l,src_3l)^_mm256_shuffle_epi8(multab_1h,src_3h);
		__m256i ab1 = _mm256_shuffle_epi8(multab_01l,src_3l^src_2l)^_mm256_shuffle_epi8(multab_01h,src_3h^src_2h)^ab0;
		__m256i ab2_h = _mm256_srli_epi16(ab2,4)&ml;
		__m256i ab2r = _mm256_shuffle_epi8(multab_80l,ab2&ml)^_mm256_shuffle_epi8(multab_80h,ab2_h);
	ret.ymm2 = ab0^ab2r;
	ret.ymm3 = ab1;
	}

	__m256i multab_2h = _mm256_slli_epi16( multab_2l , 4 );
	ret.ymm2 ^= _mm256_shuffle_epi8(multab_2l,src_0l) ^ _mm256_shuffle_epi8( multab_2h ,src_0h);
	ret.ymm3 ^= _mm256_shuffle_epi8(multab_2l,src_1l) ^ _mm256_shuffle_epi8( multab_2h ,src_1h);

	__m256i src_4l = _mm256_shuffle_epi8( multab_2l , src_2l );
	__m256i src_4h = _mm256_shuffle_epi8( multab_2l , src_2h );
	__m256i src_5l = _mm256_shuffle_epi8( multab_2l , src_3l );
	__m256i src_5h = _mm256_shuffle_epi8( multab_2l , src_3h );
	ret.ymm2 ^= src_4l ^ _mm256_slli_epi16( src_4h , 4 );
	ret.ymm3 ^= src_5l ^ _mm256_slli_epi16( src_5h , 4 );

	/// ret.ymm0,ret.ymm1 ^= (src_4,src_5)x0x8000
	ret.ymm1 ^= _mm256_shuffle_epi8(multab_80l,src_4l) ^ _mm256_shuffle_epi8(multab_80h,src_4h);
	__m256i src_xx = _mm256_shuffle_epi8(multab_80l,src_5l) ^ _mm256_shuffle_epi8(multab_80h,src_5h);
	ret.ymm1 ^= src_xx;

	__m256i src_xxl = src_xx & ml;
	__m256i src_xxh = _mm256_srli_epi16(src_xx,4) & ml;
	ret.ymm0 ^= _mm256_shuffle_epi8(multab_80l,src_xxl) ^ _mm256_shuffle_epi8(multab_80h,src_xxh);

	return ret;
}



///////////////////////////////////////////////////////

////////////////////////////////////////////////
///
/// logtab series start here
///
/////////////////////////////////////////////////


static inline
__m256i gf256_mul_avx2_logtab( __m256i a , __m256i b , struct ymm_x2 logtab , __m256i mul_8 , __m256i mask_f ) {

	__m256i a0 = a&mask_f;
	__m256i a1 = _mm256_srli_epi16(a,4)&mask_f;
	__m256i b0 = b&mask_f;
	__m256i b1 = _mm256_srli_epi16(b,4)&mask_f;

	__m256i la0 = _mm256_shuffle_epi8(logtab.ymm0,a0);
	__m256i la1 = _mm256_shuffle_epi8(logtab.ymm0,a1);
	__m256i lb0 = _mm256_shuffle_epi8(logtab.ymm0,b0);
	__m256i lb1 = _mm256_shuffle_epi8(logtab.ymm0,b1);

	__m256i la0b0 = _mm256_add_epi8(la0,lb0);
	__m256i la1b0 = _mm256_add_epi8(la1,lb0);
	__m256i la0b1 = _mm256_add_epi8(la0,lb1);
	__m256i la1b1 = _mm256_add_epi8(la1,lb1);

	__m256i r0 = _mm256_shuffle_epi8(logtab.ymm1, _mm256_sub_epi8(la0b0, mask_f&_mm256_cmpgt_epi8(la0b0,mask_f) ) );
	__m256i r1 = _mm256_shuffle_epi8(logtab.ymm1, _mm256_sub_epi8(la1b0, mask_f&_mm256_cmpgt_epi8(la1b0,mask_f) ) )
		^_mm256_shuffle_epi8(logtab.ymm1, _mm256_sub_epi8(la0b1, mask_f&_mm256_cmpgt_epi8(la0b1,mask_f) ) );
	__m256i r2 = _mm256_shuffle_epi8(logtab.ymm1, _mm256_sub_epi8(la1b1, mask_f&_mm256_cmpgt_epi8(la1b1,mask_f) ) );

	return _mm256_slli_epi16(r1^r2,4)^r0^_mm256_shuffle_epi8(mul_8,r2);
}

/// return seperated high/low nibble
static inline
struct ymm_x2 gf256_mul_avx2_logtab_2( __m256i a , __m256i b , struct ymm_x2 logtab , __m256i mul_8 , __m256i mask_f ) {

	__m256i a0 = a&mask_f;
	__m256i a1 = _mm256_srli_epi16(a,4)&mask_f;
	__m256i b0 = b&mask_f;
	__m256i b1 = _mm256_srli_epi16(b,4)&mask_f;

	__m256i la0 = _mm256_shuffle_epi8(logtab.ymm0,a0);
	__m256i la1 = _mm256_shuffle_epi8(logtab.ymm0,a1);
	__m256i lb0 = _mm256_shuffle_epi8(logtab.ymm0,b0);
	__m256i lb1 = _mm256_shuffle_epi8(logtab.ymm0,b1);

	__m256i la0b0 = _mm256_add_epi8(la0,lb0);
	__m256i la1b0 = _mm256_add_epi8(la1,lb0);
	__m256i la0b1 = _mm256_add_epi8(la0,lb1);
	__m256i la1b1 = _mm256_add_epi8(la1,lb1);

	__m256i r0 = _mm256_shuffle_epi8(logtab.ymm1, _mm256_sub_epi8(la0b0, mask_f&_mm256_cmpgt_epi8(la0b0,mask_f) ) );
	__m256i r1 = _mm256_shuffle_epi8(logtab.ymm1, _mm256_sub_epi8(la1b0, mask_f&_mm256_cmpgt_epi8(la1b0,mask_f) ) )
		^_mm256_shuffle_epi8(logtab.ymm1, _mm256_sub_epi8(la0b1, mask_f&_mm256_cmpgt_epi8(la0b1,mask_f) ) );
	__m256i r2 = _mm256_shuffle_epi8(logtab.ymm1, _mm256_sub_epi8(la1b1, mask_f&_mm256_cmpgt_epi8(la1b1,mask_f) ) );

	struct ymm_x2 ret;
	ret.ymm0 = r0^_mm256_shuffle_epi8(mul_8,r2);
	ret.ymm1 = r1^r2;
	return ret;
}

////////////////////////////////////////////

static inline
struct ymm_x2 gf65536_mul_avx2_logtab( struct ymm_x2 a , struct ymm_x2 b ,
	struct ymm_x2 logtab , struct ymm_x2 multab_80 , __m256i mul_8 , __m256i mask_f ) {

	__m256i ab0 = gf256_mul_avx2_logtab( a.ymm0 , b.ymm0 , logtab , mul_8 , mask_f );
	__m256i ab1 = gf256_mul_avx2_logtab( a.ymm0^a.ymm1 , b.ymm0^b.ymm1 , logtab , mul_8 , mask_f )^ab0;
	__m256i ab2 = gf256_mul_avx2_logtab( a.ymm1 , b.ymm1 , logtab , mul_8 , mask_f );

	struct ymm_x2 ret;
	ret.ymm0 = ab0^bs_gf256_mul_avx2(ab2,multab_80,mask_f);
	ret.ymm1 = ab1;
	return ret;
}

///////////////////////////////////////////////////////

static inline
struct ymm_x4 gf232_mul_avx2_logtab( struct ymm_x4 a , struct ymm_x4 b ,
	struct ymm_x2 logtab , struct ymm_x2 multab_80 , __m256i mul_8 , __m256i mask_f ) {

	struct ymm_x2 ab0 = gf65536_mul_avx2_logtab( *(struct ymm_x2*)(&a) , *(struct ymm_x2*)(&b) , logtab , multab_80 , mul_8 , mask_f );

	struct ymm_x2 a0_a1;
	a0_a1.ymm0 = a.ymm0^a.ymm2;
	a0_a1.ymm1 = a.ymm1^a.ymm3;
	struct ymm_x2 b0_b1;
	b0_b1.ymm0 = b.ymm0^b.ymm2;
	b0_b1.ymm1 = b.ymm1^b.ymm3;
	struct ymm_x2 ab1 = gf65536_mul_avx2_logtab( a0_a1 , b0_b1 , logtab , multab_80 , mul_8 , mask_f );
	ab1.ymm0 ^= ab0.ymm0;
	ab1.ymm1 ^= ab0.ymm1;

	struct ymm_x2 ab2 = gf65536_mul_avx2_logtab( *(struct ymm_x2*)(&a.ymm2) , *(struct ymm_x2*)(&b.ymm2) , logtab , multab_80 , mul_8 , mask_f );

	struct ymm_x4 ret;
	ret.ymm0 = ab0.ymm0;
	ret.ymm1 = ab0.ymm1;
	ret.ymm2 = ab1.ymm0;
	ret.ymm3 = ab1.ymm1;

	ab0 = bs_gf65536_mul_0x8000_avx2(ab2,multab_80,mask_f);
	ret.ymm0 ^= ab0.ymm0;
	ret.ymm1 ^= ab0.ymm1;
	return ret;
}

static inline
struct ymm_x4 gf232_mul_0x80000000_avx2( struct ymm_x4 a , struct ymm_x2 multab_80 , __m256i mask_f ) {

	struct ymm_x2 a0x8000 = bs_gf65536_mul_0x8000_avx2( *(struct ymm_x2*)(&a) , multab_80 , mask_f );
	struct ymm_x2 a1x8000 = bs_gf65536_mul_0x8000_avx2( *(struct ymm_x2*)(&a.ymm2) , multab_80 , mask_f );
	struct ymm_x2 a1x8000x8000 = bs_gf65536_mul_0x8000_avx2( a1x8000 , multab_80 , mask_f );

	struct ymm_x4 ret;
	ret.ymm0 = a1x8000x8000.ymm0;
	ret.ymm1 = a1x8000x8000.ymm1;
	ret.ymm2 = a0x8000.ymm0^a1x8000.ymm0;
	ret.ymm3 = a0x8000.ymm1^a1x8000.ymm1;
	return ret;
}




////////////////////////////////////////////////

static inline
void gf264_mul_avx2_logtab( __m256i * r , __m256i * a , __m256i * b ,
	struct ymm_x2 logtab , struct ymm_x2 multab_80 , __m256i mul_8 , __m256i mask_f ) {

	struct ymm_x4 ab0 = gf232_mul_avx2_logtab( *(struct ymm_x4*)(a) , *(struct ymm_x4*)(b) , logtab , multab_80 , mul_8 , mask_f );
	struct ymm_x4 ab2 = gf232_mul_avx2_logtab( *(struct ymm_x4*)(a+4) , *(struct ymm_x4*)(b+4) , logtab , multab_80 , mul_8 , mask_f );

	struct ymm_x4 a0_a1;
	__m256i * ptr = (__m256i*)(&a0_a1);
	for(unsigned i=0;i<4;i++) ptr[i] = a[i]^a[4+i];
	struct ymm_x4 b0_b1;
	ptr = (__m256i*)(&b0_b1);
	for(unsigned i=0;i<4;i++) ptr[i] = b[i]^b[4+i];

	struct ymm_x4 ab1 = gf232_mul_avx2_logtab( a0_a1 , b0_b1 , logtab , multab_80 , mul_8 , mask_f );
	ab1.ymm0 ^= ab0.ymm0;
	ab1.ymm1 ^= ab0.ymm1;
	ab1.ymm2 ^= ab0.ymm2;
	ab1.ymm3 ^= ab0.ymm3;

	ptr = (__m256i*)(&ab0);
	for(unsigned i=0;i<4;i++) r[i] = ptr[i];
	ptr = (__m256i*)(&ab1);
	for(unsigned i=0;i<4;i++) r[4+i] = ptr[i];

	ab1 = gf232_mul_0x80000000_avx2(ab2,multab_80,mask_f);
	r[0] ^= ab1.ymm0;
	r[1] ^= ab1.ymm1;
	r[2] ^= ab1.ymm2;
	r[3] ^= ab1.ymm3;
}

static inline
void gf264_mul_0x80_00x7_avx2( __m256i * r , __m256i * a , struct ymm_x2 multab_80 , __m256i mask_f ) {

	struct ymm_x4 a0x8 = gf232_mul_0x80000000_avx2( *(struct ymm_x4*)(a) , multab_80 , mask_f );
	struct ymm_x4 a1x8 = gf232_mul_0x80000000_avx2( *(struct ymm_x4*)(a+4) , multab_80 , mask_f );
	struct ymm_x4 a1x8x8 = gf232_mul_0x80000000_avx2( a1x8 , multab_80 , mask_f );

	r[0] = a1x8x8.ymm0;
	r[1] = a1x8x8.ymm1;
	r[2] = a1x8x8.ymm2;
	r[3] = a1x8x8.ymm3;
	r[4] = a1x8.ymm0^a0x8.ymm0;
	r[5] = a1x8.ymm1^a0x8.ymm1;
	r[6] = a1x8.ymm2^a0x8.ymm2;
	r[7] = a1x8.ymm3^a0x8.ymm3;
}



////////////////////////////////////////////////

static inline
void gf2128_mul_avx2_logtab( __m256i * r , __m256i * a , __m256i * b ,
	struct ymm_x2 logtab , struct ymm_x2 multab_80 , __m256i mul_8 , __m256i mask_f ) {

	__m256i ab0[8];
	__m256i ab1[8];
	__m256i ab2[8];

	for(unsigned i=0;i<8;i++) ab0[i] = a[i]^a[8+i];
	for(unsigned i=0;i<8;i++) ab2[i] = b[i]^b[8+i];
	gf264_mul_avx2_logtab( ab1 , ab0 , ab2 , logtab , multab_80 , mul_8 , mask_f );

	gf264_mul_avx2_logtab( ab0 , a , b , logtab , multab_80 , mul_8 , mask_f );
	gf264_mul_avx2_logtab( ab2 , a+8 , b+8 , logtab , multab_80 , mul_8 , mask_f );

	for(unsigned i=0;i<8;i++) ab1[i]^=ab0[i];

	for(unsigned i=0;i<8;i++) r[i] = ab0[i];
	for(unsigned i=0;i<8;i++) r[8+i] = ab1[i];

	gf264_mul_0x80_00x7_avx2( ab1 , ab2 , multab_80 , mask_f );
	for(unsigned i=0;i<8;i++) r[i] ^= ab1[i];
}


static inline
void gf2128_mul_0x80_00x15_avx2( __m256i * r , __m256i * a , struct ymm_x2 multab_80 , __m256i mask_f ) {

	__m256i a0x8[8];
	__m256i a1x8[8];

	gf264_mul_0x80_00x7_avx2( a1x8 , a+8 , multab_80 , mask_f );
	gf264_mul_0x80_00x7_avx2( a0x8 , a , multab_80 , mask_f );
	gf264_mul_0x80_00x7_avx2( r , a1x8 , multab_80 , mask_f );
	for(unsigned i=0;i<8;i++) r[8+i] = a0x8[i]^a1x8[i];
}


////////////////////////////////////////////////

static inline
void gf2256_mul_avx2_logtab( __m256i * r , __m256i * a , __m256i * b ,
	struct ymm_x2 logtab , struct ymm_x2 multab_80 , __m256i mul_8 , __m256i mask_f ) {

	__m256i ab0[16];
	__m256i ab1[16];
	__m256i ab2[16];

	for(unsigned i=0;i<16;i++) ab0[i] = a[i]^a[16+i];
	for(unsigned i=0;i<16;i++) ab2[i] = b[i]^b[16+i];
	gf2128_mul_avx2_logtab( ab1 , ab0 , ab2 , logtab , multab_80 , mul_8 , mask_f );

	gf2128_mul_avx2_logtab( ab0 , a , b , logtab , multab_80 , mul_8 , mask_f );
	gf2128_mul_avx2_logtab( ab2 , a+16 , b+16 , logtab , multab_80 , mul_8 , mask_f );

	for(unsigned i=0;i<16;i++) ab1[i]^=ab0[i];

	for(unsigned i=0;i<16;i++) r[i] = ab0[i];
	for(unsigned i=0;i<16;i++) r[16+i] = ab1[i];

	gf2128_mul_0x80_00x15_avx2( ab1 , ab2 , multab_80 , mask_f );
	for(unsigned i=0;i<16;i++) r[i] ^= ab1[i];
}




#endif

