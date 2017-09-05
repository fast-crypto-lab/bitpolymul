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

#ifndef _TRANSPOSE_H_
#define _TRANSPOSE_H_

#include <stdint.h>
#include <immintrin.h>


static inline
void tr_ref_4x4_b8( uint8_t * _r , const uint8_t * a ) {
	uint8_t r[4*4*8] __attribute__((aligned(32)));
	for(unsigned j=0;j<4;j++) {
		for(unsigned i=0;i<8;i++) {
			r[j*32+i*4+0] = a[i*4+j+0];
			r[j*32+i*4+1] = a[i*4+j+32];
			r[j*32+i*4+2] = a[i*4+j+32*2];
			r[j*32+i*4+3] = a[i*4+j+32*3];
		}
	}
	for(unsigned i=0;i<4*4*8;i++) _r[i] = r[i];
}


static uint8_t _tr_4x4[32] __attribute__((aligned(32))) = {0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15 ,0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15};

static inline
void tr_avx2_4x4_b8( uint8_t * _r , const uint8_t * a ) {

	__m256i r0 = _mm256_load_si256( (__m256i*) a );
	__m256i r1 = _mm256_load_si256( (__m256i*) (a+32) );
	__m256i r2 = _mm256_load_si256( (__m256i*) (a+64) );
	__m256i r3 = _mm256_load_si256( (__m256i*) (a+96) );

	__m256i t0 = _mm256_shuffle_epi8( r0 , *(__m256i*)_tr_4x4 ); // 00,01,02,03
	__m256i t1 = _mm256_shuffle_epi8( r1 , *(__m256i*)_tr_4x4 ); // 10,11,12,13
	__m256i t2 = _mm256_shuffle_epi8( r2 , *(__m256i*)_tr_4x4 ); // 20,21,22,23
	__m256i t3 = _mm256_shuffle_epi8( r3 , *(__m256i*)_tr_4x4 ); // 30,31,32,33

	__m256i t01lo = _mm256_unpacklo_epi32( t0 , t1 ); // 00,10,01,11
	__m256i t01hi = _mm256_unpackhi_epi32( t0 , t1 ); // 02,12,03,13
	__m256i t23lo = _mm256_unpacklo_epi32( t2 , t3 ); // 20,30,21,31
	__m256i t23hi = _mm256_unpackhi_epi32( t2 , t3 ); // 22,32,23,33

	__m256i r01lo = _mm256_unpacklo_epi64( t01lo , t23lo ); // 00,10,20,30
	__m256i r01hi = _mm256_unpackhi_epi64( t01lo , t23lo ); // 01,11,21,31
	__m256i r23lo = _mm256_unpacklo_epi64( t01hi , t23hi ); // 02,12,22,32
	__m256i r23hi = _mm256_unpackhi_epi64( t01hi , t23hi ); // 03,13,23,33

	__m256i s0 = _mm256_shuffle_epi8( r01lo , *(__m256i*)_tr_4x4 );
	__m256i s1 = _mm256_shuffle_epi8( r01hi , *(__m256i*)_tr_4x4 );
	__m256i s2 = _mm256_shuffle_epi8( r23lo , *(__m256i*)_tr_4x4 );
	__m256i s3 = _mm256_shuffle_epi8( r23hi , *(__m256i*)_tr_4x4 );

	_mm256_store_si256( (__m256i*) _r , s0 );
	_mm256_store_si256( (__m256i*) (_r+32) , s1 );
	_mm256_store_si256( (__m256i*) (_r+64) , s2 );
	_mm256_store_si256( (__m256i*) (_r+96) , s3 );
}

static inline
void transpose_4x4_b8( uint8_t * r , const uint8_t * a ) {
	tr_avx2_4x4_b8(r,a);
}


static inline
void tr_16x16_b2_from_4x4_b8_1_4( uint8_t * r , const uint8_t * a ) {

	__m256i a0 = _mm256_load_si256( (__m256i*) a );         // 00,04,08,0c
	__m256i a4 = _mm256_load_si256( (__m256i*) (a+32*4) );  // 10,14,18,1c
	__m256i a8 = _mm256_load_si256( (__m256i*) (a+32*8) );  // 20,24,28,2c
	__m256i ac = _mm256_load_si256( (__m256i*) (a+32*12) ); // 30,34,38,3c

	__m256i a04l = _mm256_unpacklo_epi32( a0 , a4 ); // 00,10,04,14
	__m256i a04h = _mm256_unpackhi_epi32( a0 , a4 ); // 08,18,0c,1c
	__m256i a8cl = _mm256_unpacklo_epi32( a8 , ac ); // 20,30,24,34
	__m256i a8ch = _mm256_unpackhi_epi32( a8 , ac ); // 28,38,2c,3c

	__m256i b0 = _mm256_unpacklo_epi64( a04l , a8cl ); // 00,10,20,30
	__m256i b4 = _mm256_unpackhi_epi64( a04l , a8cl ); // 04,14,24,34
	__m256i b8 = _mm256_unpacklo_epi64( a04h , a8ch ); // 08,18,28,38
	__m256i bc = _mm256_unpackhi_epi64( a04h , a8ch ); // 0c,1c,2c,3c

	_mm256_store_si256( (__m256i*) (r+32*0) , b0 );
	_mm256_store_si256( (__m256i*) (r+32*4) , b4 );
	_mm256_store_si256( (__m256i*) (r+32*8) , b8 );
	_mm256_store_si256( (__m256i*) (r+32*12) , bc );
}

static inline
void tr_16x16_b2_from_4x4_b8( uint8_t * r , const uint8_t * a ) {
	tr_16x16_b2_from_4x4_b8_1_4( r , a );
	tr_16x16_b2_from_4x4_b8_1_4( r+32 , a+32 );
	tr_16x16_b2_from_4x4_b8_1_4( r+32*2 , a+32*2 );
	tr_16x16_b2_from_4x4_b8_1_4( r+32*3 , a+32*3 );
}

/////////////////////////////////////


static inline
void tr_32x32_from_4x4_b8_1_4( uint8_t * r , const uint8_t * a )
{
	__m256i a0 = _mm256_load_si256( (__m256i*) a );         // 00,04,08,0c
	__m256i a4 = _mm256_load_si256( (__m256i*) (a+32*4) );  // 10,14,18,1c
	__m256i a8 = _mm256_load_si256( (__m256i*) (a+32*8) );  // 20,24,28,2c
	__m256i ac = _mm256_load_si256( (__m256i*) (a+32*12) ); // 30,34,38,3c
	__m256i a10 = _mm256_load_si256( (__m256i*) (a+32*16) );
	__m256i a14 = _mm256_load_si256( (__m256i*) (a+32*20) );
	__m256i a18 = _mm256_load_si256( (__m256i*) (a+32*24) );
	__m256i a1c = _mm256_load_si256( (__m256i*) (a+32*28) );

	__m256i a04l = _mm256_unpacklo_epi32( a0 , a4 ); // 00,10,04,14
	__m256i a04h = _mm256_unpackhi_epi32( a0 , a4 ); // 08,18,0c,1c
	__m256i a8cl = _mm256_unpacklo_epi32( a8 , ac ); // 20,30,24,34
	__m256i a8ch = _mm256_unpackhi_epi32( a8 , ac ); // 28,38,2c,3c
	__m256i a1014l = _mm256_unpacklo_epi32( a10 , a14 );
	__m256i a1014h = _mm256_unpackhi_epi32( a10 , a14 );
	__m256i a181cl = _mm256_unpacklo_epi32( a18 , a1c );
	__m256i a181ch = _mm256_unpackhi_epi32( a18 , a1c );

	__m256i b0 = _mm256_unpacklo_epi64( a04l , a8cl ); // 00,10,20,30
	__m256i b4 = _mm256_unpackhi_epi64( a04l , a8cl ); // 04,14,24,34
	__m256i b8 = _mm256_unpacklo_epi64( a04h , a8ch ); // 08,18,28,38
	__m256i bc = _mm256_unpackhi_epi64( a04h , a8ch ); // 0c,1c,2c,3c
	__m256i b10 = _mm256_unpacklo_epi64( a1014l , a181cl );
	__m256i b14 = _mm256_unpackhi_epi64( a1014l , a181cl );
	__m256i b18 = _mm256_unpacklo_epi64( a1014h , a181ch );
	__m256i b1c = _mm256_unpackhi_epi64( a1014h , a181ch );

	__m256i c0 = _mm256_permute2x128_si256( b0 , b10 , 0x20 );
	__m256i c10 = _mm256_permute2x128_si256( b0 , b10 , 0x31 );
	__m256i c4 = _mm256_permute2x128_si256( b4 , b14 , 0x20 );
	__m256i c14 = _mm256_permute2x128_si256( b4 , b14 , 0x31 );
	__m256i c8 = _mm256_permute2x128_si256( b8 , b18 , 0x20 );
	__m256i c18 = _mm256_permute2x128_si256( b8 , b18 , 0x31 );
	__m256i cc = _mm256_permute2x128_si256( bc , b1c , 0x20 );
	__m256i c1c = _mm256_permute2x128_si256( bc , b1c , 0x31 );

	_mm256_store_si256( (__m256i*) (r+32*0) , c0 );
	_mm256_store_si256( (__m256i*) (r+32*4) , c4 );
	_mm256_store_si256( (__m256i*) (r+32*8) , c8 );
	_mm256_store_si256( (__m256i*) (r+32*12) , cc );
	_mm256_store_si256( (__m256i*) (r+32*16) , c10 );
	_mm256_store_si256( (__m256i*) (r+32*20) , c14 );
	_mm256_store_si256( (__m256i*) (r+32*24) , c18 );
	_mm256_store_si256( (__m256i*) (r+32*28) , c1c );
}

static inline
void tr_32x32_from_4x4_b8( uint8_t * r , const uint8_t * a ) {
	tr_32x32_from_4x4_b8_1_4( r , a );
	tr_32x32_from_4x4_b8_1_4( r+32 , a+32 );
	tr_32x32_from_4x4_b8_1_4( r+32*2 , a+32*2 );
	tr_32x32_from_4x4_b8_1_4( r+32*3 , a+32*3 );
}

/////////////////////////////////


static inline
void tr_4x4_b8_from_32x32_1_4( uint8_t * r , const uint8_t * a )
{
	__m256i b0 = _mm256_load_si256( (__m256i*) a );         // 00,04,08,0c
	__m256i b4 = _mm256_load_si256( (__m256i*) (a+32*4) );  // 10,14,18,1c
	__m256i b8 = _mm256_load_si256( (__m256i*) (a+32*8) );  // 20,24,28,2c
	__m256i bc = _mm256_load_si256( (__m256i*) (a+32*12) ); // 30,34,38,3c
	__m256i b10 = _mm256_load_si256( (__m256i*) (a+32*16) );
	__m256i b14 = _mm256_load_si256( (__m256i*) (a+32*20) );
	__m256i b18 = _mm256_load_si256( (__m256i*) (a+32*24) );
	__m256i b1c = _mm256_load_si256( (__m256i*) (a+32*28) );

	__m256i a0 = _mm256_permute2x128_si256( b0 , b10 , 0x20 );
	__m256i a10 = _mm256_permute2x128_si256( b0 , b10 , 0x31 );
	__m256i a4 = _mm256_permute2x128_si256( b4 , b14 , 0x20 );
	__m256i a14 = _mm256_permute2x128_si256( b4 , b14 , 0x31 );
	__m256i a8 = _mm256_permute2x128_si256( b8 , b18 , 0x20 );
	__m256i a18 = _mm256_permute2x128_si256( b8 , b18 , 0x31 );
	__m256i ac = _mm256_permute2x128_si256( bc , b1c , 0x20 );
	__m256i a1c = _mm256_permute2x128_si256( bc , b1c , 0x31 );

	__m256i a04l = _mm256_unpacklo_epi32( a0 , a4 ); // 00,10,04,14
	__m256i a04h = _mm256_unpackhi_epi32( a0 , a4 ); // 08,18,0c,1c
	__m256i a8cl = _mm256_unpacklo_epi32( a8 , ac ); // 20,30,24,34
	__m256i a8ch = _mm256_unpackhi_epi32( a8 , ac ); // 28,38,2c,3c
	__m256i a1014l = _mm256_unpacklo_epi32( a10 , a14 );
	__m256i a1014h = _mm256_unpackhi_epi32( a10 , a14 );
	__m256i a181cl = _mm256_unpacklo_epi32( a18 , a1c );
	__m256i a181ch = _mm256_unpackhi_epi32( a18 , a1c );

	__m256i k0 = _mm256_unpacklo_epi64( a04l , a8cl ); // 00,10,20,30
	__m256i k4 = _mm256_unpackhi_epi64( a04l , a8cl ); // 04,14,24,34
	__m256i k8 = _mm256_unpacklo_epi64( a04h , a8ch ); // 08,18,28,38
	__m256i kc = _mm256_unpackhi_epi64( a04h , a8ch ); // 0c,1c,2c,3c
	__m256i k10 = _mm256_unpacklo_epi64( a1014l , a181cl );
	__m256i k14 = _mm256_unpackhi_epi64( a1014l , a181cl );
	__m256i k18 = _mm256_unpacklo_epi64( a1014h , a181ch );
	__m256i k1c = _mm256_unpackhi_epi64( a1014h , a181ch );

	_mm256_store_si256( (__m256i*) (r+32*0) , k0 );
	_mm256_store_si256( (__m256i*) (r+32*4) , k4 );
	_mm256_store_si256( (__m256i*) (r+32*8) , k8 );
	_mm256_store_si256( (__m256i*) (r+32*12) , kc );
	_mm256_store_si256( (__m256i*) (r+32*16) , k10 );
	_mm256_store_si256( (__m256i*) (r+32*20) , k14 );
	_mm256_store_si256( (__m256i*) (r+32*24) , k18 );
	_mm256_store_si256( (__m256i*) (r+32*28) , k1c );
}


static inline
void tr_4x4_b8_from_32x32( uint8_t * r , const uint8_t * a ) {
	tr_4x4_b8_from_32x32_1_4( r , a );
	tr_4x4_b8_from_32x32_1_4( r+32 , a+32 );
	tr_4x4_b8_from_32x32_1_4( r+32*2 , a+32*2 );
	tr_4x4_b8_from_32x32_1_4( r+32*3 , a+32*3 );
}



//////////////////////////////////////

static inline
void tr_ref_16x16( uint8_t * _r , const uint8_t * a ) {
        uint8_t r[16*16] __attribute__((aligned(32)));
        for(unsigned j=0;j<16;j++)
                for(unsigned k=0;k<16;k++) r[j*16+k] = a[k*16+j];
        for(unsigned i=0;i<16*16;i++) _r[i] = r[i];
}

//////////////////////////////////////

static uint64_t gath_ref_16x32[4] __attribute__((aligned(32))) = {0,1*8,16*16,16*16+8 };

static inline
void tr_ref_16x32( uint8_t * r , const uint8_t * a )
{
        uint8_t tr2[16*16*2] __attribute__((aligned(32)));
        tr_ref_16x16( tr2 , a );
        tr_ref_16x16( tr2+(16*16) , a+(16*16) );

        _mm256_store_si256( (__m256i*)(r) , _mm256_i64gather_epi64( (long long const*)tr2 , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+32) , _mm256_i64gather_epi64( (long long const*)(tr2+16) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+64) , _mm256_i64gather_epi64( (long long const*)(tr2+32) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+96) , _mm256_i64gather_epi64( (long long const*)(tr2+48) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+128) , _mm256_i64gather_epi64( (long long const*)(tr2+64) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+160) , _mm256_i64gather_epi64( (long long const*)(tr2+80) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+192) , _mm256_i64gather_epi64( (long long const*)(tr2+96) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+224) , _mm256_i64gather_epi64( (long long const*)(tr2+112) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+256) , _mm256_i64gather_epi64( (long long const*)(tr2+128) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+288) , _mm256_i64gather_epi64( (long long const*)(tr2+144) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+320) , _mm256_i64gather_epi64( (long long const*)(tr2+160) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+352) , _mm256_i64gather_epi64( (long long const*)(tr2+176) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+384) , _mm256_i64gather_epi64( (long long const*)(tr2+192) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+416) , _mm256_i64gather_epi64( (long long const*)(tr2+208) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+448) , _mm256_i64gather_epi64( (long long const*)(tr2+224) , *(__m256i*)gath_ref_16x32 , 1 ) );
        _mm256_store_si256( (__m256i*)(r+480) , _mm256_i64gather_epi64( (long long const*)(tr2+240) , *(__m256i*)gath_ref_16x32 , 1 ) );
}

static inline
void transpose_16x32( uint8_t * r , const uint8_t * a )
{
        tr_ref_16x32( r , a );
}

#if 0
static uint64_t gath_32x16[4] __attribute__((aligned(32))) = {0, 8 , 32 , 40};
#endif

static inline
void transpose_32x16( uint8_t * r , const uint8_t * a )
{
#if 1
        uint8_t tr2[16*16*2] __attribute__((aligned(32)));
        for(unsigned j=0;j<32;j++)
                for(unsigned k=0;k<16;k++) tr2[j*16+k] = a[k*32+j];
        for(unsigned i=0;i<32*16;i++) r[i] = tr2[i];
#else
        uint8_t tr2[16*16*2] __attribute__((aligned(32)));
        _mm256_store_si256( (__m256i*)(tr2+0 ) , _mm256_i64gather_epi64( (long long const*)(a  + 0) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+32) , _mm256_i64gather_epi64( (long long const*)(a  + 16) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+64) , _mm256_i64gather_epi64( (long long const*)(a  + 32) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+96) , _mm256_i64gather_epi64( (long long const*)(a  + 48) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+128) , _mm256_i64gather_epi64( (long long const*)(a  + 64) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+160) , _mm256_i64gather_epi64( (long long const*)(a  + 80) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+192) , _mm256_i64gather_epi64( (long long const*)(a  + 96) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+224) , _mm256_i64gather_epi64( (long long const*)(a  + 112) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+256) , _mm256_i64gather_epi64( (long long const*)(a  + 128) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+288) , _mm256_i64gather_epi64( (long long const*)(a  + 144) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+320) , _mm256_i64gather_epi64( (long long const*)(a  + 160) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+352) , _mm256_i64gather_epi64( (long long const*)(a  + 176) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+384) , _mm256_i64gather_epi64( (long long const*)(a  + 192) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+416) , _mm256_i64gather_epi64( (long long const*)(a  + 208) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+448) , _mm256_i64gather_epi64( (long long const*)(a  + 224) , *(__m256i*)gath_32x16 , 1 ) );
        _mm256_store_si256( (__m256i*)(tr2+480) , _mm256_i64gather_epi64( (long long const*)(a  + 240) , *(__m256i*)gath_32x16 , 1 ) );

        tr_ref_16x16( r , tr2 );
        tr_ref_16x16( r+16*16 , tr2+16*16 );
#endif
}



#endif


