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

#ifndef _GF_EXT_AESNI_H_
#define _GF_EXT_AESNI_H_


#include <stdint.h>

//#include <emmintrin.h>
//#include <tmmintrin.h>
#include <immintrin.h>



/// X^128 + X^7 + X^2 + X + 1
/// 0x       8       7
static const uint64_t _gf2ext128_reducer[2] __attribute__((aligned(16)))  = {0x87ULL,0x0ULL};

static inline
__m128i _gf2ext128_reduce_sse( __m128i x0 , __m128i x128 )
{
	__m128i reducer = _mm_load_si128( (__m128i const*)_gf2ext128_reducer );
	//__m128i *reducer = (__m128i *)_gf2ext128_reducer;
	__m128i x64 = _mm_clmulepi64_si128( x128 , reducer , 1 );  /// 0_32 , xx2_32 , xx1 , xx0
	x128 ^= _mm_shuffle_epi32( x64 , 0xfe ); // 0,0,0,xx2 ; 0xfe --> 3,3,3,2
	x0 ^= _mm_shuffle_epi32( x64 , 0x4f ); // xx1 , xx0 , 0 , 0   ;  0x4f -->  1,0,3,3  --> xx1,xx0,0,0
	x0 ^= _mm_clmulepi64_si128( x128 , reducer , 0 );
	return x0;
}

static inline
__m128i _gf_aesgcm_reduce_sse( __m128i x0 , __m128i x128 )
{
	//__m128i *mask_32 = (__m128i*) _low_32bit_on;
	__m128i mask_32 = _mm_setr_epi32(0xffffffff,0,0,0);
	__m128i tmp = _mm_srli_epi32(x128,31) ^ _mm_srli_epi32(x128,30) ^ _mm_srli_epi32(x128,25);

	__m128i tmp_rol_32 = _mm_shuffle_epi32(tmp,0x93);
	x128 ^= (mask_32)&tmp_rol_32;
	x0 ^= _mm_andnot_si128( mask_32 , tmp_rol_32 );

	x0 ^= x128 ^ _mm_slli_epi32( x128 , 1 ) ^ _mm_slli_epi32( x128 , 2 ) ^_mm_slli_epi32( x128 , 7 );
	return x0;
}

#define _MUL_128( c0,c2,a0,b0 ) \
do {\
  __m128i tt = _mm_clmulepi64_si128( a0,b0 , 0x01 ); \
  c0 = _mm_clmulepi64_si128( a0,b0, 0 ); \
  c2 = _mm_clmulepi64_si128( a0,b0, 0x11 ); \
  tt ^= _mm_clmulepi64_si128( a0,b0 , 0x10 ); \
  c0 ^= _mm_slli_si128( tt , 8 ); \
  c2 ^= _mm_srli_si128( tt , 8 ); \
} while(0)

#define _MUL_128_KARATSUBA( c0,c1,a0,b0 ) \
do {\
  c0 = _mm_clmulepi64_si128( a0,b0 , 0x00 ); \
  c1 = _mm_clmulepi64_si128( a0,b0 , 0x11 ); \
  __m128i _tt0 = a0^_mm_srli_si128(a0,8); \
  __m128i _tt1 = b0^_mm_srli_si128(b0,8); \
  _tt0 = _mm_clmulepi64_si128( _tt0, _tt1 , 0 )^c0^c1; \
  c0 ^= _mm_slli_si128( _tt0 , 8 ); \
  c1 ^= _mm_srli_si128( _tt0 , 8 ); \
} while(0)


static inline
void gf2ext128_mul_sse( uint8_t * c , const uint8_t * a , const uint8_t * b )
{
	__m128i a0 = _mm_load_si128( (__m128i const *)a );
	__m128i b0 = _mm_load_si128( (__m128i const *)b );
	__m128i c0,c128;
	_MUL_128_KARATSUBA( c0,c128, a0,b0 );

	__m128i c3 = _gf2ext128_reduce_sse( c0 , c128 );
	//__m128i c3 = _gf_aesgcm_reduce_sse( c0 , c128 );
	_mm_store_si128((__m128i*) c , c3 );
}




#endif

