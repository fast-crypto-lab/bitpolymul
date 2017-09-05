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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "bc.h"

#include "butterfly_net.h"


#include "config_profile.h"

#define MAX_TERMS 65536

#include "gfext_aesni.h"


#ifdef _PROFILE_

#include "benchmark.h"

struct benchmark bm_ch;
struct benchmark bm_bc;
struct benchmark bm_butterfly;
struct benchmark bm_pointmul;
struct benchmark bm_pointmul_tower;

struct benchmark bm_ich;
struct benchmark bm_ibc;
struct benchmark bm_ibutterfly;

struct benchmark bm_tr;
struct benchmark bm_tr2;

#endif


#define LOG2(X) ((unsigned) (8*sizeof (unsigned long long) - __builtin_clzll((X)) - 1))

// for removing warning.
void *aligned_alloc( size_t alignment, size_t size );


void bitpolymul_simple( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned _n_64 )
{
	if( 0 == _n_64 ) return;
	unsigned n_64 = 0;
	if( 1 == _n_64 ) n_64 = _n_64;
	else {
		unsigned log_2_n64 = LOG2(_n_64);
		unsigned log_2_n64_1 = LOG2(_n_64-1);
		if( log_2_n64 == log_2_n64_1 )log_2_n64 += 1;
		n_64 = 1<<log_2_n64;
	}

	//uint64_t a_bc[MAX_TERMS] __attribute__((aligned(32)));
	//uint64_t b_bc[MAX_TERMS] __attribute__((aligned(32)));
	uint64_t * a_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	if( NULL == a_bc ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	if( NULL == b_bc ) { printf("alloc fail.\n"); exit(-1); }

#ifdef _PROFILE_SIMPLE_
bm_start(&bm_bc);
#endif
	memcpy( a_bc , a , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) a_bc[i] = 0;
	bc_to_lch( a_bc , n_64 );

	memcpy( b_bc , b , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) b_bc[i] = 0;
	bc_to_lch( b_bc , n_64 );
#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_bc);
#endif

	unsigned n_terms = 2*n_64;

	//uint64_t a_fx[4*MAX_TERMS] __attribute__((aligned(32))) = {0};
	//uint64_t b_fx[4*MAX_TERMS] __attribute__((aligned(32))) = {0};
	uint64_t * a_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*n_terms );
	if( NULL == a_fx ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*n_terms );
	if( NULL == b_fx ) { printf("alloc fail.\n"); exit(-1); }

	for(unsigned i=0;i<_n_64;i++) {
		a_fx[2*i] = a_bc[i];
		a_fx[2*i+1] = 0;
	}
	for(unsigned i=_n_64;i<n_64;i++) { a_fx[2*i]=0; a_fx[2*i+1]=0; }
	for(unsigned i=0;i<_n_64;i++) {
		b_fx[2*i] = b_bc[i];
		b_fx[2*i+1] = 0;
	}
	for(unsigned i=_n_64;i<n_64;i++) { b_fx[2*i]=0; b_fx[2*i+1]=0; }

#ifdef _PROFILE_SIMPLE_
bm_start(&bm_butterfly);
#endif
	butterfly_net_half_inp_clmul( a_fx , n_terms );
	butterfly_net_half_inp_clmul( b_fx , n_terms );

#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_butterfly);
bm_start(&bm_pointmul);
#endif
	for(unsigned i=0;i<n_terms;i++) {
		gf2ext128_mul_sse( (uint8_t *)&a_fx[i*2] , (uint8_t *)&a_fx[i*2] , (uint8_t*)& b_fx[i*2] );
	}

#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_pointmul);
bm_start(&bm_ibutterfly);
#endif

	i_butterfly_net_clmul( a_fx , n_terms );

#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_ibutterfly);
bm_start(&bm_ibc);
#endif
	bc_to_mono_128( a_fx , n_terms );

#ifdef _PROFILE_SIMPLE_
bm_stop(&bm_ibc);
#endif


	c[0] = a_fx[0];
	for(unsigned i=1;i<(2*_n_64);i++) {
		c[i] = a_fx[i*2];
		c[i] ^= a_fx[(i-1)*2+1];
	}

	free(a_bc);
	free(b_bc);
	free(a_fx);
	free(b_fx);

}




/////////////////////////////////////////////////////////////////////////////////





#include "gftower.h"

#include "gf2128_tower_iso.h"

#include "bitmat_prod.h"

#include "transpose.h"


void bitpolymul_128( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned _n_64 )
{
	if( 0 == _n_64 ) return;
	unsigned n_64 = 0;
	if( 1 == _n_64 ) n_64 = _n_64;
	else if( 32 > _n_64 ) n_64 = 32;
	else {
		unsigned log_2_n64 = LOG2(_n_64);
		unsigned log_2_n64_1 = LOG2(_n_64-1);
		if( log_2_n64 == log_2_n64_1 )log_2_n64 += 1;
		n_64 = 1<<log_2_n64;
	}

	//uint64_t a_bc[MAX_TERMS] __attribute__((aligned(32)));
	//uint64_t b_bc[MAX_TERMS] __attribute__((aligned(32)));
	uint64_t * a_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	if( NULL == a_bc ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	if( NULL == b_bc ) { printf("alloc fail.\n"); exit(-1); }

#ifdef _PROFILE_
bm_start(&bm_bc);
#endif
	memcpy( a_bc , a , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) a_bc[i] = 0;
	bc_to_lch( a_bc , n_64 );

	memcpy( b_bc , b , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) b_bc[i] = 0;
	bc_to_lch( b_bc , n_64 );
#ifdef _PROFILE_
bm_stop(&bm_bc);
#endif

	unsigned n_terms = 2*n_64;

	//uint64_t a_fx[4*MAX_TERMS] __attribute__((aligned(32))) = {0};
	//uint64_t b_fx[4*MAX_TERMS] __attribute__((aligned(32))) = {0};
	uint64_t * a_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*n_terms );
	if( NULL == a_fx ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_fx = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*2*n_terms );
	if( NULL == b_fx ) { printf("alloc fail.\n"); exit(-1); }

#ifdef _PROFILE_
bm_start(&bm_ch);
#endif
	for(unsigned i=_n_64;i<n_64;i++) { a_fx[2*i]=0; a_fx[2*i+1]=0; }
	for(unsigned i=0;i<_n_64;i+=8) {
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&a_fx[2*i]) , gf2128toTower_4R , a_bc[i] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&a_fx[2*i+2]) , gf2128toTower_4R , a_bc[i+1] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&a_fx[2*i+4]) , gf2128toTower_4R , a_bc[i+2] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&a_fx[2*i+6]) , gf2128toTower_4R , a_bc[i+3] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&a_fx[2*i+8]) , gf2128toTower_4R , a_bc[i+4] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&a_fx[2*i+10]) , gf2128toTower_4R , a_bc[i+5] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&a_fx[2*i+12]) , gf2128toTower_4R , a_bc[i+6] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&a_fx[2*i+14]) , gf2128toTower_4R , a_bc[i+7] );
		transpose_4x4_b8( (uint8_t*)(&a_fx[2*i]) , (const uint8_t *)(&a_fx[2*i]) );
	}
#ifdef _PROFILE_
//bm_stop(&bm_ch);
#endif
	//memset( a_fx + n_64*2 , 0 , sizeof(uint64_t)*(2*(n_terms-n_64)) );

#ifdef _PROFILE_
//bm_start(&bm_ch);
#endif
	for(unsigned i=_n_64;i<n_64;i++) { b_fx[2*i]=0; b_fx[2*i+1]=0; }
	for(unsigned i=0;i<_n_64;i+=8) {
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&b_fx[2*i]) , gf2128toTower_4R , b_bc[i] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&b_fx[2*i+2]) , gf2128toTower_4R , b_bc[i+1] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&b_fx[2*i+4]) , gf2128toTower_4R , b_bc[i+2] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&b_fx[2*i+6]) , gf2128toTower_4R , b_bc[i+3] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&b_fx[2*i+8]) , gf2128toTower_4R , b_bc[i+4] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&b_fx[2*i+10]) , gf2128toTower_4R , b_bc[i+5] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&b_fx[2*i+12]) , gf2128toTower_4R , b_bc[i+6] );
		bitmatrix_prod_64x128_4R_sse( (uint8_t*)(&b_fx[2*i+14]) , gf2128toTower_4R , b_bc[i+7] );
		transpose_4x4_b8( (uint8_t*)(&b_fx[2*i]) , (const uint8_t *)(&b_fx[2*i]) );
	}
#ifdef _PROFILE_
bm_stop(&bm_ch);
bm_start(&bm_butterfly);
#endif
	//memset( b_fx + n_64*2 , 0 , sizeof(uint64_t)*(2*(n_terms-n_64)) );

	butterfly_net_half_inp( a_fx , n_terms );
	butterfly_net_half_inp( b_fx , n_terms );

	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );
	struct ymm_x2 logtab;
	logtab.ymm0 = _mm256_load_si256( (__m256i*) __gf16_log );
	logtab.ymm1 = _mm256_load_si256( (__m256i*) __gf16_exp );
	__m256i mul_8 = _mm256_srli_epi16( multab_80.ymm0 , 4 );
	__m256i mask_f = _mm256_load_si256( (__m256i*) __mask_low );

#ifdef _PROFILE_
bm_stop(&bm_butterfly);
bm_start(&bm_pointmul_tower);
#endif
	for(unsigned i=0;i<n_terms;i+=32) {
		tr_16x16_b2_from_4x4_b8( (uint8_t*)(&a_fx[i*2]) , (const uint8_t *)(&a_fx[i*2]) );
		tr_16x16_b2_from_4x4_b8( (uint8_t*)(&b_fx[i*2]) , (const uint8_t *)(&b_fx[i*2]) );

		gf2128_mul_avx2_logtab( (__m256i*)&a_fx[i*2] ,  (__m256i*)&a_fx[i*2] ,  (__m256i*)&b_fx[i*2] , logtab , multab_80 , mul_8 , mask_f );

		tr_16x16_b2_from_4x4_b8( (uint8_t*)(&a_fx[i*2]) , (const uint8_t *)(&a_fx[i*2]) );
	}
#ifdef _PROFILE_
bm_stop(&bm_pointmul_tower);
bm_start(&bm_ibutterfly);
#endif

	i_butterfly_net( a_fx , n_terms );
#ifdef _PROFILE_
bm_stop(&bm_ibutterfly);
bm_start(&bm_ich);
#endif
	for(unsigned i=0;i<n_terms;i+=8) {
		transpose_4x4_b8( (uint8_t*)(&a_fx[2*i]) , (const uint8_t *)(&a_fx[2*i]) );
		bitmatrix_prod_128x128_4R_sse( (uint8_t*)(&a_fx[i*2]) , gfTowerto2128_4R , (uint8_t*)(&a_fx[i*2]) );
		bitmatrix_prod_128x128_4R_sse( (uint8_t*)(&a_fx[i*2+2]) , gfTowerto2128_4R , (uint8_t*)(&a_fx[i*2+2]) );
		bitmatrix_prod_128x128_4R_sse( (uint8_t*)(&a_fx[i*2+4]) , gfTowerto2128_4R , (uint8_t*)(&a_fx[i*2+4]) );
		bitmatrix_prod_128x128_4R_sse( (uint8_t*)(&a_fx[i*2+6]) , gfTowerto2128_4R , (uint8_t*)(&a_fx[i*2+6]) );
		bitmatrix_prod_128x128_4R_sse( (uint8_t*)(&a_fx[i*2+8]) , gfTowerto2128_4R , (uint8_t*)(&a_fx[i*2+8]) );
		bitmatrix_prod_128x128_4R_sse( (uint8_t*)(&a_fx[i*2+10]) , gfTowerto2128_4R , (uint8_t*)(&a_fx[i*2+10]) );
		bitmatrix_prod_128x128_4R_sse( (uint8_t*)(&a_fx[i*2+12]) , gfTowerto2128_4R , (uint8_t*)(&a_fx[i*2+12]) );
		bitmatrix_prod_128x128_4R_sse( (uint8_t*)(&a_fx[i*2+14]) , gfTowerto2128_4R , (uint8_t*)(&a_fx[i*2+14]) );
	}
#ifdef _PROFILE_
bm_stop(&bm_ich);
bm_start(&bm_ibc);
#endif

	bc_to_mono_128( a_fx , n_terms );
#ifdef _PROFILE_
bm_stop(&bm_ibc);
#endif

	c[0] = a_fx[0];
	for(unsigned i=1;i<(2*_n_64);i++) {
		c[i] = a_fx[i*2];
		c[i] ^= a_fx[(i-1)*2+1];
	}

	free(a_bc);
	free(b_bc);
	free(a_fx);
	free(b_fx);
}





///////////////////////////////////////////////////////////////////////////////////////////////



#include "gf2256_tower_iso.h"



void bitpolymul_256( uint64_t * _c , const uint64_t * a , const uint64_t * b , unsigned _n_64 )
{
	if( 0 == _n_64 ) return;
	unsigned n_64 = 0;
	if( 1 == _n_64 ) n_64 = _n_64;
	else if( 32 > _n_64 ) n_64 = 32;
	else {
		unsigned log_2_n64 = LOG2(_n_64);
		unsigned log_2_n64_1 = LOG2(_n_64-1);
		if( log_2_n64 == log_2_n64_1 )log_2_n64 += 1;
		n_64 = 1<<log_2_n64;
	}


	uint64_t * a_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	if( NULL == a_bc ) { printf("alloc fail.\n"); exit(-1); }
	uint64_t * b_bc = (uint64_t*)aligned_alloc( 32 , sizeof(uint64_t)*n_64 );
	if( NULL == b_bc ) { printf("alloc fail.\n"); exit(-1); }

	unsigned n_128 = n_64/2;

#ifdef _PROFILE_
bm_start(&bm_bc);
#endif
	memcpy( a_bc , a , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) a_bc[i] = 0;
	bc_to_lch_128( a_bc , n_128 );

	memcpy( b_bc , b , sizeof(uint64_t)*_n_64 );
	for(unsigned i=_n_64;i<n_64;i++) b_bc[i] = 0;
	bc_to_lch_128( b_bc , n_128 );
#ifdef _PROFILE_
bm_stop(&bm_bc);
#endif

	unsigned n_terms = 2*n_128;

	__m128i * a_fx = (__m128i*)aligned_alloc( 32 , sizeof(__m128i)*2*n_terms );
	if( NULL == a_fx ) { printf("alloc fail.\n"); exit(-1); }
	__m128i * b_fx = (__m128i*)aligned_alloc( 32 , sizeof(__m128i)*2*n_terms );
	if( NULL == b_fx ) { printf("alloc fail.\n"); exit(-1); }

	unsigned _n_128 = (_n_64+1)/2;

#ifdef _PROFILE_
bm_start(&bm_ch);
#endif
	/// XXX
	for(unsigned i=_n_128;i<n_128;i++) { a_fx[2*i]^=a_fx[2*i]; a_fx[2*i+1]^=a_fx[2*i+1]; }
	for(unsigned i=0;i<_n_128;i+=4) {
		bitmatrix_prod_128x256_4R_avx( (uint8_t*)(&a_fx[2*i]) , gf2256toTower_4R , &a_bc[2*i] );
		bitmatrix_prod_128x256_4R_avx( (uint8_t*)(&a_fx[2*i+2]) , gf2256toTower_4R , &a_bc[2*i+2] );
		bitmatrix_prod_128x256_4R_avx( (uint8_t*)(&a_fx[2*i+4]) , gf2256toTower_4R , &a_bc[2*i+4] );
		bitmatrix_prod_128x256_4R_avx( (uint8_t*)(&a_fx[2*i+6]) , gf2256toTower_4R , &a_bc[2*i+6] );
		transpose_4x4_b8( (uint8_t*)(&a_fx[2*i]) , (const uint8_t *)(&a_fx[2*i]) );
	}
#ifdef _PROFILE_
//bm_stop(&bm_ch);
#endif
	//memset( a_fx + n_64*2 , 0 , sizeof(uint64_t)*(2*(n_terms-n_64)) );

#ifdef _PROFILE_
//bm_start(&bm_ch);
#endif
	/// XXX
	for(unsigned i=_n_128;i<n_128;i++) { b_fx[2*i]^=b_fx[2*i]; b_fx[2*i+1]^=b_fx[2*i+1]; }
	for(unsigned i=0;i<_n_128;i+=4) {
		bitmatrix_prod_128x256_4R_avx( (uint8_t*)(&b_fx[2*i]) , gf2256toTower_4R , &b_bc[2*i] );
		bitmatrix_prod_128x256_4R_avx( (uint8_t*)(&b_fx[2*i+2]) , gf2256toTower_4R , &b_bc[2*i+2] );
		bitmatrix_prod_128x256_4R_avx( (uint8_t*)(&b_fx[2*i+4]) , gf2256toTower_4R , &b_bc[2*i+4] );
		bitmatrix_prod_128x256_4R_avx( (uint8_t*)(&b_fx[2*i+6]) , gf2256toTower_4R , &b_bc[2*i+6] );
		transpose_4x4_b8( (uint8_t*)(&b_fx[2*i]) , (const uint8_t *)(&b_fx[2*i]) );
	}
#ifdef _PROFILE_
bm_stop(&bm_ch);
bm_start(&bm_butterfly);
#endif
	//memset( b_fx + n_64*2 , 0 , sizeof(uint64_t)*(2*(n_terms-n_64)) );

	butterfly_net_half_inp_256( (uint64_t*)a_fx , n_terms );
	butterfly_net_half_inp_256( (uint64_t*)b_fx , n_terms );

	struct ymm_x2 multab_80 = get_multab_avx2( 0x80 );
	struct ymm_x2 logtab;
	logtab.ymm0 = _mm256_load_si256( (__m256i*) __gf16_log );
	logtab.ymm1 = _mm256_load_si256( (__m256i*) __gf16_exp );
	__m256i mul_8 = _mm256_srli_epi16( multab_80.ymm0 , 4 );
	__m256i mask_f = _mm256_load_si256( (__m256i*) __mask_low );

#ifdef _PROFILE_
bm_stop(&bm_butterfly);
bm_start(&bm_pointmul_tower);
#endif
	for(unsigned i=0;i<n_terms;i+=32) {
		tr_32x32_from_4x4_b8( (uint8_t*)(&a_fx[i*2]) , (const uint8_t *)(&a_fx[i*2]) );
		tr_32x32_from_4x4_b8( (uint8_t*)(&b_fx[i*2]) , (const uint8_t *)(&b_fx[i*2]) );

		gf2256_mul_avx2_logtab( (__m256i*)&a_fx[i*2] ,  (__m256i*)&a_fx[i*2] ,  (__m256i*)&b_fx[i*2] , logtab , multab_80 , mul_8 , mask_f );

		tr_4x4_b8_from_32x32( (uint8_t*)(&a_fx[i*2]) , (const uint8_t *)(&a_fx[i*2]) );
	}
#ifdef _PROFILE_
bm_stop(&bm_pointmul_tower);
bm_start(&bm_ibutterfly);
#endif

	i_butterfly_net_256( (uint64_t*)a_fx , n_terms );
#ifdef _PROFILE_
bm_stop(&bm_ibutterfly);
bm_start(&bm_ich);
#endif
	for(unsigned i=0;i<n_terms;i+=4) {
		transpose_4x4_b8( (uint8_t*)(&a_fx[2*i]) , (const uint8_t *)(&a_fx[2*i]) );
		bitmatrix_prod_256x256_4R_avx( (uint8_t*)(&a_fx[i*2]) , gfTowerto2256_4R , (uint64_t*)(&a_fx[i*2]) );
		bitmatrix_prod_256x256_4R_avx( (uint8_t*)(&a_fx[i*2+2]) , gfTowerto2256_4R , (uint64_t*)(&a_fx[i*2+2]) );
		bitmatrix_prod_256x256_4R_avx( (uint8_t*)(&a_fx[i*2+4]) , gfTowerto2256_4R , (uint64_t*)(&a_fx[i*2+4]) );
		bitmatrix_prod_256x256_4R_avx( (uint8_t*)(&a_fx[i*2+6]) , gfTowerto2256_4R , (uint64_t*)(&a_fx[i*2+6]) );
	}
#ifdef _PROFILE_
bm_stop(&bm_ich);
bm_start(&bm_ibc);
#endif

	bc_to_mono_256( (uint64_t*)a_fx , n_terms );
#ifdef _PROFILE_
bm_stop(&bm_ibc);
#endif


	__m128i * c = (__m128i*)_c;
	c[0] = a_fx[0];
	for(unsigned i=1;i<(2*_n_128);i++) {
		c[i] = a_fx[i*2];
		c[i] ^= a_fx[(i-1)*2+1];
	}

	free(a_bc);
	free(b_bc);
	free(a_fx);
	free(b_fx);
}








