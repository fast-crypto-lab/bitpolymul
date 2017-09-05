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

#include "bc.h"



static inline
unsigned get_num_blocks( unsigned poly_len , unsigned blk_size ) {
	return poly_len/blk_size;
}


static inline
unsigned deg_si( unsigned si ) {
	return (1<<si);
}

static inline
unsigned get_si_2_pow( unsigned si , unsigned deg ) {
	unsigned si_deg = (1<<si);
	unsigned r=1;
	while( (si_deg<<r) < deg ) {
		r += 1;
	}
	return (1<<(r-1));
}

static inline
unsigned get_max_si( unsigned deg ) {
	unsigned si = 0;
	unsigned si_attempt = 1;
	uint64_t deg64 = deg;
	while( deg64 > ((1ULL)<<si_attempt) ) {
		si = si_attempt;
		si_attempt <<= 1;
	}
	return si;
}


//////////////////////////////////////////////////////////////////////


//#include <x86intrin.h>
#include <emmintrin.h>
#include <immintrin.h>




static inline
void xor_down( bc_sto_t * poly , unsigned st , unsigned len , unsigned diff )
{
#if 0
	for( unsigned i=0;i<len;i++) {
		poly[st-i-1] ^= poly[st-i-1+diff];
	}
#else
	while( ((unsigned long)(poly+st)) & 31 ) {
		poly[st-1] ^= poly[st-1+diff];
		st--;
		len--;
		if( 0 == len ) break;
	}
	__m256i * poly256 = (__m256i*)(poly+st);
	unsigned _len = len>>2;
	for( unsigned i=0;i<_len;i++ ) {
		*(poly256-i-1) ^= _mm256_loadu_si256( (__m256i*)(poly+st+diff-(i*4)-4) );
	}
	for( unsigned i=(_len<<2);i<len;i++) poly[st-i-1] ^= poly[st-i-1+diff];
#endif
}

static inline
void poly_div( bc_sto_t * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;
#if 1
	xor_down( poly , (deg_blk-deg_diff+1)*blk_size , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
#else
	for(unsigned i=deg_blk;i>=si_degree;i--) {
		for(int j=((int)blk_size)-1;j>=0;j--) {
			poly[(i-deg_diff)*blk_size+j] ^= poly[i*blk_size+j];
		}
	}
#endif
}

static inline
void represent_in_si( bc_sto_t * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

#if 1
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	while( 0 < pow ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			poly_div( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow >>= 1;
	}
#else
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	poly_div( poly , n_terms , blk_size , si , pow );
	if( 1 < pow ) {
		represent_in_si( poly , pow*deg_si(si)*blk_size , blk_size , si );
		represent_in_si( poly + pow*deg_si(si)*blk_size , n_terms - pow*deg_si(si)*blk_size , blk_size , si );
	}
#endif
}


void _bc_to_lch( bc_sto_t * poly , unsigned n_terms , unsigned blk_size )
{
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );
	represent_in_si( poly , n_terms , blk_size , si );

	unsigned new_blk_size = deg_si(si)*blk_size;
	_bc_to_lch( poly , n_terms , new_blk_size );
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_lch( poly + i , new_blk_size , blk_size );
	}
}


void bc_to_lch( bc_sto_t * poly , unsigned n_terms )
{
	_bc_to_lch( poly , n_terms , 1 );
}



/////////////////////////////////////


static inline
void xor_up( bc_sto_t * poly , unsigned st , unsigned len , unsigned diff )
{
#if 0
	for( unsigned i=0;i<len;i++) {
		poly[st+i] ^= poly[st+i+diff];
	}
#else
	while( ((unsigned long)(poly+st)) & 31 ) {
		poly[st] ^= poly[st+diff];
		st++;
		len--;
		if( 0 == len ) break;
	}
	__m256i * poly256 = (__m256i*)(poly+st);
	unsigned _len = len>>2;
	for( unsigned i=0;i<_len;i++ ) {
		poly256[i] ^= _mm256_loadu_si256( (__m256i*)(poly+st+diff+(i*4)) );
	}
	for( unsigned i=(_len<<2);i<len;i++) poly[st+i] ^= poly[st+i+diff];
#endif
}


static inline
void i_poly_div( bc_sto_t * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;
#if 1
	xor_up( poly , (blk_size)*(si_degree-deg_diff) , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
#else
	for(unsigned i=si_degree;i<=deg_blk;i++) {
		for(unsigned j=0; j<blk_size ;j++) {
			poly[(i-deg_diff)*blk_size+j] ^= poly[i*blk_size+j];
		}
	}
#endif
}

static inline
void i_represent_in_si( bc_sto_t * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

	unsigned pow = 1;
	while( pow*deg_si(si) <= degree_in_blocks ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			i_poly_div( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow *= 2;

	}
}


void _bc_to_mono( bc_sto_t * poly , unsigned n_terms , unsigned blk_size )
{
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );


	unsigned new_blk_size = deg_si(si)*blk_size;
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_mono( poly + i , new_blk_size , blk_size );
	}
	_bc_to_mono( poly , n_terms , new_blk_size );
	i_represent_in_si( poly , n_terms , blk_size , si );
}


void bc_to_mono( bc_sto_t * poly , unsigned n_terms )
{
	_bc_to_mono( poly , n_terms , 1 );
}





//////////////////////////////////////////////


static inline
void xor_down_128( __m128i * poly , unsigned st , unsigned len , unsigned diff )
{
#if 0
	for( unsigned i=0;i<len;i++) {
		poly[st-i-1] ^= poly[st-i-1+diff];
	}
#else
	if( ((unsigned long)(poly+st)) & 31 ) {
		poly[st-1] ^= poly[st+diff-1];
		st--;
		len--;
	}
	__m256i * poly256 = (__m256i*)(poly+st);
	unsigned _len = len>>1;
	for( unsigned i=0;i<_len;i++ ) {
		*(poly256 - i-1) ^= _mm256_loadu_si256( (__m256i*)(poly+st+diff-(i*2)-2) );
	}
	if( len&1 ) {
		poly[st-len] ^= poly[st-len+diff];
	}
#endif
}



static inline
void poly_div_128( __m128i * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;

	xor_down_128( poly , (deg_blk-deg_diff+1)*blk_size , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
}

static inline
void represent_in_si_128( __m128i * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

#if 1
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	while( 0 < pow ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			poly_div_128( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow >>= 1;
	}
#else
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	poly_div( poly , n_terms , blk_size , si , pow );
	if( 1 < pow ) {
		represent_in_si( poly , pow*deg_si(si)*blk_size , blk_size , si );
		represent_in_si( poly + pow*deg_si(si)*blk_size , n_terms - pow*deg_si(si)*blk_size , blk_size , si );
	}
#endif
}


void _bc_to_lch_128( __m128i * poly , unsigned n_terms , unsigned blk_size )
{
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );
	represent_in_si_128( poly , n_terms , blk_size , si );

	unsigned new_blk_size = deg_si(si)*blk_size;
	_bc_to_lch_128( poly , n_terms , new_blk_size );
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_lch_128( poly + i , new_blk_size , blk_size );
	}
}


void bc_to_lch_128( bc_sto_t * poly , unsigned n_terms )
{
	_bc_to_lch_128( (__m128i*) poly , n_terms , 1 );
}


///////////////////////////////////


static inline
void xor_up_128( __m128i * poly , unsigned st , unsigned len , unsigned diff )
{
#if 0
	for( unsigned i=0;i<len;i++) {
		poly[st+i] ^= poly[st+i+diff];
	}
#else
	if( ((unsigned long)(poly+st)) & 31 ) {
		poly[st] ^= poly[st+diff];
		st++;
		len--;
	}
	__m256i * poly256 = (__m256i*)(poly+st);
	unsigned _len = len>>1;
	for( unsigned i=0;i<_len;i++ ) {
		poly256[i] ^= _mm256_loadu_si256( (__m256i*)(poly+st+diff+(i*2)) );
	}
	if( len&1 ) {
		poly[st+len-1] ^= poly[st+len-1+diff];
	}
#endif
}


static inline
void i_poly_div_128( __m128i * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;

#if 1
	xor_up_128( poly , blk_size*(si_degree-deg_diff) , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
#else
	for(unsigned i=si_degree;i<=deg_blk;i++) {
		for(unsigned j=0; j<blk_size ;j++) {
			poly[(i-deg_diff)*blk_size+j] ^= poly[i*blk_size+j];
		}
	}
#endif
}

static inline
void i_represent_in_si_128( __m128i * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

	unsigned pow = 1;
	while( pow*deg_si(si) <= degree_in_blocks ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			i_poly_div_128( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow *= 2;
	}
}


void _bc_to_mono_128( __m128i * poly , unsigned n_terms , unsigned blk_size )
{

//printf("ibc: %d/%d\n", n_terms , blk_size );

	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;

//printf("deg: %d\n", degree_in_blocks);
	unsigned si = get_max_si( degree_in_blocks );
//printf("si: %d\n",si);

	unsigned new_blk_size = deg_si(si)*blk_size;
//printf("new blksize: %d\n", new_blk_size);
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_mono_128( poly + i , new_blk_size , blk_size );
	}
	_bc_to_mono_128( poly , n_terms , new_blk_size );
	i_represent_in_si_128( poly , n_terms , blk_size , si );
}


void bc_to_mono_128( bc_sto_t * poly , unsigned n_terms )
{

	_bc_to_mono_128( (__m128i*)poly , n_terms , 1 );
}





//////////////////////////////////////////////






static inline
void xor_down_256( __m256i * poly , unsigned st , unsigned len , unsigned diff )
{
#if 1
	for( unsigned i=0;i<len;i++) {
		poly[st-i-1] ^= poly[st-i-1+diff];
	}
#else
	if( ((unsigned long)(poly+st)) & 31 ) {
		poly[st-1] ^= poly[st+diff-1];
		st--;
		len--;
	}
	__m256i * poly256 = (__m256i*)(poly+st);
	unsigned _len = len>>1;
	for( unsigned i=0;i<_len;i++ ) {
		*(poly256 - i-1) ^= _mm256_loadu_si256( (__m256i*)(poly+st+diff-(i*2)-2) );
	}
	if( len&1 ) {
		poly[st-len] ^= poly[st-len+diff];
	}
#endif
}



static inline
void poly_div_256( __m256i * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;

	xor_down_256( poly , (deg_blk-deg_diff+1)*blk_size , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
}

static inline
void represent_in_si_256( __m256i * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

#if 1
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	while( 0 < pow ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			poly_div_256( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow >>= 1;
	}
#else
	unsigned pow = get_si_2_pow( si , degree_in_blocks );
	poly_div( poly , n_terms , blk_size , si , pow );
	if( 1 < pow ) {
		represent_in_si( poly , pow*deg_si(si)*blk_size , blk_size , si );
		represent_in_si( poly + pow*deg_si(si)*blk_size , n_terms - pow*deg_si(si)*blk_size , blk_size , si );
	}
#endif
}


void _bc_to_lch_256( __m256i * poly , unsigned n_terms , unsigned blk_size )
{
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned si = get_max_si( degree_in_blocks );
	represent_in_si_256( poly , n_terms , blk_size , si );

	unsigned new_blk_size = deg_si(si)*blk_size;
	_bc_to_lch_256( poly , n_terms , new_blk_size );
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_lch_256( poly + i , new_blk_size , blk_size );
	}
}


void bc_to_lch_256( bc_sto_t * poly , unsigned n_terms )
{
	_bc_to_lch_256( (__m256i*) poly , n_terms , 1 );
}


///////////////////////////////////


static inline
void xor_up_256( __m256i * poly , unsigned st , unsigned len , unsigned diff )
{
#if 1
	for( unsigned i=0;i<len;i++) {
		poly[st+i] ^= poly[st+i+diff];
	}
#else
	if( ((unsigned long)(poly+st)) & 31 ) {
		poly[st] ^= poly[st+diff];
		st++;
		len--;
	}
	__m256i * poly256 = (__m256i*)(poly+st);
	unsigned _len = len>>1;
	for( unsigned i=0;i<_len;i++ ) {
		poly256[i] ^= _mm256_loadu_si256( (__m256i*)(poly+st+diff+(i*2)) );
	}
	if( len&1 ) {
		poly[st+len-1] ^= poly[st+len-1+diff];
	}
#endif
}


static inline
void i_poly_div_256( __m256i * poly , unsigned n_terms , unsigned blk_size , unsigned si , unsigned pow )
{
	if( 0 == si ) return;
	unsigned si_degree = deg_si(si)*pow;
	unsigned deg_diff = si_degree - pow;
	unsigned deg_blk = get_num_blocks( n_terms , blk_size ) -1;

#if 1
	xor_up_256( poly , blk_size*(si_degree-deg_diff) , (deg_blk-si_degree+1)*blk_size , deg_diff*blk_size );
#else
	for(unsigned i=si_degree;i<=deg_blk;i++) {
		for(unsigned j=0; j<blk_size ;j++) {
			poly[(i-deg_diff)*blk_size+j] ^= poly[i*blk_size+j];
		}
	}
#endif
}

static inline
void i_represent_in_si_256( __m256i * poly , unsigned n_terms , unsigned blk_size , unsigned si )
{
	if( 0 == si ) return;
	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;
	unsigned degree_basic_form_si = deg_si(si);
	if( degree_basic_form_si > degree_in_blocks ) return;

	unsigned pow = 1;
	while( pow*deg_si(si) <= degree_in_blocks ) {
		for(unsigned i=0;i<n_terms;i+= blk_size*2*pow*deg_si(si) ) {
			i_poly_div_256( poly + i , blk_size*2*pow*deg_si(si) , blk_size , si , pow );
		}
		pow *= 2;
	}
}


void _bc_to_mono_256( __m256i * poly , unsigned n_terms , unsigned blk_size )
{

//printf("ibc: %d/%d\n", n_terms , blk_size );

	unsigned num_blocks = get_num_blocks( n_terms , blk_size );
	if( 2 >= num_blocks ) return;
	unsigned degree_in_blocks = num_blocks - 1;

//printf("deg: %d\n", degree_in_blocks);
	unsigned si = get_max_si( degree_in_blocks );
//printf("si: %d\n",si);

	unsigned new_blk_size = deg_si(si)*blk_size;
//printf("new blksize: %d\n", new_blk_size);
	for(unsigned i=0;i<n_terms;i+= new_blk_size ) {
		_bc_to_mono_256( poly + i , new_blk_size , blk_size );
	}
	_bc_to_mono_256( poly , n_terms , new_blk_size );
	i_represent_in_si_256( poly , n_terms , blk_size , si );
}


void bc_to_mono_256( bc_sto_t * poly , unsigned n_terms )
{

	_bc_to_mono_256( (__m256i*)poly , n_terms , 1 );
}





