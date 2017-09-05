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

#ifndef _BYTE_INLINE_FUNC_H_
#define _BYTE_INLINE_FUNC_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>



static inline
void byte_rand( uint64_t * vec , unsigned n ) { for(unsigned i=0;i<n;i++) vec[i] = rand()&0xff; }

static inline
unsigned byte_is_zero( const uint64_t * vec , unsigned n ) { unsigned r=0; for(unsigned i=0;i<n;i++) r|= vec[i]; return (0==r); }

static inline
void byte_xor( uint64_t * v1 , const uint64_t * v2 , unsigned n ) { for(unsigned i=0;i<n;i++) v1[i]^= v2[i]; }

static inline
void byte_fdump( FILE * fp, const uint64_t *v, unsigned _num_byte) {
	fprintf(fp,"[%2d][",_num_byte);
	for(unsigned i=0;i<_num_byte;i++) { fprintf(fp,"0x%02lx,",v[i]); if(7==(i%8)) fprintf(fp," ");}
	fprintf(fp,"]");
}

static inline
void byte_dump( const uint64_t *v, unsigned _num_byte ) { byte_fdump(stdout,v,_num_byte); }


#endif
