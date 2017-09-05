/*
Copyright (C) 2017 Ming-Shing Chen

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

#ifndef _GF16_TABS_H_
#define _GF16_TABS_H_

#include <stdint.h>

#ifdef  __cplusplus
extern  "C" {
#endif

extern const unsigned char __mask_0x55[];
extern const unsigned char __mask_low[];
extern const unsigned char __mask_16bit_low[];
extern const unsigned char __mask_32bit_low[];
extern const unsigned char __mask_16[];
extern const unsigned char __gf16_inv[];
extern const unsigned char __gf16_squ[];
extern const unsigned char __gf16_squ_x8[];
extern const unsigned char __gf16_squ_sl4[];
extern const unsigned char __gf16_exp[];
extern const char __gf16_log[];
extern const unsigned char * __gf16_mul;
extern const unsigned char __gf256_mul[];
extern const unsigned char __gf16_mulx2[];


#ifdef  __cplusplus
}
#endif



#endif
