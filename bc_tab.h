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


#ifndef _BC_TAB_H_
#define _BC_TAB_H_

#include <stdint.h>



#ifdef  __cplusplus
extern  "C" {
#endif


extern uint64_t bc_tab_from_mono_128[];

extern uint64_t bc_tab_from_mono_256_h128[];

extern uint64_t bc_tab_to_mono_128[];

extern uint64_t bc_tab_to_mono_256_h128[];

extern uint64_t bc_tab_from_mono_128_m4r[];

extern uint64_t bc_tab_from_mono_256_h128_m4r[];

extern uint64_t bc_tab_to_mono_128_m4r[];

extern uint64_t bc_tab_to_mono_256_h128_m4r[];


#ifdef  __cplusplus
}
#endif



#endif
