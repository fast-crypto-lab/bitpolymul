#ifndef _BC_H_
#define _BC_H_

#include <stdint.h>



#ifdef  __cplusplus
extern  "C" {
#endif


typedef uint64_t bc_sto_t;


void bc_to_lch( bc_sto_t * poly , unsigned n_terms );

void bc_to_mono( bc_sto_t * poly , unsigned n_terms );


void bc_to_lch_128( bc_sto_t * poly , unsigned n_terms );

void bc_to_mono_128( bc_sto_t * poly , unsigned n_terms );


void bc_to_lch_256( bc_sto_t * poly , unsigned n_terms );

void bc_to_mono_256( bc_sto_t * poly , unsigned n_terms );


#ifdef  __cplusplus
}
#endif



#endif
