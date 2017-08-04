
#ifndef _BUTTERFLY_NET_H_
#define _BUTTERFLY_NET_H_


#include <stdint.h>


#ifdef  __cplusplus
extern  "C" {
#endif



void butterfly_net_half_inp( uint64_t * fx , unsigned n_fx );

void i_butterfly_net( uint64_t * fx , unsigned n_fx );


void butterfly_net_half_inp_clmul( uint64_t * fx , unsigned n_fx );

void i_butterfly_net_clmul( uint64_t * fx , unsigned n_fx );


void butterfly_net_half_inp_256( uint64_t * fx , unsigned n_fx );

void i_butterfly_net_256( uint64_t * fx , unsigned n_fx );



#ifdef  __cplusplus
}
#endif


#endif
