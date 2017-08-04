
#ifndef _BITPOLYMUL_H_
#define _BITPOLYMUL_H_


#include <stdint.h>

#define bitpolymul bitpolymul_256

#ifdef  __cplusplus
extern  "C" {
#endif


void bitpolymul_simple( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned n_64 );

void bitpolymul_128( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned n_64 );

void bitpolymul_256( uint64_t * c , const uint64_t * a , const uint64_t * b , unsigned n_64 );


#ifdef  __cplusplus
}
#endif


#endif
