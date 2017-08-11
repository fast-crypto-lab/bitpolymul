#ifndef _SKA_H_
#define _SKA_H_


unsigned get_s_k_a( unsigned k , unsigned a );

static inline
unsigned get_s_k_a_cantor( unsigned k , unsigned a ) { return (a>>k); }

#endif
