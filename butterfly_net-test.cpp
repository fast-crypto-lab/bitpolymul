
#include <stdio.h>

#include "benchmark.h"
#include "byte_inline_func.h"
#include "immintrin.h"

#include "butterfly_net.h"

#define TEST_RUN 100


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define bm_func1 butterfly_net_half_inp_256
#define n_fn1 "fn:" TOSTRING(bm_func1) "()"

#define bm_func2 i_butterfly_net_256
#define n_fn2 "fn:" TOSTRING(bm_func2) "()"


//typedef uint64_t sto_t;
//typedef __m128i sto_t;
typedef __m256i sto_t;


//#define _EXIT_WHILE_FAIL_

#define LEN 1024




int main()
{
	unsigned char seed[32] = {0};

	sto_t poly1[LEN] __attribute__((aligned(32)));
	sto_t poly2[LEN] __attribute__((aligned(32)));

	benchmark bm1;
	bm_init(&bm1);
	benchmark bm2;
	bm_init(&bm2);


	byte_rand( (uint64_t*)poly1 , (sizeof(sto_t)/sizeof(uint64_t))*LEN/2 );
	for(unsigned i=LEN/2;i<LEN;i++) poly1[i] ^= poly1[i];

	memcpy( poly2 , poly1 , LEN*sizeof(sto_t) );
	if(1024>LEN) byte_dump( (uint64_t*)poly2 , (sizeof(sto_t)/sizeof(uint64_t))*LEN ); puts("");
	bm_func1( (uint64_t*) poly2 , LEN );
	if(1024>LEN) byte_dump( (uint64_t*)poly2 , (sizeof(sto_t)/sizeof(uint64_t))*LEN ); puts("");


	for(unsigned i=0;i<TEST_RUN;i++) {
		byte_rand( (uint64_t*)poly1 , (sizeof(sto_t)/sizeof(uint64_t))*LEN/2 );
		memcpy( poly2 , poly1 , LEN*sizeof(sto_t) );
BENCHMARK( bm1 , {
		bm_func1( (uint64_t*)poly2 , LEN );
} );
BENCHMARK( bm2 , {
		bm_func2( (uint64_t*)poly2 , LEN );
} );

		byte_xor( (uint64_t*)poly2 , (uint64_t*)poly1 , (sizeof(sto_t)/sizeof(uint64_t))*LEN );
		if( ! byte_is_zero( (uint64_t*)poly2 , (sizeof(sto_t)/sizeof(uint64_t))*LEN ) ) {
			printf("consistency fail: %d.\n", i);
			printf("diff:"); byte_fdump(stdout,(uint64_t*)poly2,LEN);
			printf("\n");
#define _EXIT_WHILE_FAIL_
#ifdef _EXIT_WHILE_FAIL_
			exit(-1);
#endif
		}
	}

	printf("check: %x\n", *((unsigned *)poly2) );
	char msg[256];
	bm_dump( msg , 256 , &bm1 );
	printf("benchmark (%s) :\n%s\n\n", n_fn1 , msg );

	bm_dump( msg , 256 , &bm2 );
	printf("benchmark (%s) :\n%s\n\n", n_fn2 , msg );

	return 0;
}
