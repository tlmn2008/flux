#include <cstdio>
__global__ void k(int* o){ *o = 42; }
int main(){ int *d; cudaMalloc(&d,4); k<<<1,1>>>(d); int h=0; cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost); printf("val=%d\n",h); return 0;}
