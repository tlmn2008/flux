__global__ void k(float* d, unsigned* a, unsigned* b){
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
    "{%0,%1,%2,%3}, %4, %5, 1, 1, 1;"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) : "l"(*(unsigned long long*)a),"l"(*(unsigned long long*)b));
}
int main(){return 0;}
