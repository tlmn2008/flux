__global__ void k(int* g){
  __shared__ int s[4];
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"((int)(size_t)s),"l"(g));
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
}
int main(){return 0;}
