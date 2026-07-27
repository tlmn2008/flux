__global__ void k(void* dst, const void* tmap, int c0){
  asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
    "[%0], [%1, {%2}], [%0];" :: "l"(dst),"l"(tmap),"r"(c0));
}
int main(){return 0;}
