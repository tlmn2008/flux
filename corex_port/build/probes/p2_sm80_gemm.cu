#include "cutlass/gemm/device/gemm.h"
using Gemm = cutlass::gemm::device::Gemm<
  cutlass::half_t, cutlass::layout::RowMajor,
  cutlass::half_t, cutlass::layout::ColumnMajor,
  cutlass::half_t, cutlass::layout::RowMajor,
  float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;
int main(){
  Gemm op; Gemm::Arguments args({128,128,128},{nullptr,128},{nullptr,128},{nullptr,128},{nullptr,128},{1.0f,0.0f});
  auto st = op(args); return (int)st;
}
