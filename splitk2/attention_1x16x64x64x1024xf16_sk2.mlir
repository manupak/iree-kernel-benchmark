!dtype = f16
!Q     = tensor<1x16x64xf16>
!K     = tensor<1x1024x64xf16>
!K_SK     = tensor<1x4x256x64xf16>
!V     = tensor<1x1024x64xf16>
!V_SK     = tensor<1x4x256x64xf16>
!O     = tensor<1x16x64xf16>
!O_f32     = tensor<1x16x64xf32>
!O_SK     = tensor<1x4x16x64xf32>

!ROWRED_SK = tensor<1x4x16xf32>
!ROWRED = tensor<1x16xf32>

#tuning = #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{ workgroup = [1, 1, 16, 0, 0, 0], reduction = [0, 0, 0, 0, 0, 32],promote_operands = [0, 1, 2] }>, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64] subgroup_size = 64,{llvm_func_attrs = { "amdgpu-waves-per-eu" = "2","denormal-fp-math-f32" = "preserve-sign" }}>>

#Q_SK = affine_map<(b1, b2, m, n, k1, k2) -> (b1, m, k1)>
#K_SK = affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, k2, k1)>
#V_SK = affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, k2, n)>
#S_SK = affine_map<(b1, b2, m, n, k1, k2) -> ()>
#O_SK = affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, m, n)>
#ROWRED = affine_map<(b1, b2, m, n, k1, k2) -> (b1, b2, m)>


func.func @main(%Q : !Q, %K : !K, %V : !V) -> !O {
  %scale = arith.constant 1.0 : !dtype

  //reshape to SK shape
  %sk_shape = arith.constant dense<[1, 4, 256, 64]> : tensor<4xi64>
  %K_SK = tensor.reshape %K(%sk_shape) : (!K, tensor<4xi64>) -> !K_SK
  %V_SK = tensor.reshape %V(%sk_shape) : (!V, tensor<4xi64>) -> !V_SK

  %c1 = arith.constant 1.000000e+00 : f32
  %cneginf = arith.constant -3.40282347E+38 : f32
  %c0 = arith.constant 0.000000e+00 : f32
  %c0_dtype = arith.constant 0.000000e+00 : !dtype

  %empty = tensor.empty() : !O
  %empty_0 = linalg.fill ins(%c0_dtype : !dtype) outs(%empty : !O) -> !O

  %empty_f32 = tensor.empty() : !O_f32
  %empty_f32_0 = linalg.fill ins(%c0 : f32) outs(%empty_f32 : !O_f32) -> !O_f32

  %empty_sk = tensor.empty() : !O_SK
  %empty_sk_0 = linalg.fill ins(%c0 : f32) outs(%empty_sk : !O_SK) -> !O_SK

  %rowmax_sk = tensor.empty() : !ROWRED_SK
  %rowmax_sk_0 = linalg.fill ins(%cneginf : f32) outs(%rowmax_sk : !ROWRED_SK) -> !ROWRED_SK
  %rowsum_sk = tensor.empty() : !ROWRED_SK
  %rowsum_sk_0 = linalg.fill ins(%c0 : f32) outs(%rowsum_sk : !ROWRED_SK) -> !ROWRED_SK

  %O_SK:3 = iree_linalg_ext.online_attention 
  {
    indexing_maps = [
      #Q_SK, 
      #K_SK, 
      #V_SK, 
      #S_SK, 
      #O_SK,
      #ROWRED,
      #ROWRED
    ],
    decomposition_config = {
      pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1, promote_operands = [1]}>}, 
      qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1, promote_operands = [0, 1]}>}
    }, 
    compilation_info = #tuning
  }
  ins(%Q, %K_SK, %V_SK, %scale : !Q, !K_SK, !V_SK, !dtype) outs(%empty_sk_0, %rowmax_sk_0, %rowsum_sk_0 : !O_SK, !ROWRED_SK, !ROWRED_SK) {
    ^bb0(%arg4: f32):
      iree_linalg_ext.yield %arg4 : f32
  } -> !O_SK, !ROWRED_SK, !ROWRED_SK
  
  // Get max of all split-k s
  // Split-k recombiner maps
  %rowmax0 = tensor.empty() : !ROWRED
  %rowmax1 = linalg.fill ins(%cneginf : f32) outs(%rowmax0 : !ROWRED) -> !ROWRED
  %rowmax2 = linalg.generic {indexing_maps = [affine_map<(b1, b2, m) -> (b1, b2, m)>, affine_map<(b1, b2, m) -> (b1, m)>], iterator_types = ["parallel", "reduction", "parallel"]} ins(%O_SK#1 : !ROWRED_SK) outs(%rowmax1 : !ROWRED) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.maximumf %in, %out : f32
      linalg.yield %0 : f32
  } -> !ROWRED
  // replace the old max with new max
  %updated_o = linalg.generic {indexing_maps = [affine_map<(b1, b2, m, n) -> (b1, m)>, affine_map<(b1, b2, m, n) -> (b1, b2, m)>, affine_map<(b1, b2, m, n) -> (b1, b2, m, n)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%rowmax2, %O_SK#1 : !ROWRED, !ROWRED_SK) outs(%O_SK#0 : !O_SK) {
  ^bb0(%max: f32, %split_max: f32, %out: f32):
    %0 = arith.subf %split_max, %max : f32
    %1 = math.exp2 %0 : f32
    %2 = arith.mulf %1, %out : f32
    linalg.yield %2 : f32
  } -> !O_SK
  // accumulate
  %acc = linalg.generic {indexing_maps = [affine_map<(b1, b2, m, n) -> (b1, b2, m, n)>, affine_map<(b1, b2, m, n) -> (b1, m, n)>], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins(%updated_o : !O_SK) outs(%empty_f32_0 : !O_f32) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
  } -> !O_f32
  // update sum
  %rowsum0 = tensor.empty() : !ROWRED
  %rowsum1 = linalg.fill ins(%c0 : f32) outs(%rowsum0 : !ROWRED) -> !ROWRED
  %rowsum2 = linalg.generic {indexing_maps = [affine_map<(b1, b2, m) -> (b1, b2, m)>, affine_map<(b1, b2, m) -> (b1, b2, m)>, affine_map<(b1, b2, m) -> (b1, m)>, affine_map<(b1, b2, m) -> (b1, m)>], iterator_types = ["parallel", "reduction", "parallel"]} ins(%O_SK#2, %O_SK#1, %rowmax2: !ROWRED_SK, !ROWRED_SK, !ROWRED) outs(%rowsum1 : !ROWRED) {
    ^bb0(%split_sum: f32, %split_max: f32, %max : f32, %sum: f32):
      %0 = arith.subf %split_max, %max : f32
      %1 = math.exp2 %0 : f32
      %2 = arith.mulf %1, %split_sum : f32
      %3 = arith.addf %2, %sum : f32
      linalg.yield %3 : f32
  } -> !ROWRED
  // division
  %out = linalg.generic {indexing_maps = [affine_map<(b, m, n) -> (b, m)>, affine_map<(b, m, n) -> (b, m, n)>, affine_map<(b, m, n) -> (b, m, n)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%rowsum2, %acc : !ROWRED, !O_f32) outs(%empty : !O) {
  ^bb0(%sum: f32, %acc_: f32, %out: f16):
    %17 = arith.divf %c1, %sum : f32
    %acc_div = arith.mulf %17, %acc_ : f32
    %acc_divt = arith.truncf %acc_div : f32 to f16
    linalg.yield %acc_divt : f16
  } -> !O
  return %out : !O
}
