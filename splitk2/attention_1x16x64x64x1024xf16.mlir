!dtype = f16
!Q     = tensor<1x16x64xf16>
!K     = tensor<1x1024x64xf16>
!V     = tensor<1x1024x64xf16>
!O     = tensor<1x16x64xf16>

#tuning = #iree_codegen.compilation_info<lowering_config = #iree_gpu.lowering_config<{ workgroup = [1, 16, 0, 0, 0], reduction = [0, 0, 0, 0, 32],promote_operands = [0, 1, 2] }>, translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute workgroup_size = [64] subgroup_size = 64 , {llvm_func_attrs = { "amdgpu-waves-per-eu" = "2","denormal-fp-math-f32" = "preserve-sign" }}>>


#Q = affine_map<(b, m, n, k1, k2) -> (b, m, k1)>
#K = affine_map<(b, m, n, k1, k2) -> (b, k2, k1)>
#V = affine_map<(b, m, n, k1, k2) -> (b, k2, n)>
#S = affine_map<(b, m, n, k1, k2) -> ()>
#O = affine_map<(b, m, n, k1, k2) -> (b, m, n)>

func.func @main(%Q : !Q, %K : !K, %V : !V) -> !O {
  %scale = arith.constant 1.0 : !dtype
  //%c1 = arith.constant 1 : index
  //%size1 = tensor.dim %Q, %c1 : !O
  %empty = tensor.empty() : !O
  %O = iree_linalg_ext.attention 
       { indexing_maps = [#Q, #K, #V, #S, #O],
         decomposition_config = {
              pv_attrs = {attention_pv_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1, promote_operands = [1]}>}, 
              qk_attrs = {attention_qk_matmul, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>, subgroup_m_count = 1, subgroup_n_count = 1, promote_operands = [0, 1]}>}
            }, 
         compilation_info = #tuning
       }
       ins(%Q, %K, %V, %scale : !Q, !K, !V, !dtype) outs(%empty : !O) {
          ^bb0(%score: f32):
            iree_linalg_ext.yield %score : f32
        } -> !O
  return %O : !O
}
