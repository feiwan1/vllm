export VLLM_XPU_FUSED_MOE_USE_TRITON=True
export ZE_AFFINITY_MASK=1,2,3,4

python basic.py
