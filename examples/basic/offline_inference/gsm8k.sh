#pip install lm_eval[api]
export HF_ENDPOINT="https://hf-mirror.com"
#export ZE_AFFINITY_MASK=4,5,6,7
export ZE_AFFINITY_MASK=1,2,3,4

model=/data0/Qwen3-30B-A3B/
bs=64
tp=4
model_len=16384
task=gsm8k

lm_eval --model vllm --tasks $task \
        --model_args "pretrained=$model,trust_remote_code=True,tensor_parallel_size=$tp,distributed_executor_backend=mp,max_num_seqs=$bs,max_length=$model_len,max_gen_toks=2048,gpu_memory_utilization=0.7" \
        --device 'xpu' \
        --batch_size $bs \
        --log_samples \
        --limit 500 \
	--gen_kwargs temperature=0,do_sample=false \
	--num_fewshot 5 \
        --output_path ./lm_eval_output
