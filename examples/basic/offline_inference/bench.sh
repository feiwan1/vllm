#!/usr/bin/env bash

set -euo pipefail

#export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
#export ZE_AFFINITY_MASK=1,2,3,4
export ZE_AFFINITY_MASK=4,5,6,7

MODEL_PATH="/data0/Qwen3-30B-A3B/"

#MODEL_PATH="/data0/Qwen3-235B-A22B-modified/"
# 限制加载前 N 层的 Qwen3 MoE expert weights，可设为 1~94
#export VLLM_QWEN3_MOE_MAX_LOADED_LAYERS=8

HOST="127.0.0.1"
PORT="8000"
TP_SIZE="4"
MAX_MODEL_LEN="20480"
GPU_MEMORY_UTILIZATION="0.8"

# TTFT benchmark settings

#export VLLM_XPU_FUSED_MOE_USE_TRITON=1
export VLLM_XPU_FUSED_MOE_USE_TRITON=0
export TRITON_PRINT_AUTOTUNING=1

#export VLLM_XPU_FUSED_MOE_TRITON_TUNING_MODE=fixed
export VLLM_XPU_FUSED_MOE_TRITON_TUNING_MODE=autotune

# 256 512 1024 2048 4096 8192 16384
INPUT_LEN="256"
OUTPUT_LEN="128"
NUM_PROMPTS="32"
MAX_CONCURRENCY="1"
MAX_BATCH_SIZE="8"

BASE_URL="http://${HOST}:${PORT}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_LOG="${SCRIPT_DIR}/bench_server.log"
SERVER_PID=""


cleanup() {
	if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
		kill "${SERVER_PID}" 2>/dev/null || true
		wait "${SERVER_PID}" 2>/dev/null || true
	fi
}

wait_for_server() {
	local retries="180"
	local interval="5"

	for _ in $(seq 1 "${retries}"); do
		if curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; then
			return 0
		fi
		sleep "${interval}"
	done

	return 1
}

trap cleanup EXIT

if ! curl -sf "${BASE_URL}/v1/models" >/dev/null 2>&1; then
	echo "[bench] 启动本地 vLLM 服务: ${MODEL_PATH}"
	nohup vllm serve "${MODEL_PATH}" \
		--max-model-len "${MAX_MODEL_LEN}" \
		--max-num-seqs "${MAX_BATCH_SIZE}" \
		--tensor-parallel-size "${TP_SIZE}" \
		--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
		--port "${PORT}" \
		> "${SERVER_LOG}" 2>&1 &
	SERVER_PID=$!

	echo "[bench] 等待服务就绪: ${BASE_URL}"
	if ! wait_for_server; then
		echo "[bench] 服务启动失败，请检查日志: ${SERVER_LOG}"
		exit 1
	fi
fi

echo "[bench] 开始 TTFT 测试"
echo "[bench] Triton MoE 开关: VLLM_XPU_FUSED_MOE_USE_TRITON=${VLLM_XPU_FUSED_MOE_USE_TRITON:-0}"
echo "[bench] INPUT_LEN=${INPUT_LEN} OUTPUT_LEN=${OUTPUT_LEN} NUM_PROMPTS=${NUM_PROMPTS} MAX_CONCURRENCY=${MAX_CONCURRENCY} TP_SIZE=${TP_SIZE} MAX_BATCH_SIZE=${MAX_BATCH_SIZE}"

vllm bench serve \
	--base-url "${BASE_URL}" \
	--model "${MODEL_PATH}" \
	--dataset-name random \
	--input-len "${INPUT_LEN}" \
	--output-len "${OUTPUT_LEN}" \
	--max-concurrency "${MAX_CONCURRENCY}" \
	--num-prompts "${NUM_PROMPTS}" \
	--save-result \
	--save-detailed \
	--result-dir ./bench_results
	#--percentile-metrics ttft \
	#--metric-percentiles 50,90,95,99
