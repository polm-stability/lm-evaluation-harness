#!/usr/bin/env python
# This script runs eval in the cluster. Use it as a basis for your own harnesses.
from run_eval import build_executor, run_job
from run_eval import JAEVAL8_TASKS, JAEVAL8_FEWSHOT
from main_eval import main as main_eval

def main():
    executor = build_executor("eval", gpus_per_task=8, cpus_per_gpu=12)
    eval_args = {
        "tasks": JAEVAL8_TASKS,
        "num_fewshot": JAEVAL8_FEWSHOT,
        "model": "hf-causal",
        "model_args": "pretrained=rinna/japanese-gpt-1b,use_fast=False",
        "device": "cuda",
        "limit": 100,
        "verbose": True,
    }

    run_job(executor, main_eval, args=eval_args, output_path="./check.json")

if __name__ == "__main__":
    main()
