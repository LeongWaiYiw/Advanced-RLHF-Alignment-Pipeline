# MLOps Makefile for Alignment Pipeline

.PHONY: install format lint train-sft train-dpo evaluate clean docker-build

install:
	pip install -r requirements.txt
	pip install flash-attn --no-build-isolation

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	mypy src/

train-sft:
	accelerate launch --config_file configs/accelerate.yaml src/pipeline/01_sft.py --config configs/sft_config.yaml

train-dpo:
	accelerate launch --config_file configs/accelerate.yaml src/pipeline/03_dpo_alignment.py --config configs/dpo_sea_lion.yaml

evaluate:
	python src/evaluation/llm_judge.py --model_path ./outputs/dpo_final --dataset data/eval.jsonl

docker-build:
	docker build -t rlhf-pipeline:latest .

clean:
	rm -rf __pycache__ .pytest_cache outputs/ wandb/
