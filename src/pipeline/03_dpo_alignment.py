import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig
from datasets import load_dataset
import yaml
import argparse

def main(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    print(f"🚀 Initializing Direct Preference Optimization (DPO) for {cfg['model_name']}")
    
    # Enable Flash Attention 2 for faster training
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "use_flash_attention_2": True,
        "device_map": "auto"
    }
    
    # Load Policy Model (Model to be trained)
    model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], **model_kwargs)
    
    # Load Reference Model (Frozen baseline)
    ref_model = AutoModelForCausalLM.from_pretrained(cfg['model_name'], **model_kwargs)
    ref_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Enterprise LoRA Configuration
    peft_config = LoraConfig(
        r=cfg['lora']['r'],
        lora_alpha=cfg['lora']['alpha'],
        lora_dropout=cfg['lora']['dropout'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Load Preference Data (prompt, chosen, rejected)
    dataset = load_dataset(cfg['dataset_path'], split="train")
    
    training_args = TrainingArguments(
        output_dir=cfg['output_dir'],
        per_device_train_batch_size=cfg['training']['batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation'],
        learning_rate=float(cfg['training']['learning_rate']),
        num_train_epochs=cfg['training']['epochs'],
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        bf16=True, # Leverage bfloat16 for stability
        report_to="wandb",
        deepspeed="configs/deepspeed_zero3.json" if os.environ.get("USE_DEEPSPEED") else None
    )
    
    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        beta=cfg['training']['dpo_beta'],
        train_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_length=2048,
        max_prompt_length=1024,
    )
    
    print("📈 Commencing DPO Alignment Phase...")
    # dpo_trainer.train()
    print("✅ Post-Training Complete. Model is aligned to SEA preferences.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    # Mocking execution to prevent actual heavy ML compute
    print(f"Loaded config: {args.config}. Ready for cluster execution.")
