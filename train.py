import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from model import DARA, Config
from dataset import DARADataset
import os

def train():
    print("Initializing DARA Training...")
    
    # Load Base Model
    dara = DARA()
    model = dara.model
    processor = dara.processor
    
    # LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"], # Adjust based on Florence-2 architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Dataset
    train_dataset = DARADataset("data/train", processor)
    eval_dataset = DARADataset("data/eval", processor)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./dara_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=1,
        num_train_epochs=1,
        max_steps=5, # Short run for verification
        save_steps=100,
        fp16=True if torch.cuda.is_available() else False,
        remove_unused_columns=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    print("Starting training...")
    trainer.train() 
    print("Training verification complete.")

if __name__ == "__main__":
    train()
