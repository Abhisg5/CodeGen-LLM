import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from pathlib import Path
import logging
from tqdm import tqdm
import json
from typing import Dict, Any
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model_name: str = "gpt2",
        output_dir: str = "models/codegen",
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        max_length: int = 512,
        use_wandb: bool = True
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_length = max_length
        self.use_wandb = use_wandb
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def load_data(self, data_path: str) -> DataLoader:
        """Load and prepare the dataset."""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create dataset
        encodings = self.tokenizer(
            data,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, train_dataloader: DataLoader):
        """Train the model."""
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_dataloader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Initialize wandb if enabled
        if self.use_wandb:
            wandb.init(project="codegen-llm", config={
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_epochs": self.num_epochs,
                "max_length": self.max_length
            })
        
        # Training loop
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "loss": loss.item(),
                        "epoch": epoch + 1
                    })
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            checkpoint_dir = self.output_dir / f"checkpoint-epoch-{epoch + 1}"
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save final model
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Saved final model to {self.output_dir}")
        
        if self.use_wandb:
            wandb.finish()

def main():
    # Set up paths
    data_path = "data/processed/processed_data.json"
    output_dir = "models/codegen"
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_name="gpt2",
        output_dir=output_dir,
        batch_size=8,
        learning_rate=5e-5,
        num_epochs=3
    )
    
    # Load data
    train_dataloader = trainer.load_data(data_path)
    
    # Train model
    trainer.train(train_dataloader)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    main() 