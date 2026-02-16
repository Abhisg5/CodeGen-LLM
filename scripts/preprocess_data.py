import os
import json
from pathlib import Path
import logging
from typing import List, Dict
import re
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CodeDataset(Dataset):
    def __init__(self, data: List[str], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

def clean_code(code: str) -> str:
    """Clean and normalize code."""
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    # Remove empty lines
    code = '\n'.join(line for line in code.split('\n') if line.strip())
    
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)
    
    return code.strip()

def process_file(file_path: Path) -> str:
    """Process a single file and return cleaned code."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return clean_code(content)
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return ""

def create_dataset(input_dir: str, output_dir: str, tokenizer_name: str = "gpt2"):
    """Create a processed dataset from raw code files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Collect all Python files
    python_files = list(input_path.rglob("*.py"))
    logger.info(f"Found {len(python_files)} Python files")
    
    # Process files
    processed_data = []
    for file_path in tqdm(python_files, desc="Processing files"):
        cleaned_code = process_file(file_path)
        if cleaned_code:
            processed_data.append(cleaned_code)
    
    # Create dataset
    dataset = CodeDataset(processed_data, tokenizer)
    
    # Save processed data
    output_file = output_path / "processed_data.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Saved processed data to {output_file}")
    
    # Create and save tokenized dataset
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_path / "tokenizer")
    
    return dataset, dataloader

def main():
    # Set up paths
    input_dir = "data/raw/github"
    output_dir = "data/processed"
    
    # Create dataset
    dataset, dataloader = create_dataset(input_dir, output_dir)
    
    logger.info("Data preprocessing completed successfully")

if __name__ == "__main__":
    main() 