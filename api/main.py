from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(
    title="CodeGen LLM API",
    description="API for generating code using a fine-tuned language model",
    version="1.0.0"
)

# Initialize model and tokenizer
MODEL_PATH = os.getenv("MODEL_PATH", "models/codegen")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None
    tokenizer = None

class CodeGenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    num_return_sequences: Optional[int] = 1

class CodeGenerationResponse(BaseModel):
    generated_code: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to CodeGen LLM API"}

@app.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        
        # Generate code
        outputs = model.generate(
            inputs["input_ids"],
            max_length=request.max_length,
            temperature=request.temperature,
            num_return_sequences=request.num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode generated sequences
        generated_sequences = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return CodeGenerationResponse(generated_code=generated_sequences)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    } 