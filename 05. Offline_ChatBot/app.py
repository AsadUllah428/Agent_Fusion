from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import os

app = FastAPI()

# Global variables for model and tokenizer
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    try:
        tokenizer_path = "model/tokenizer"
        model_path = "model"
        
        # Check if files exist
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}. "
                "Run 'python install_model.py' first to download and convert the model."
            )
        
        if not os.path.exists(os.path.join(model_path, "model.onnx")):
            raise FileNotFoundError(
                f"Model not found at {model_path}/model.onnx. "
                "Run 'python install_model.py' first to download and convert the model."
            )
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("Loading ONNX model with Optimum...")
        model = ORTModelForCausalLM.from_pretrained(model_path)
        
        print("✅ Model and tokenizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise

class InputText(BaseModel):
    text: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_p: float = 0.9
    do_sample: bool = True

@app.post("/generate")
async def generate_text(input_data: InputText):
    try:
        if tokenizer is None or model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Tokenize input
        inputs = tokenizer(input_data.text, return_tensors="pt")
        
        # Generate text using the model
        outputs = model.generate(
            **inputs,
            max_new_tokens=input_data.max_new_tokens,
            temperature=input_data.temperature,
            top_p=input_data.top_p,
            do_sample=input_data.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"generated_text": generated_text}
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {str(e)}\nTraceback: {error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

@app.get("/")
async def root():
    return {
        "message": "SLM Agent API",
        "endpoints": {
            "/generate": "POST - Generate text",
            "/health": "GET - Health check"
        }
    }