import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

def create_model_directory():
    """Create model directory if it doesn't exist"""
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir

def download_and_convert_model():
    print("Starting model download and conversion process...")
    
    # Create model directory
    model_dir = create_model_directory()
    
    model_name = "distilgpt2"
    
    print(f"Downloading and converting {model_name} to ONNX format...")
    print("This may take a few minutes on first run...\n")
    
    # Load tokenizer
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save tokenizer locally
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    print(f"💾 Saving tokenizer to {tokenizer_path}...")
    tokenizer.save_pretrained(tokenizer_path)
    
    # Load and export model to ONNX using Optimum
    print("🔄 Converting model to ONNX format (this handles all the complexity)...")
    model = ORTModelForCausalLM.from_pretrained(
        model_name,
        export=True  # This automatically exports to ONNX
    )
    
    # Save the ONNX model
    onnx_model_path = model_dir
    print(f"💾 Saving ONNX model to {onnx_model_path}...")
    model.save_pretrained(onnx_model_path)
    
    # Get model file size
    model_file = os.path.join(model_dir, "model.onnx")
    if os.path.exists(model_file):
        model_size = os.path.getsize(model_file) / (1024*1024)
        print(f"\n📊 Model size: {model_size:.2f} MB")
    
    print(f"\n✅ Model successfully converted and saved to {onnx_model_path}")
    print(f"✅ Tokenizer saved to {tokenizer_path}")
    
    return onnx_model_path

if __name__ == "__main__":
    import logging
    
    # Setup logging
    logging.basicConfig(
        filename='model_conversion.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        logging.info("Starting model conversion process")
        print("\n" + "="*60)
        print("  SLM Agent - Model Download & Conversion")
        print("  Using Optimum for seamless ONNX export")
        print("="*60 + "\n")
        
        download_and_convert_model()
        
        print("\n" + "="*60)
        print("  ✅ Setup Complete!")
        print("="*60)
        print("\nNext steps:")
        print("1. Start the backend: uvicorn app:app --reload")
        print("2. In a new terminal, start Streamlit: streamlit run streamlit_app.py")
        print("\n")
        
        logging.info("Model conversion completed successfully")
    except Exception as e:
        error_msg = f"❌ Error during model conversion: {str(e)}"
        print("\n" + error_msg)
        logging.error(error_msg)
        
        print("\n🔧 Troubleshooting tips:")
        print("- Make sure you have enough disk space (~500MB needed)")
        print("- Install optimum: pip install optimum[onnxruntime]")
        print("- Check the model_conversion.log file for details")
        
        import traceback
        traceback.print_exc()
        
        raise