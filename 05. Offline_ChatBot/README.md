# SLM Agent (ONNX + Streamlit)

A completely free, local chatbot agent using an open-source small language model (SLM) in ONNX format. No paid APIs or cloud services required.

## Features
- Runs locally on your machine
- Uses ONNX Runtime for fast inference
- Streamlit web interface
- 100% open-source and free

## Quick Start Instructions

### 1. Clone or Download This Repository
Clone the repo or download the ZIP and extract it.

### 2. Download a Small ONNX Language Model
- Visit the HuggingFace ONNX model hub: [DistilGPT2 ONNX](https://huggingface.co/onnx/models/tree/main/distilgpt2-onnx)
- Download the `model.onnx` file.
- Place the downloaded `model.onnx` file in the `model/` folder of this project.

### 3. Install Python Dependencies
Open a terminal in the project folder and run:
```sh
pip install -r requirements.txt
```

### 4. Run the Streamlit App
Start the agent with:
```sh
streamlit run app.py
```
This will open a web interface in your browser.

### 5. Chat With the Agent
Type your message in the input box and click "Send". The agent will respond using the local ONNX model.

---

## Project Structure
```
slm-agent/
│
├── model/                  # Place your ONNX model here
│   └── model.onnx
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # Instructions
└── utils.py                # Helper functions (tokenization, etc.)
```

## Notes
- This project uses a small language model for demonstration. For larger models, ensure your hardware can handle them.
- All computation is local; no data leaves your machine.
- No paid APIs or cloud services are used—completely free to test and use.

## Credits
- Model: DistilGPT2 (HuggingFace, ONNX)
- Inference: ONNX Runtime
- UI: Streamlit









Your free, local SLM agent project with Streamlit is set up! Here’s what you have:

app.py: Streamlit web app for chatting with the agent.
utils.py: Handles tokenization and ONNX input/output.
requirements.txt: All free, open-source dependencies.
README.md: Setup and usage instructions.
model/: Place your ONNX model here (e.g., DistilGPT2 ONNX).
Next steps:

Download a small ONNX model (e.g., DistilGPT2) from HuggingFace and put it in the model/ folder as model.onnx.
Run pip install -r requirements.txt in the slm-agent folder.
Start the agent with streamlit run app.py.
Let me know if you want a direct download link for a model or further customization!




uvicorn app:app --reload
