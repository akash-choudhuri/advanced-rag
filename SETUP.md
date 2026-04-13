# 🚀 RAG Project Setup Guide

This guide will walk you through setting up and running the RAG (Retrieval-Augmented Generation) project on your local machine.

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (for version control)
- At least 4GB of free RAM (recommended 8GB+)

## 🛠️ Installation Steps

### 1. Clone or Navigate to the Project
```bash
cd /path/to/your/RAG/project
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration (Optional)
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your preferred settings (optional)
# nano .env  # or use any text editor
```

### 5. Run the Application
```bash
# Option 1: Using the run script
python run.py

# Option 2: Direct Streamlit command
streamlit run app.py
```

## 🌐 Accessing the Application

Once the application starts, you can access it at:
- **URL**: http://localhost:8501
- The application will automatically open in your default web browser

## 📚 Using the RAG System

### Step 1: Upload Documents
1. Go to the **"Document Management"** tab
2. Click **"Choose files"** and select PDF or TXT files
3. Click **"Process Documents"** to add them to the knowledge base

### Step 2: Chat with Your Documents
1. Go to the **"Chat Interface"** tab
2. Type your question in the chat input
3. The system will retrieve relevant documents and generate an answer
4. View sources by clicking the **"View Sources"** expander

### Step 3: Monitor System Status
1. Go to the **"System Status"** tab to view:
   - Vector store information
   - Model details
   - Health checks

## ⚙️ Configuration Options

You can customize the system behavior by editing the `.env` file:

### Model Configuration
```env
# Change embedding model (affects similarity search quality)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Change LLM model (affects response generation)
LLM_MODEL=distilgpt2
```

### Processing Parameters
```env
# Document chunking settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Generation parameters
MAX_TOKENS=512
TEMPERATURE=0.7
TOP_K_RESULTS=5
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "Module not found" errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

#### 2. Memory issues during model loading
- Use a smaller model: `LLM_MODEL=distilgpt2`
- Close other memory-intensive applications
- Reduce `MAX_TOKENS` and `CHUNK_SIZE` in `.env`

#### 3. Slow document processing
- Reduce `CHUNK_SIZE` for faster processing
- Process fewer documents at once
- Use a machine with more RAM

#### 4. Poor answer quality
- Upload more relevant documents
- Increase `TOP_K_RESULTS` to retrieve more context
- Try different embedding models
- Adjust `TEMPERATURE` for more/less creative responses

#### 5. Port already in use
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Network**: Internet connection (for model downloads)

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 5GB free space
- **GPU**: CUDA-compatible GPU (optional, for faster inference)

## 🔄 Updating the System

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Clear Vector Store (if needed)
1. Go to **Document Management** tab
2. Click **"Clear Vector Store"**
3. Re-upload your documents

## 🐛 Getting Help

### Logs and Debugging
- Check the terminal/console for error messages
- Streamlit logs appear in the browser and terminal
- Enable debug mode by setting `LOG_LEVEL=DEBUG` in `.env`

### Performance Tips
1. **For better accuracy**: Use larger embedding models
2. **For faster responses**: Use smaller LLM models
3. **For GPU acceleration**: Install CUDA-compatible PyTorch
4. **For production**: Consider using cloud-hosted models

## 📝 Sample Questions to Try

After uploading the sample AI document, try these questions:

1. "What is machine learning?"
2. "Explain the difference between supervised and unsupervised learning"
3. "What are the main applications of AI in healthcare?"
4. "What are the challenges in AI development?"
5. "Tell me about neural networks"

## 🔧 Advanced Configuration

### Using GPU (if available)
```bash
# Install CUDA-compatible PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Custom Models
Edit `.env` to use different models:
```env
# For better embeddings (larger download)
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2

# For better text generation (requires more memory)
LLM_MODEL=microsoft/DialoGPT-large
```

## 📞 Support

If you encounter issues:
1. Check this setup guide
2. Review error messages in the terminal
3. Ensure all requirements are installed
4. Try with the default configuration first
5. Check system requirements

---

**Happy RAG-ing! 🧠✨**