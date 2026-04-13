"""
Simple script to run the RAG application.
This script provides an easy way to start the Streamlit application.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the RAG Streamlit application."""
    
    # Get the directory containing this script
    current_dir = Path(__file__).parent
    
    # Change to the RAG directory
    os.chdir(current_dir)
    
    # Set Python path to include src directory
    src_path = current_dir / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    print("🧠 Starting RAG Application...")
    print(f"📁 Working directory: {current_dir}")
    print("🔧 Make sure you have installed all requirements:")
    print("   pip install -r requirements.txt")
    print("\n🚀 Launching Streamlit application...")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--theme.base=light"
        ], check=True)
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
    
    except KeyboardInterrupt:
        print("\n👋 Shutting down RAG application...")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()