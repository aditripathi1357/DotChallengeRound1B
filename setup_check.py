#!/usr/bin/env python3
"""
Simple Setup Verification Script - System Check Only
"""

import os
import sys
import importlib
from pathlib import Path

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "info": "\033[94m",      # Blue
        "success": "\033[92m",   # Green  
        "warning": "\033[93m",   # Yellow
        "error": "\033[91m",     # Red
        "reset": "\033[0m"       # Reset
    }
    
    icons = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️",
        "error": "❌"
    }
    
    color = colors.get(status, colors["info"])
    icon = icons.get(status, "•")
    reset = colors["reset"]
    
    print(f"{color}{icon} {message}{reset}")

def check_python_version():
    """Check if Python version is compatible"""
    print_status("Checking Python version...", "info")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Compatible!", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - Need Python 3.11+", "error")
        return False

def check_core_files():
    """Check if core project files exist"""
    print_status("Checking core project files...", "info")
    
    required_files = [
        "src/main.py",
        "src/nlp_processor.py", 
        "src/pdf_parser.py",
        "src/output_generator.py",
        "config/settings.yaml",
        "requirements.txt"
    ]
    
    all_good = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print_status(f"Found: {file_path}", "success")
        else:
            print_status(f"Missing: {file_path}", "error")
            all_good = False
    
    return all_good

def check_collections():
    """Check collections directory"""
    print_status("Checking collections...", "info")
    
    collections_dir = Path("collections")
    if not collections_dir.exists():
        print_status("Collections directory not found", "warning")
        return False
    
    collections = [d for d in collections_dir.iterdir() if d.is_dir()]
    if not collections:
        print_status("No collections found", "warning")
        return False
    
    print_status(f"Found {len(collections)} collections:", "success")
    for collection in collections:
        pdfs_dir = collection / "pdfs"
        pdf_count = len(list(pdfs_dir.glob("*.pdf"))) if pdfs_dir.exists() else 0
        has_json = (collection / "challenge1b_input.json").exists()
        
        status = "✅" if has_json and pdf_count > 0 else "⚠️"
        print_status(f"  {status} {collection.name} ({pdf_count} PDFs, JSON: {'Yes' if has_json else 'No'})", "info")
    
    return True

def check_dependencies():
    """Check if required Python packages are installed"""
    print_status("Checking Python dependencies...", "info")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "SentenceTransformers"),
        ("PyPDF2", "PDF processing"),
        ("pdfplumber", "Advanced PDF parsing"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("yaml", "YAML parsing"),
        ("nltk", "Natural Language Toolkit"),
        ("tqdm", "Progress bars"),
        ("joblib", "Parallel processing")
    ]
    
    all_good = True
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            print_status(f"{description} - Installed", "success")
        except ImportError:
            print_status(f"{description} - Missing", "error")
            all_good = False
    
    return all_good

def check_model():
    """Check if AI model can be loaded"""
    print_status("Testing AI model...", "info")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
        
        # Test encoding
        test_text = "System verification test."
        embedding = model.encode(test_text)
        
        print_status(f"AI model working (dimension: {len(embedding)})", "success")
        
        # Check model size
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.rglob("*"))
            if model_files:
                total_size = sum(f.stat().st_size for f in model_files if f.is_file())
                size_mb = total_size / (1024 * 1024)
                print_status(f"Model cache size: {size_mb:.0f}MB", "info")
        
        return True
        
    except Exception as e:
        print_status(f"Model test failed: {str(e)}", "error")
        return False

def check_directories():
    """Check if basic directories exist"""
    print_status("Checking directories...", "info")
    
    directories = ["models"]
    for directory in directories:
        if Path(directory).exists():
            print_status(f"Found: {directory}/", "success")
        else:
            print_status(f"Missing: {directory}/ (will be created when needed)", "warning")

def main():
    """Main setup verification function"""
    print("🔍 Document Analysis System - Setup Verification")
    print("=" * 60)
    
    # Track overall status
    all_checks_passed = True
    
    # Run core checks
    checks = [
        ("Python Version", check_python_version),
        ("Core Files", check_core_files),
        ("Collections", check_collections),
        ("Dependencies", check_dependencies),
        ("AI Model", check_model),
        ("Directories", check_directories)
    ]
    
    print_status("Running system verification...", "info")
    print()
    
    for check_name, check_func in checks:
        print(f"📋 {check_name}")
        result = check_func()
        if not result and check_name in ["Python Version", "Core Files", "Dependencies", "AI Model"]:
            all_checks_passed = False
        print()
    
    # Final status
    if all_checks_passed:
        print_status("🎉 SYSTEM VERIFICATION COMPLETE!", "success")
        print_status("Your system is ready for document analysis!", "success")
        print()
        print_status("To use the system:", "info")
        print_status("  python collection_manager.py  # Create and manage collections", "info")
        print_status("  python src/main.py           # Run analysis", "info")
    else:
        print_status("❌ System verification found issues", "error")
        print_status("Please fix the errors above before proceeding", "info")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")