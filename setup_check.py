#!/usr/bin/env python3
"""
Setup Verification Script for Document Analysis System
Run this after cloning from Git to verify everything is working correctly.
"""

import os
import sys
import json
import importlib
from pathlib import Path

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

def check_directory_structure():
    """Check if required directories and files exist"""
    print_status("Checking directory structure...", "info")
    
    required_paths = [
        "src/main.py",
        "src/nlp_processor.py", 
        "src/pdf_parser.py",
        "input/pdfs",
        "input/persona.txt",
        "input/job.txt",
        "config/settings.yaml",
        "requirements.txt"
    ]
    
    all_good = True
    for path in required_paths:
        if os.path.exists(path):
            print_status(f"Found: {path}", "success")
        else:
            print_status(f"Missing: {path}", "error")
            all_good = False
    
    return all_good

def check_pdf_files():
    """Check if sample PDF files are present"""
    print_status("Checking sample PDF files...", "info")
    
    pdf_dir = Path("input/pdfs")
    if not pdf_dir.exists():
        print_status("PDF directory missing!", "error")
        return False
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if len(pdf_files) == 0:
        print_status("No PDF files found - you'll need to add your own", "warning")
        return False
    
    print_status(f"Found {len(pdf_files)} PDF files:", "success")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print_status(f"  • {pdf.name} ({size_mb:.1f}MB)", "info")
    
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

def check_model_cache():
    """Check if AI model is downloaded"""
    print_status("Checking AI model cache...", "info")
    
    models_dir = Path("models")
    if not models_dir.exists():
        print_status("Models directory doesn't exist - will be created on first run", "warning")
        return False
    
    # Check for model files
    model_files = list(models_dir.rglob("*"))
    if len(model_files) == 0:
        print_status("No model files found - will download on first run", "warning")
        return False
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_files if f.is_file())
    size_mb = total_size / (1024 * 1024)
    
    print_status(f"AI model cached ({size_mb:.0f}MB)", "success")
    return True

def test_model_loading():
    """Test if the AI model can be loaded"""
    print_status("Testing AI model loading...", "info")
    
    try:
        from sentence_transformers import SentenceTransformer
        print_status("Attempting to load model...", "info")
        
        # This will download if not present
        model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models')
        print_status("AI model loaded successfully!", "success")
        
        # Test encoding
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print_status(f"Model test encoding successful (dimension: {len(embedding)})", "success")
        
        return True
        
    except Exception as e:
        print_status(f"Model loading failed: {str(e)}", "error")
        return False

def create_missing_directories():
    """Create missing directories"""
    print_status("Creating missing directories...", "info")
    
    directories = ["models", "output"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print_status(f"Created: {directory}/", "success")

def run_full_test():
    """Run the complete analysis to test everything"""
    print_status("Running full system test...", "info")
    
    try:
        # Import and run main function
        sys.path.append('src')
        from main import main
        
        print_status("Starting document analysis...", "info")
        main()
        
        # Check if output was created
        if os.path.exists("output/challenge1b_output.json"):
            with open("output/challenge1b_output.json", 'r') as f:
                data = json.load(f)
            
            sections = len(data.get("extracted_sections", []))
            subsections = len(data.get("subsection_analysis", []))
            accuracy = data.get("metadata", {}).get("accuracy_percentage", 0)
            
            print_status(f"Analysis completed successfully!", "success")
            print_status(f"Generated {sections} sections, {subsections} subsections", "success") 
            print_status(f"Estimated accuracy: {accuracy}%", "success")
            return True
        else:
            print_status("Analysis completed but no output file found", "error")
            return False
            
    except Exception as e:
        print_status(f"Full test failed: {str(e)}", "error")
        return False

def main():
    """Main setup verification function"""
    print("🔍 Document Analysis System - Setup Verification")
    print("=" * 60)
    
    # Track overall status
    all_checks_passed = True
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directory_structure), 
        ("PDF Files", check_pdf_files),
        ("Dependencies", check_dependencies),
        ("Model Cache", check_model_cache)
    ]
    
    print_status("Running setup verification checks...", "info")
    print()
    
    for check_name, check_func in checks:
        print(f"📋 {check_name}")
        result = check_func()
        if not result:
            all_checks_passed = False
        print()
    
    # Create missing directories
    create_missing_directories()
    print()
    
    # Test model loading
    print("🧠 AI Model Test")
    model_test = test_model_loading()
    print()
    
    if all_checks_passed and model_test:
        print("🎉 All checks passed! Running full system test...")
        print()
        full_test = run_full_test()
        
        if full_test:
            print()
            print_status("🎉 SETUP VERIFICATION COMPLETE!", "success")
            print_status("Your system is ready to analyze documents!", "success")
            print_status("Run 'python src/main.py' to start analyzing", "info")
        else:
            print_status("Setup verification completed with issues", "warning")
    else:
        print_status("Setup verification found issues", "error")
        print_status("Please fix the errors above and run this script again", "info")
        print_status("For help, see the troubleshooting section in README.md", "info")

if __name__ == "__main__":
    main()