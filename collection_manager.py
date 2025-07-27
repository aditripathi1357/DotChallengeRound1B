#!/usr/bin/env python3
"""
Enhanced Collection Manager - Unified JSON creation and analysis
"""

import os
import json
import subprocess
import sys
from pathlib import Path

# Suppress TensorFlow warnings and other noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

def scan_collections():
    """Find all collections in both 'collections' and 'Challenge_1b' directories"""
    collections = []
    
    # Check both possible directories
    for base_dir in ["collections", "Challenge_1b"]:
        collections_dir = Path(base_dir)
        if not collections_dir.exists():
            continue
            
        for item in collections_dir.iterdir():
            if item.is_dir():
                pdfs_dir = item / "pdfs"
                pdf_count = len(list(pdfs_dir.glob("*.pdf"))) if pdfs_dir.exists() else 0
                has_json = (item / "challenge1b_input.json").exists()
                
                if pdf_count > 0:  # Only collections with PDFs
                    collections.append({
                        "name": item.name,
                        "path": str(item),
                        "pdf_count": pdf_count,
                        "has_json": has_json,
                        "base_dir": base_dir
                    })
    
    return collections

def create_input_json(collection_path):
    """Create challenge1b_input.json with user input"""
    collection_name = Path(collection_path).name
    print(f"\n📝 Creating JSON for: {collection_name}")
    
    # Scan PDFs
    pdfs_dir = Path(collection_path) / "pdfs"
    pdf_files = list(pdfs_dir.glob("*.pdf")) if pdfs_dir.exists() else []
    
    if not pdf_files:
        print("❌ No PDF files found in pdfs directory")
        return None
    
    documents = []
    for pdf in pdf_files:
        title = pdf.stem.replace("_", " ").replace("-", " ").title()
        documents.append({"filename": pdf.name, "title": title})
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    # Get user input with validation
    print("\n" + "="*50)
    print("📋 Please provide the following information:")
    print("="*50)
    
    # Role input
    while True:
        role = input("👤 Your role/job title: ").strip()
        if role:
            break
        print("   Role is required - please enter your role")
    
    # Task input
    while True:
        task = input("🎯 Main task/analysis goal: ").strip()
        if task:
            break
        print("   Task description is required")
    
    # Optional requirements
    print("\n📝 Additional requirements (optional):")
    print("   Enter requirements one by one. Press Enter on empty line to finish.")
    requirements = []
    while True:
        req = input(f"   Requirement {len(requirements)+1}: ").strip()
        if not req:
            break
        requirements.append(req)
    
    # Optional deliverable
    deliverable = input("📦 Expected deliverable (or press Enter for default): ").strip()
    if not deliverable:
        deliverable = "Comprehensive analysis and insights from documents"
    
    # Create JSON structure
    json_data = {
        "challenge_info": {
            "challenge_id": f"round_1b_{collection_name.lower().replace(' ', '_').replace('-', '_')}",
            "test_case_name": f"{collection_name.lower().replace(' ', '_').replace('-', '_')}_analysis",
            "description": task
        },
        "documents": documents,
        "persona": {
            "role": role
        },
        "job_to_be_done": {
            "task": task,
            "requirements": requirements,
            "deliverable": deliverable
        }
    }
    
    # Save JSON file
    json_file = Path(collection_path) / "challenge1b_input.json"
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Created: {json_file}")
        print(f"📊 Configuration summary:")
        print(f"   • Role: {role}")
        print(f"   • Task: {task}")
        print(f"   • Documents: {len(documents)} PDFs")
        print(f"   • Requirements: {len(requirements)} items")
        
        return json_file
        
    except Exception as e:
        print(f"❌ Error creating JSON file: {e}")
        return None

def run_analysis(collection_path):
    """Run analysis with clean output"""
    collection_name = Path(collection_path).name
    print(f"\n🚀 Running analysis on '{collection_name}'...")
    print("⏳ This may take a few minutes...")
    
    try:
        # Run analysis with captured output
        result = subprocess.run([
            sys.executable, "src/main.py", str(collection_path)
        ], capture_output=True, text=True, check=True, timeout=300)  # 5 minute timeout
        
        # Filter and display important output lines
        lines = result.stdout.split('\n')
        important_lines = []
        
        for line in lines:
            line = line.strip()
            if any(keyword in line for keyword in [
                '✅ Complete:', '📊 Estimated accuracy:', '💾 Output saved to:',
                'Generated', 'sections successfully', 'Processing completed',
                '🎉 Analysis complete', 'Results saved'
            ]):
                important_lines.append(line)
        
        if important_lines:
            print("\n📈 Analysis Results:")
            for line in important_lines:
                print(f"   {line}")
        
        # Check for output files
        output_dir = Path("output")
        if output_dir.exists():
            output_files = list(output_dir.glob("*output.json"))
            if output_files:
                latest_output = max(output_files, key=lambda x: x.stat().st_mtime)
                print(f"\n✅ Analysis complete! Results saved to: {latest_output}")
            else:
                print(f"\n✅ Analysis complete! Check the output folder for results.")
        else:
            print(f"\n✅ Analysis complete!")
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Analysis timed out (5 minutes). Try with fewer documents.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed with error code {e.returncode}")
        if e.stderr:
            error_lines = e.stderr.split('\n')
            # Show only the last few error lines to avoid spam
            relevant_errors = [line for line in error_lines[-10:] if line.strip() and 'WARNING' not in line]
            if relevant_errors:
                print("Error details:")
                for line in relevant_errors[:3]:  # Show max 3 error lines
                    print(f"   {line}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def display_collections(collections):
    """Display collections in a formatted table"""
    if not collections:
        print("❌ No collections found in 'collections' or 'Challenge_1b' directories")
        return
    
    print("\n📁 Available Collections:")
    print("=" * 70)
    print(f"{'#':<3} {'Status':<8} {'Name':<25} {'PDFs':<6} {'Location'}")
    print("-" * 70)
    
    for i, col in enumerate(collections, 1):
        status = "✅ Ready" if col["has_json"] else "❌ No JSON"
        name = col["name"][:24] + "..." if len(col["name"]) > 24 else col["name"]
        print(f"{i:<3} {status:<8} {name:<25} {col['pdf_count']:<6} {col['base_dir']}")

def main():
    """Enhanced main function with better UX"""
    print("🔧 Enhanced Collection Manager")
    print("=" * 50)
    print("Unified tool for JSON creation and document analysis")
    print("=" * 50)
    
    # Scan for collections
    collections = scan_collections()
    display_collections(collections)
    
    if not collections:
        print("\n💡 Tip: Make sure you have collections in 'collections/' or 'Challenge_1b/' directories")
        print("   Each collection should have a 'pdfs/' subdirectory with PDF files")
        return
    
    # Main menu
    print(f"\n🎯 What would you like to do?")
    print("1. 📝 Create/Update JSON configuration for a collection")
    print("2. 🚀 Run analysis on a collection")
    print("3. 🔄 Create JSON and run analysis (full workflow)")
    print("4. ❓ Show collection details")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice not in ['1', '2', '3', '4']:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4")
            return
        
        # Get collection selection
        try:
            col_num = int(input(f"Select collection number (1-{len(collections)}): ")) - 1
            if not (0 <= col_num < len(collections)):
                print(f"❌ Invalid selection. Please enter a number between 1 and {len(collections)}")
                return
        except ValueError:
            print("❌ Please enter a valid number")
            return
        
        selected_collection = collections[col_num]
        collection_path = Path(selected_collection["path"])
        
        # Execute chosen action
        if choice == '1':
            # Create/Update JSON
            create_input_json(collection_path)
            
        elif choice == '2':
            # Run analysis
            if not selected_collection["has_json"]:
                print("⚠️  No JSON configuration found.")
                create_first = input("Would you like to create one first? (y/n): ").strip().lower()
                if create_first == 'y':
                    if create_input_json(collection_path):
                        run_analysis(collection_path)
                else:
                    print("❌ Cannot run analysis without JSON configuration")
            else:
                run_analysis(collection_path)
                
        elif choice == '3':
            # Full workflow
            print("🔄 Running full workflow: JSON creation + analysis")
            if create_input_json(collection_path):
                print("\n" + "="*50)
                run_analysis(collection_path)
                
        elif choice == '4':
            # Show details
            print(f"\n📋 Collection Details: {selected_collection['name']}")
            print("-" * 40)
            print(f"Path: {selected_collection['path']}")
            print(f"PDF files: {selected_collection['pdf_count']}")
            print(f"Has JSON config: {'Yes' if selected_collection['has_json'] else 'No'}")
            print(f"Base directory: {selected_collection['base_dir']}")
            
            # Show PDF files
            pdfs_dir = collection_path / "pdfs"
            if pdfs_dir.exists():
                pdf_files = list(pdfs_dir.glob("*.pdf"))
                if pdf_files:
                    print(f"\nPDF Files ({len(pdf_files)}):")
                    for i, pdf in enumerate(pdf_files[:10], 1):  # Show max 10
                        print(f"  {i}. {pdf.name}")
                    if len(pdf_files) > 10:
                        print(f"  ... and {len(pdf_files) - 10} more")
        
        print(f"\n🎉 Operation completed!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()