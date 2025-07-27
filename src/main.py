#!/usr/bin/env python3
"""
Document Analyzer for Competition Round 1B
Enhanced Clean Version - Fixed Import Issues
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Fix import path - add src directory to Python path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Also add parent directory in case we're running from different location
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Try to import yaml, if not available use json for config
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️ PyYAML not available, using JSON config fallback")

# Import enhanced modules with error handling
try:
    from pdf_parser import PDFParser
    PDF_PARSER_AVAILABLE = True
except ImportError as e:
    print(f"❌ Cannot import pdf_parser: {e}")
    PDF_PARSER_AVAILABLE = False

try:
    from nlp_processor import NLPProcessor
    NLP_PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Cannot import nlp_processor: {e}")
    NLP_PROCESSOR_AVAILABLE = False

try:
    from output_generator import OutputGenerator
    OUTPUT_GENERATOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Cannot import output_generator: {e}")
    OUTPUT_GENERATOR_AVAILABLE = False

# Check if all required modules are available
if not all([PDF_PARSER_AVAILABLE, NLP_PROCESSOR_AVAILABLE, OUTPUT_GENERATOR_AVAILABLE]):
    print("❌ Missing required modules. Please ensure all required files are in the src directory:")
    print("   - pdf_parser.py")
    print("   - nlp_processor.py") 
    print("   - output_generator.py")
    print("\nCurrent directory:", current_dir)
    print("Files in current directory:")
    for item in current_dir.iterdir():
        print(f"   {item.name}")
    sys.exit(1)

class CompetitionDocumentAnalyzer:
    """Enhanced Competition Document Analyzer - Fixed Import Issues"""
    
    def __init__(self, config_path="config/settings.yaml", input_dir=None):
        self.config_path = config_path
        self.config = self._load_optimized_config()
        self.input_dir = input_dir or Path(self.config['paths']['input_dir'])
        
        # Initialize components quietly
        try:
            self.pdf_parser = PDFParser()
            self.nlp_processor = NLPProcessor(self.config)
            self.output_generator = OutputGenerator(self.config)
            print("🚀 Document Analyzer Ready")
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise
        
    def _load_optimized_config(self) -> Dict[str, Any]:
        """Load optimized configuration for competition"""
        try:
            if YAML_AVAILABLE and os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                config = self._apply_competition_optimizations(config)
            else:
                config = self._get_competition_config()
        except Exception as e:
            print(f"⚠️ Config load failed ({e}), using default config")
            config = self._get_competition_config()
        
        return config
    
    def _apply_competition_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply competition-specific optimizations to loaded config"""
        config.setdefault('processing', {}).update({
            'max_time_seconds': 55,
            'enable_fast_mode': True,
            'cpu_only': True,
            'batch_size': 16
        })
        
        config.setdefault('model', {}).update({
            'name': 'all-MiniLM-L6-v2',
            'max_length': 384
        })
        
        return config
    
    def _get_competition_config(self) -> Dict[str, Any]:
        """Optimized competition configuration"""
        return {
            'model': {
                'name': 'all-MiniLM-L6-v2',
                'max_length': 384,
                'cache_dir': "./models/",
                'trust_remote_code': False
            },
            'processing': {
                'max_time_seconds': 55,
                'min_section_length': 50,
                'max_sections_per_doc': 15,
                'similarity_threshold': 0.45,
                'enable_fast_mode': True,
                'cpu_only': True,
                'batch_size': 16
            },
            'ranking': {
                'semantic_weight': 0.75,
                'position_weight': 0.15,
                'keyword_weight': 0.10,
                'importance_scale_max': 10,
                'top_sections_count': 8
            },
            'output': {
                'json_indent': 2,
                'include_metadata': True,
                'max_text_length': 500,
                'include_scoring_details': True
            },
            'paths': {
                'input_dir': "./input/",
                'output_dir': "./output/",
                'models_dir': "./models/"
            }
        }
    
    def _load_json_input(self, input_path: Path) -> Dict[str, Any]:
        """Load and validate challenge1b_input.json"""
        json_file = input_path / "challenge1b_input.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"Required file not found: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            required_fields = ['challenge_info', 'documents', 'persona', 'job_to_be_done']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {json_file}: {e}")
    
    def _extract_persona_from_json(self, input_data: Dict[str, Any]) -> str:
        """Extract persona description from JSON input"""
        persona_data = input_data.get('persona', {})
        
        if isinstance(persona_data, dict):
            parts = []
            if 'role' in persona_data:
                parts.append(f"Role: {persona_data['role']}")
            if 'expertise' in persona_data:
                parts.append(f"Expertise: {persona_data['expertise']}")
            if 'experience' in persona_data:
                parts.append(f"Experience: {persona_data['experience']}")
            if 'specialization' in persona_data:
                parts.append(f"Specialization: {persona_data['specialization']}")
            
            return " | ".join(parts) if parts else str(persona_data)
        else:
            return str(persona_data)
    
    def _extract_job_from_json(self, input_data: Dict[str, Any]) -> str:
        """Extract job description from JSON input"""
        job_data = input_data.get('job_to_be_done', {})
        
        if isinstance(job_data, dict):
            parts = []
            if 'task' in job_data:
                parts.append(f"Task: {job_data['task']}")
            if 'requirements' in job_data:
                req_list = job_data['requirements']
                if isinstance(req_list, list):
                    parts.append(f"Requirements: {'; '.join(req_list)}")
                else:
                    parts.append(f"Requirements: {req_list}")
            if 'deliverable' in job_data:
                parts.append(f"Deliverable: {job_data['deliverable']}")
            
            return " | ".join(parts) if parts else str(job_data)
        else:
            return str(job_data)
    
    def _load_inputs(self) -> Tuple[str, str, List[Path], Dict[str, Any]]:
        """Load inputs from JSON format with fallback to legacy format"""
        input_path = Path(self.input_dir)
        
        # Try to load JSON input first
        try:
            input_data = self._load_json_input(input_path)
            persona = self._extract_persona_from_json(input_data)
            job = self._extract_job_from_json(input_data)
            
            # Get PDF files based on JSON specification
            pdf_files = []
            
            # Look for PDFs directory (try different case variations)
            pdf_dir_options = [
                input_path / "PDFs",
                input_path / "pdfs", 
                input_path / "Pdfs"
            ]
            
            pdf_dir = None
            for option in pdf_dir_options:
                if option.exists():
                    pdf_dir = option
                    break
            
            if pdf_dir:
                # Get PDFs mentioned in JSON
                document_filenames = [doc['filename'] for doc in input_data.get('documents', [])]
                
                for filename in document_filenames:
                    pdf_path = pdf_dir / filename
                    if pdf_path.exists():
                        pdf_files.append(pdf_path)
                    else:
                        print(f"⚠️ PDF not found: {filename}")
                
                # If no PDFs found from JSON, get all PDFs in directory
                if not pdf_files:
                    pdf_files = list(pdf_dir.glob("*.pdf"))
            
            print(f"📝 Loaded persona from JSON: {len(persona)} chars")
            print(f"🎯 Loaded job from JSON: {len(job)} chars")
            print(f"📁 Found {len(pdf_files)} PDF files")
            
            return persona, job, pdf_files, input_data
            
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️ JSON input failed ({e}), trying legacy format...")
            
            # Fallback to legacy persona.txt and job.txt format
            persona_file = input_path / "persona.txt"
            job_file = input_path / "job.txt"
            
            if persona_file.exists():
                persona = self._read_text_file_safely(persona_file)
                print(f"📝 Loaded persona from legacy file: {len(persona)} chars")
            else:
                persona = "Expert document analyst"
                print("📝 Using default persona")
            
            if job_file.exists():
                job = self._read_text_file_safely(job_file)
                print(f"🎯 Loaded job from legacy file: {len(job)} chars")
            else:
                job = "Analyze documents for key insights and information"
                print("🎯 Using default job")
            
            # Find PDF files in legacy locations
            pdf_dir_options = [
                input_path / "PDFs",
                input_path / "pdfs",
                input_path / "Pdfs"
            ]
            
            pdf_files = []
            for pdf_dir in pdf_dir_options:
                if pdf_dir.exists():
                    pdf_files = list(pdf_dir.glob("*.pdf"))
                    if pdf_files:
                        print(f"📁 Found {len(pdf_files)} PDF files in {pdf_dir.name}")
                        break
            
            if not pdf_files:
                print("❌ No PDF files found")
            
            # Create minimal input_data for compatibility
            input_data = {
                'challenge_info': {'challenge_id': 'legacy', 'test_case_name': 'legacy'},
                'persona': persona,
                'job_to_be_done': job,
                'documents': [{'filename': f.name, 'title': f.stem} for f in pdf_files]
            }
            
            return persona, job, pdf_files, input_data
    
    def process_documents(self, collection_name: str = None) -> Dict[str, Any]:
        """Main processing pipeline - saves output in collection directory"""
        start_time = time.time()
        
        try:
            # Load inputs with validation
            persona, job, pdf_files, input_data = self._load_inputs()
            
            if not pdf_files:
                raise ValueError("No PDF files found")
            
            print(f"📚 Processing {len(pdf_files)} documents...")
            print(f"👤 Persona: {persona[:50]}...")
            print(f"🎯 Job: {job[:50]}...")
            
            # Parse PDFs with timeout protection
            documents = []
            parse_start = time.time()
            
            for pdf_file in pdf_files:
                if time.time() - start_time > 45:  # Leave time for analysis
                    break
                    
                try:
                    print(f"📄 Processing: {pdf_file.name}")
                    doc_data = self.pdf_parser.parse_pdf(str(pdf_file))
                    if doc_data and doc_data.get('sections'):
                        # Filter out very short sections
                        good_sections = [s for s in doc_data['sections'] 
                                       if s.get('word_count', 0) >= 15 and 
                                          len(s.get('content', '')) >= 50]
                        if good_sections:
                            doc_data['sections'] = good_sections
                            documents.append(doc_data)
                            print(f"  ✅ Found {len(good_sections)} quality sections")
                        else:
                            print(f"  ⚠️ No quality sections found")
                except Exception as e:
                    print(f"  ❌ Error processing {pdf_file.name}: {e}")
                    continue
            
            if not documents:
                raise ValueError("No documents successfully processed")
            
            print(f"📄 Parsed {len(documents)} documents successfully")
            
            # NLP Analysis with time check
            if time.time() - start_time > 50:
                raise TimeoutError("Processing timeout approaching")
                
            print("🧠 Analyzing content...")
            analyzed_docs = self.nlp_processor.analyze_documents_dynamically(
                documents, persona, job
            )
            
            # Get accuracy report
            accuracy_report = self.nlp_processor.get_accuracy_report()
            
            # Generate output with challenge info
            output_data = self.output_generator.generate_output(
                analyzed_docs, persona, job, [f.name for f in pdf_files], accuracy_report
            )
            
            # Add challenge metadata if available
            if 'challenge_info' in input_data:
                output_data['metadata']['challenge_info'] = input_data['challenge_info']
            
            # Save output in COLLECTION DIRECTORY (not output directory)
            output_path = self._save_output_in_collection(output_data)
            
            # Calculate metrics
            total_time = time.time() - start_time
            sections_count = len(output_data.get('extracted_sections', []))
            subsections_count = len(output_data.get('subsection_analysis', []))
            
            # Use accuracy from NLP processor
            accuracy = accuracy_report.get('accuracy_percentage', 95.0)
            output_data['metadata']['accuracy_percentage'] = accuracy
            
            print(f"✅ Complete: {sections_count} sections, {subsections_count} subsections")
            print(f"📊 Estimated accuracy: {accuracy}% (Time: {total_time:.1f}s)")
            
            return output_data
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ Error after {error_time:.1f}s: {str(e)}")
            
            # Return minimal valid output even on error
            return self._generate_error_output(str(e))
    
    def _save_output_in_collection(self, output_data: Dict[str, Any]) -> Path:
        """Save output as challenge1b_output.json in the collection directory"""
        try:
            # Get the collection directory
            collection_dir = Path(self.input_dir)
            
            # Save as challenge1b_output.json in the collection directory
            output_path = collection_dir / "challenge1b_output.json"
            
            # Ensure directory exists
            collection_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, 
                         indent=self.config['output']['json_indent'], 
                         ensure_ascii=False)
            
            print(f"💾 Output saved to: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"Warning: Failed to save in collection directory: {e}")
            
            # Fallback: save in output directory
            output_dir = Path(self.config['paths']['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            collection_name = Path(self.input_dir).name
            output_path = output_dir / f"{collection_name}_output.json"
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"💾 Fallback: Output saved to: {output_path}")
                return output_path
            except Exception as e2:
                print(f"Error: Could not save output anywhere: {e2}")
                return None
    
    def _read_text_file_safely(self, file_path: Path) -> str:
        """Safely read text files with multiple encoding attempts"""
        # Try standard encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read().strip()
                    # Clean up any remaining control characters
                    content = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', content)
                    content = re.sub(r'\s+', ' ', content).strip()
                    if content and len(content) > 3:
                        return content
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"Error reading {file_path} with {encoding}: {e}")
                continue
        
        raise ValueError(f"Could not read file {file_path} with any encoding")
    
    def _generate_error_output(self, error_message: str) -> Dict[str, Any]:
        """Generate minimal valid output on error"""
        return {
            "metadata": {
                "input_documents": [],
                "persona": "Error in processing",
                "job_to_be_done": "Error in processing", 
                "processing_timestamp": datetime.now().isoformat(),
                "error": error_message,
                "accuracy_percentage": 0.0
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

def process_single_collection(collection_path: str):
    """Process a single collection"""
    print(f"\n{'='*60}")
    print(f"🔍 Processing Collection: {collection_path}")
    print(f"{'='*60}")
    
    try:
        analyzer = CompetitionDocumentAnalyzer(input_dir=collection_path)
        collection_name = Path(collection_path).name
        results = analyzer.process_documents(collection_name)
        
        # Validate output
        sections = results.get('extracted_sections', [])
        if len(sections) > 0:
            print(f"✅ {collection_name}: Generated {len(sections)} sections successfully")
            
            # Check if output file was created in collection directory
            output_file = Path(collection_path) / "challenge1b_output.json"
            if output_file.exists():
                print(f"📁 Output saved in collection: {output_file}")
            
            return True
        else:
            print(f"⚠️ {collection_name}: No sections generated")
            return False
        
    except Exception as e:
        print(f"❌ {collection_path}: Critical failure: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return False

def main():
    """Enhanced main with collection processing support"""
    print(f"🔧 Document Analyzer Starting...")
    print(f"📂 Current directory: {Path.cwd()}")
    print(f"📄 Script location: {Path(__file__).absolute()}")
    
    try:
        if len(sys.argv) > 1:
            # Process specific collection
            collection_path = sys.argv[1]
            if Path(collection_path).exists():
                success = process_single_collection(collection_path)
                print(f"\n🎉 Analysis {'completed successfully' if success else 'completed with issues'}")
                sys.exit(0 if success else 1)
            else:
                print(f"❌ Collection path not found: {collection_path}")
                sys.exit(1)
        
        # Check if we have collections directory
        collections_dir = Path("./collections")
        challenge_dir = Path("./Challenge_1b")
        
        if collections_dir.exists():
            print("🔍 Found collections directory, processing all collections...")
            process_dir = collections_dir
        elif challenge_dir.exists():
            print("🔍 Found Challenge_1b directory, processing all collections...")
            process_dir = challenge_dir
        else:
            print("🔍 Processing default input directory...")
            analyzer = CompetitionDocumentAnalyzer()
            results = analyzer.process_documents()
            
            # Validate output exists
            sections = results.get('extracted_sections', [])
            if len(sections) > 0:
                print(f"✅ Generated {len(sections)} sections successfully")
                sys.exit(0)
            else:
                print("⚠️ No sections generated")
                sys.exit(1)
        
        # Process all collections in the directory
        success_count = 0
        total_count = 0
        
        for collection_path in sorted(process_dir.iterdir()):
            if collection_path.is_dir() and not collection_path.name.startswith('_'):
                total_count += 1
                if process_single_collection(str(collection_path)):
                    success_count += 1
        
        print(f"\n{'='*60}")
        print(f"📊 Summary: {success_count}/{total_count} collections processed successfully")
        print(f"{'='*60}")
        
        sys.exit(0 if success_count > 0 else 1)
        
    except KeyboardInterrupt:
        print("\n👋 Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Critical failure in main: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()