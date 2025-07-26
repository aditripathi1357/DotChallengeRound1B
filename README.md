# Docker Setup and Usage Instructions

## Quick Start Guide

### Step 1: Prepare Your Input
```bash
# 1. Place your PDF documents in the input folder
cp your_documents/*.pdf input/pdfs/

# 2. Create persona description
echo "Investment analyst with expertise in financial analysis and market research" > input/persona.txt

# 3. Create job description  
echo "Analyze revenue trends and identify investment opportunities from the provided financial reports" > input/job.txt
```

### Step 2: Build and Run with Docker
```bash
# Build the Docker container
docker build -t doc-analyzer .

# Run the analysis
docker run -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output doc-analyzer
```

### Step 3: View Results
```bash
# Check the generated output
cat output/challenge1b_output.json
```

## Local Development (Alternative)

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Navigate to project directory
cd doc_analyzer

# Run the system
python src/main.py
```

### Expected Output
```
🚀 Document Analyzer Ready
📚 Processing 9 documents...
🎯 Detected domain: food_catering
📋 Extracted requirements: ['quality_standards', 'vegetarian', 'gluten_free']
🔍 Enhanced adaptive filtering: 9 → 9 documents
🧠 Analyzing content...
✅ Enhanced adaptive analysis complete: 12 relevant sections
📋 Generating enhanced competition output...
✅ Enhanced output generated: 10 sections, 6 subsections
📊 Estimated accuracy: 75.0% (Time: 13.9s)
```

## Docker Container Details

### Base Image
- **Python 3.10-slim**: Lightweight Python runtime
- **CPU-optimized**: No GPU dependencies
- **Size**: ~2GB total (includes models)

### Pre-installed Components
- All Python dependencies from requirements.txt
- Pre-downloaded sentence-transformer model (all-MiniLM-L6-v2)
- Optimized for offline operation

### Volume Mounts
- `/app/input`: Mount your input folder here
- `/app/output`: Results will be written here
- `/app/models`: Model cache (internal)

### Environment Variables
```dockerfile
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
ENV TRANSFORMERS_CACHE="/app/models"
```

## Troubleshooting

### Common Issues
1. **"No PDF files found"**: Ensure PDFs are in `input/pdfs/` directory
2. **Permission errors**: Check Docker has read/write access to mounted volumes
3. **Out of memory**: Ensure at least 4GB RAM available
4. **Slow processing**: Normal for CPU-only operation, expect 30-60 seconds

### Docker Commands
```bash
# View container logs
docker logs <container_id>

# Interactive container access
docker run -it doc-analyzer /bin/bash

# Clean up containers
docker system prune
```