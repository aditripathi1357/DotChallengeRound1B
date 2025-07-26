# Optimized Dockerfile - Pre-downloads model during build
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV CUDA_VISIBLE_DEVICES=""
ENV PYTHONPATH="/app/src"
ENV PIP_TIMEOUT=300
ENV PIP_DEFAULT_TIMEOUT=300

# Install packages in smaller batches to avoid timeouts

# Batch 1: PyTorch (CPU only)
RUN pip install --no-cache-dir torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Batch 2: Core ML packages (install individually to avoid conflicts)
RUN pip install --no-cache-dir transformers==4.32.1
RUN pip install --no-cache-dir tokenizers==0.13.3  
RUN pip install --no-cache-dir huggingface-hub==0.16.4
RUN pip install --no-cache-dir sentence-transformers==2.2.2

# Batch 3: Data processing (install individually)
RUN pip install --no-cache-dir numpy==1.24.3
RUN pip install --no-cache-dir scikit-learn==1.3.0
RUN pip install --no-cache-dir scipy==1.11.3

# Batch 4: PDF and utility packages
RUN pip install --no-cache-dir PyPDF2==3.0.1 pdfplumber==0.9.0
RUN pip install --no-cache-dir pyyaml==6.0.1 nltk==3.8.1 regex==2023.8.8
RUN pip install --no-cache-dir textstat==0.7.3 tqdm==4.66.1 joblib==1.3.2

# Create directories
RUN mkdir -p /app/config /app/input/pdfs /app/models /app/output /app/src

# Copy application files
COPY config/ ./config/
COPY src/ ./src/
COPY input/ ./input/

# PRE-DOWNLOAD MODEL DURING BUILD (This is the key change!)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('🚀 Downloading model during build...'); \
model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/app/models'); \
print('✅ Model downloaded and cached successfully')"

# Enhanced entrypoint that ensures output directory exists
RUN printf '#!/bin/bash\n\
echo "🚀 Starting Document Analysis System..."\n\
\n\
# Always ensure output directory exists and is writable\n\
echo "📁 Setting up output directory..."\n\
mkdir -p /app/output\n\
chmod 755 /app/output\n\
\n\
# Check if output directory is writable\n\
if ! touch /app/output/.test 2>/dev/null; then\n\
    echo "⚠️ Output directory not writable, using temporary location"\n\
    export OUTPUT_DIR="/tmp/output"\n\
    mkdir -p "$OUTPUT_DIR"\n\
else\n\
    rm -f /app/output/.test\n\
    export OUTPUT_DIR="/app/output"\n\
fi\n\
\n\
echo "📂 Output directory: $OUTPUT_DIR"\n\
\n\
# Check if PDFs exist\n\
pdf_count=$(find /app/input/pdfs -name "*.pdf" 2>/dev/null | wc -l)\n\
if [ "$pdf_count" -eq 0 ]; then\n\
    echo "❌ No PDF files found in /app/input/pdfs/"\n\
    echo ""\n\
    echo "📋 Usage:"\n\
    echo "docker run -v /path/to/input:/app/input -v /path/to/output:/app/output doc-analyzer"\n\
    echo ""\n\
    echo "📂 Required structure:"\n\
    echo "  input/"\n\
    echo "  ├── pdfs/          # Your PDF files"\n\
    echo "  ├── persona.txt    # Persona description"\n\
    echo "  └── job.txt        # Job to be done"\n\
    exit 1\n\
fi\n\
\n\
echo "📚 Found $pdf_count PDF file(s)"\n\
\n\
# Create default input files if they dont exist\n\
if [ ! -f "/app/input/persona.txt" ]; then\n\
    echo "📝 Creating default persona.txt..."\n\
    echo "Expert analyst with deep domain knowledge and experience in extracting key insights from complex documents." > /app/input/persona.txt\n\
fi\n\
\n\
if [ ! -f "/app/input/job.txt" ]; then\n\
    echo "📝 Creating default job.txt..."\n\
    echo "Extract the most relevant insights and actionable information from the provided documents." > /app/input/job.txt\n\
fi\n\
\n\
echo "🎯 Starting analysis..."\n\
echo "📦 Using pre-downloaded model from /app/models"\n\
\n\
# Run the document analyzer\n\
cd /app\n\
if python src/main.py; then\n\
    echo ""\n\
    echo "✅ Analysis completed successfully!"\n\
    \n\
    # Copy result to mounted output if different\n\
    if [ "$OUTPUT_DIR" != "/app/output" ] && [ -f "$OUTPUT_DIR/challenge1b_output.json" ]; then\n\
        cp "$OUTPUT_DIR/challenge1b_output.json" /app/output/ 2>/dev/null || echo "Result saved to: $OUTPUT_DIR/challenge1b_output.json"\n\
    fi\n\
    \n\
    if [ -f "/app/output/challenge1b_output.json" ]; then\n\
        echo "📄 Results saved to: /app/output/challenge1b_output.json"\n\
        sections=$(python -c "import json; data=json.load(open(\"/app/output/challenge1b_output.json\")); print(len(data.get(\"extracted_sections\", [])))" 2>/dev/null || echo "?")\n\
        subsections=$(python -c "import json; data=json.load(open(\"/app/output/challenge1b_output.json\")); print(len(data.get(\"subsection_analysis\", [])))" 2>/dev/null || echo "?")\n\
        echo "📊 Generated: $sections sections, $subsections subsections"\n\
    fi\n\
else\n\
    echo "❌ Analysis failed. Check the logs above for details."\n\
    exit 1\n\
fi' > /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

VOLUME ["/app/input", "/app/output"]
ENTRYPOINT ["/app/entrypoint.sh"]