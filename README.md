# 🎓 EduTech - AI-Powered Question Generation System

An intelligent system for extracting educational content from PDFs and generating high-quality exam questions with contextual images, powered by Google Gemini AI.

## ✨ Features

- **📚 PDF Content Ingestion** - Deep analysis of educational PDFs including text, images, diagrams, and equations
- **🤖 Smart Question Generation** - AI-generated multiple-choice questions from stored content
- **📊 Capacity Preview** - Shows available question capacity before generation
- **🎨 Contextual Image Generation** - Auto-generate educational diagrams for questions
- **⚡ Batch Processing** - Automatic batching for large requests (>15 questions) to avoid token limits
- **🛡️ Anti-Hallucination** - Strict validation ensures questions only from ingested PDFs
- **🔄 Duplicate Prevention** - Automatic detection and filtering of similar questions (85% threshold)
- **💾 Persistent Storage** - Vector database (Qdrant) for efficient content retrieval
- **📈 CSV Export** - Ready-to-use format with all metadata and image paths
- **📊 Enriched Metadata** - Dual-layer system: simple CSV + comprehensive Qdrant storage
- **✅ Code Quality** - Linted with ruff, formatted, and fully tested

## 🏗️ System Architecture

### Two-Phase Design

```
┌──────────────────────────────────────────────────────────────┐
│                    PHASE 1: INGESTION                        │
│                                                              │
│  PDF Files  →  Gemini 2.5 Pro  →  Content Chunks  →  Qdrant │
│                  (Analysis)         (with metadata)          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│                  PHASE 2: GENERATION                         │
│                                                              │
│  Capacity Check  →  User Chooses  →  Generate Questions     │
│                                      (Gemini 2.5 Flash)      │
│                                      [Batch: 15/request]     │
│                                              ↓                │
│                    Generate Images  →  Export CSV            │
│                  (Gemini Flash Image)    (with paths)        │
└──────────────────────────────────────────────────────────────┘
```

### Project Structure

```
edutech/
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── models.py             # Data models (Question, PDFMetadata)
│   │   ├── embeddings.py         # Embedding generation
│   │   └── qdrant_client_wrapper.py  # Vector database operations
│   ├── generators/               # Question and image generators
│   │   ├── question_generator_v2.py  # Question generation with safeguards
│   │   └── image_generator.py        # Contextual image generation
│   └── utils/                    # Utility modules
│       ├── content_extractor.py  # PDF content extraction
│       ├── csv_exporter.py       # CSV export functionality
│       └── s3_uploader.py        # S3 image upload
├── input/                        # Input PDFs (expected structure)
│   └── gradeX/
│       └── subject/
│           └── *.pdf
├── output/                       # Generated questions
│   ├── *.csv                     # Question files
│   └── *_summary.txt             # Generation summaries
├── logs/                         # Log files
│   ├── ingest_content.log
│   └── generate_questions.log
├── generate_questions.py         # Main generation script
├── ingest_content.py             # Main ingestion script
├── requirements.txt              # Python dependencies
├── ruff.toml                     # Code quality configuration
├── .env                          # Environment configuration
└── README.md                     # This file
```

### Technology Stack

**AI Models**:
- Content Analysis: `gemini-2.5-pro` (comprehensive PDF understanding)
- Question Generation: `gemini-2.5-flash` (fast, cost-effective)
- Image Generation: `gemini-2.5-flash-image` (contextual visuals)
- Embeddings: `text-embedding-004` (768 dimensions)

**Storage**:
- Vector Database: Qdrant (self-hosted)
- Content Collections: `{grade}_content`
- Question Collections: `{grade}_questions`

**Code Quality**:
- Linter: ruff (all checks passing)
- Formatter: ruff format (consistent style)
- Testing: Integration tests passing

## 🚀 Installation

### Prerequisites

```bash
# Python 3.12+
python --version

# Conda environment (recommended)
conda create -n edutech python=3.12
conda activate edutech
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `google-generativeai>=0.3.0` - Google Gemini API client
- `qdrant-client>=1.7.0` - Vector database client
- `boto3>=1.34.0` - AWS SDK for S3 image storage
- `typing-extensions>=4.8.0` - Type hints support
- `python-dotenv>=1.0.0` - Environment variable management

### Setup Qdrant Vector Database

```bash
# Using Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
# Follow: https://qdrant.tech/documentation/quick-start/
```

### Environment Configuration

Create `.env` file in project root:

```bash
# Google Gemini API Key
GEMINI_API_KEY=your_api_key_here

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=  # Optional, leave empty for local

# S3 Configuration (Required for image generation)
# Supports AWS S3 and S3-compatible storage (MinIO, etc.)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name_here
S3_ENDPOINT_URL=https://your-s3-endpoint.com/  # Custom endpoint (optional)
```

#### S3 Setup for Image Storage

The system stores all generated images directly to S3-compatible storage (no local storage). Supports:
- **AWS S3** - Amazon's cloud storage
- **MinIO** - Self-hosted S3-compatible storage
- **Other S3-compatible** - Any service with S3 API compatibility

**For Custom S3 Endpoint (MinIO, self-hosted):**

1. **Configure Environment Variables:**
   ```bash
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   S3_BUCKET_NAME=edutech
   S3_ENDPOINT_URL=https://s3.your-domain.com/
   AWS_REGION=us-east-1  # Can be any value for MinIO
   ```

2. **Ensure Bucket Exists:**
   - Create bucket in your S3-compatible storage
   - Configure public read access if needed

3. **Verify Access:**
   ```bash
   # The script will automatically verify bucket access before uploading
   python generate_questions.py --grade grade5 --generate-images
   ```

**For AWS S3:**

1. **Create Bucket:**
   ```bash
   aws s3 mb s3://your-edutech-bucket --region us-east-1
   ```

2. **Configure Public Access:**
   ```bash
   aws s3api put-bucket-acl --bucket your-edutech-bucket --acl public-read
   ```

3. **Set Environment Variables:**
   - Omit `S3_ENDPOINT_URL` for standard AWS S3
   - Use your AWS credentials

**S3 Benefits:**
- ✅ No local storage needed
- ✅ Direct CDN-ready URLs
- ✅ Scalable and durable storage
- ✅ Easy sharing across systems
- ✅ Automatic metadata updates in Qdrant
- ✅ Supports self-hosted S3-compatible storage

### Create Required Directories

The scripts automatically create the `logs/` directory if it doesn't exist. For CSV output:

```bash
mkdir -p output
```

**Note:** Images are now stored directly in S3, so no local `output/images/` directory is needed.

## 🎯 Quick Start

### 1. Prepare Your PDFs

Organize PDFs in the following structure:

```
input/
├── grade4/
│   ├── numeracy/
│   │   └── math_workbook.pdf
│   └── english/
│       └── reading.pdf
├── grade5/
│   ├── numeracy/
│   │   └── maths.pdf
│   └── science/
│       └── biology.pdf
```

### 2. Ingest PDF Content

```bash
# Ingest all PDFs
python ingest_content.py

# Ingest specific grade
python ingest_content.py --grade grade5

# Ingest specific subject
python ingest_content.py --grade grade5 --subject numeracy
```

**Expected Output:**
```
✅ INGESTION SUMMARY
================================================================================
PDFs processed: 1/1
Total chunks stored: 13
Topics identified: 13
Concepts extracted: 78
================================================================================
```

### 3. Generate Questions (Interactive with Capacity Preview)

```bash
# Interactive mode - shows capacity and prompts for number
python generate_questions.py --grade grade5
```

**Expected Interaction:**
```
CAPACITY CHECK
================================================================================
Content chunks found: 13
Estimated capacity: ~52 questions
Already generated: 0 questions
Available to generate: ~52 new questions
================================================================================

💡 You can generate approximately 52 new questions.
How many questions would you like to generate? 10

GENERATING 10 QUESTIONS
================================================================================
...
✅ GENERATION COMPLETE
Questions: 10
CSV file: output/questions_grade5_20251127_001155.csv
```

### 4. Generate with Images

```bash
python generate_questions.py --grade grade5 --generate-images
```

**Output Files:**
- `output/questions_grade5_TIMESTAMP.csv` - Question data with S3 URLs
- `output/questions_grade5_TIMESTAMP_summary.txt` - Statistics
- Images uploaded to S3: `https://your-endpoint/bucket/edutech/images/grade5/subject/q*.png`

### 5. Batch Generation (Large Requests)

```bash
# Request 50 questions - automatically batched into groups of 15
python generate_questions.py --grade grade5 --num 50 --generate-images --no-preview
```

**Batch Processing:**
```
Batch 1/4: Generating 15 questions...
✅ Generated 15 questions

Batch 2/4: Generating 15 questions...
✅ Generated 10 questions (5 duplicates prevented)

Batch 3/4: Generating 15 questions...
⚠️  Content exhausted after 25 questions

GENERATION STATISTICS
================================================================================
Status: PARTIAL
Questions generated: 25
Duplicates prevented: 5
================================================================================
```

## 📖 Detailed Usage

### Content Ingestion Options

```bash
# All options
python ingest_content.py \
  --root-dir input/ \
  --grade grade5 \
  --subject numeracy \
  --gemini-model gemini-2.5-pro \
  --log-level INFO

# If PDFs are in a different location
python ingest_content.py \
  --root-dir /path/to/pdfs \
  --grade grade5
```

### Question Generation Options

```bash
# Specify number directly (skip preview)
python generate_questions.py --grade grade5 --num 10 --no-preview

# Filter by subject and difficulty
python generate_questions.py \
  --grade grade5 \
  --subject numeracy \
  --difficulty hard

# Filter by specific topics
python generate_questions.py \
  --grade grade5 \
  --topics "Addition" "Fractions" "Decimals"

# Generate with images
python generate_questions.py \
  --grade grade5 \
  --num 10 \
  --generate-images \
  --image-style "colorful educational diagram"

# Custom output location
python generate_questions.py \
  --grade grade5 \
  --num 10 \
  --output exams/midterm_2025.csv

# Dump all existing questions from Qdrant to CSV (no generation)
python generate_questions.py --grade grade5 --dump

# Dump with filters
python generate_questions.py \
  --grade grade5 \
  --dump \
  --subject numeracy \
  --difficulty hard

# Regenerate images for existing questions and update with S3 URLs
python generate_questions.py --grade grade5 --regenerate-images

# Regenerate images with filters
python generate_questions.py \
  --grade grade5 \
  --regenerate-images \
  --subject Mathematics
```

## 📊 Output Format

### CSV Fields (19 fields)

| Field | Description | Example |
|-------|-------------|---------|
| `question_number` | Sequential number | 1, 2, 3... |
| `year` | Year generated | 2025 |
| `class` | Grade level (display) | "Grade 5" |
| `grade` | Grade identifier | "grade5" |
| `subject` | Main subject | "numeracy" |
| `sub_subject` | Specific topic | "Addition, Subtraction" |
| `question_text` | The question | "What is 25 + 37?" |
| `option_1` to `option_4` | Answer choices | "52", "62", "72", "82" |
| `answer` | Correct answer | "62" |
| `answer_index` | Index of correct answer (0-3) | 1 |
| `question_image` | Image filename pattern | "2025_grade5_numeracy_q1.png" |
| **`file_path`** | **Path to source PDF** | **"maths.pdf"** |
| `pdf_source` | Source PDF filename | "maths.pdf" |
| `page_number` | Source page in PDF | 5 |
| **`artifacts`** | **S3 URLs of generated images (JSON)** | **["https://s3.example.com/edutech/edutech/images/grade5/math/q1.png"]** |
| `last_generated` | ISO timestamp | "2025-11-27T04:45:39.998117" |

### CSV Example

```csv
question_number,year,class,grade,subject,sub_subject,question_text,option_1,option_2,option_3,option_4,answer,answer_index,question_image,file_path,pdf_source,page_number,artifacts,last_generated
1,2025,Grade 5,grade5,maths,Fractions,"In a fraction, what does the 'denominator' represent?",The number of parts being considered.,The total number of equal parts the whole is divided into.,The line between the numerator and denominator.,The whole object or collection.,The total number of equal parts the whole is divided into.,1,2025_grade5_maths_q1.png,maths.pdf,maths.pdf,256,[],2025-11-27T04:45:39.998117
```

### Enriched Metadata (Qdrant Only - 33+ fields)

In addition to CSV fields, Qdrant stores enriched metadata:

**Quality Metrics:**
- `ocr_confidence` - Confidence score (0-1)
- `validation_status` - "validated", "flagged", "rejected"

**Generation Metadata:**
- `reasoning_steps` - AI generation steps
- `token_count` - Tokens used

**Content Analysis:**
- `question_context` - Brief context
- `question_complexity` - "easy", "medium", "hard"
- `references` - Referenced concepts
- `source_chunk` - Content chunk ID

**Image Tracking:**
- `images_tagged_count` - Number of images
- `images_path` - All image paths
- `artifacts_path` - All artifact paths

**Classification:**
- `tags` - Classification tags
- `processing_status` - "success", "partial", "failed"
- `issues_found` - Validation issues

## 🛡️ Safeguards & Quality

### Anti-Hallucination System

1. **Source Validation** - Each question references specific content chunk, PDF, and page
2. **Strict Prompts** - AI instructed to use ONLY provided content
3. **Artifact Validation** - Only real images from PDFs or generated images
4. **Traceability** - Every question traceable to source via `source_chunk` field

### Duplicate Prevention

- **Algorithm**: Cosine similarity on question embeddings
- **Threshold**: 85% (configurable)
- **Process**: All existing questions checked before generation
- **Cross-Batch**: Works across multiple batches in large requests

**Example Output:**
```
Batch 1/4: Generated 15 questions (0 duplicates)
Batch 2/4: Generated 10 questions (5 duplicates prevented)
Total duplicates prevented: 5
```

### Batch Generation System

- **Batch Size**: 15 questions per batch (configurable)
- **Automatic Splitting**: Requests >15 questions automatically batched
- **Token Management**: Prevents API token limit errors
- **Sequential Processing**: Maintains duplicate detection across batches
- **Graceful Degradation**: Handles content exhaustion mid-batch

**Benefits:**
- ✅ No JSON parsing errors on large requests
- ✅ Consistent duplicate prevention
- ✅ Sequential question numbering
- ✅ Content exhaustion detection

### Content Exhaustion Detection

- **Estimation**: ~4 unique questions per content chunk
- **Tracking**: Monitors generation history
- **Warnings**: Alerts when capacity reached
- **Status Types**: SUCCESS, PARTIAL, FAILED

**Example Warning:**
```
⚠️  Content exhausted after 25 questions
Status: PARTIAL
```

## ✅ Testing & Verification

### Run Tests

```bash
# Test ingestion
python ingest_content.py --grade grade5

# Expected: ✅ PDFs processed: 1/1

# Test question generation (without images)
python generate_questions.py --grade grade5 --num 3 --no-preview

# Expected: ✅ Questions: 3

# Test with images
python generate_questions.py --grade grade5 --num 2 --generate-images --no-preview

# Expected: ✅ Generated 2/2 images
```

### Code Quality Check

```bash
# Install ruff (if not already installed)
pip install ruff

# Run linter
ruff check .

# Expected: All checks passed!

# Format code
ruff format .

# Expected: N files reformatted
```

### Test Results (Verified ✅)

All tests passing as of 2025-11-27:

| Test | Status | Details |
|------|--------|---------|
| Content Ingestion | ✅ PASS | 13 chunks, 78 concepts extracted |
| Question Generation | ✅ PASS | 3 questions generated successfully |
| Image Generation | ✅ PASS | 2/2 images generated |
| Batch Processing | ✅ PASS | 50→25 questions (content exhausted) |
| Duplicate Detection | ✅ PASS | 5 duplicates prevented |
| Code Quality | ✅ PASS | All ruff checks passing |

## 💡 Best Practices

### For Better Question Quality

1. **Ingest high-quality PDFs** with clear content structure
2. **Use specific filters** (subject, topics) for focused questions
3. **Start small** - Test with 5-10 questions first
4. **Check capacity preview** before generating
5. **Generate images selectively** (adds time and cost)

### For System Performance

1. **Batch processing** - System handles large requests automatically
2. **Reuse content** - Generated questions stored and reused
3. **Monitor capacity** - Use preview to avoid exhaustion
4. **Check logs** - Review `logs/` directory for issues

### For Large Requests

1. **Use --no-preview** for automated workflows
2. **Request conservatively** - System will batch automatically
3. **Monitor batch progress** - Check logs for batch statistics
4. **Handle PARTIAL status** - Content may exhaust before completion

## 🔧 Troubleshooting

### No Questions Generated

```bash
# Error: "No content found"
# Solution: Ingest PDFs first
python ingest_content.py --grade grade5
```

### Content Exhausted

```bash
# Warning: "Content exhausted after N questions"
# Solutions:

# A) Ingest more PDFs
python ingest_content.py --grade grade5

# B) Remove filters
python generate_questions.py --grade grade5  # No subject/topic filter

# C) Request fewer questions
python generate_questions.py --grade grade5 --num 10
```

### Qdrant Connection Error

```bash
# Error: "Failed to connect to Qdrant"
# Solution: Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant
```

### Import Errors

```bash
# Error: "ModuleNotFoundError"
# Solution: Ensure Python path includes src/
# The scripts automatically add src/ to sys.path
```

### FileNotFoundError: logs/

```bash
# Error: "No such file or directory: logs/"
# Solution: This issue is now fixed - scripts automatically create the logs directory
# If you still encounter this, manually create it:
mkdir -p logs
```

## 📊 Performance & Costs

### Processing Time

- **Ingestion**: ~90 seconds per 13-page PDF
- **Question Generation** (without images): ~15-20 seconds for 3 questions
- **Image Generation**: ~6-10 seconds per image
- **Batch Processing**: ~30-40 seconds per 15-question batch

### API Usage (Approximate)

- **Content Ingestion** (1 PDF, 13 pages): ~$0.30-$0.50 per PDF (gemini-2.5-pro)
- **Question Generation** (10 questions): ~$0.05 (gemini-2.5-flash)
- **Image Generation** (per image): ~$0.02 per image (gemini-2.5-flash-image)

### Batch Generation Efficiency

- **Token Savings**: Batching prevents failed large requests
- **Time Overhead**: ~2-3 seconds per batch (minimal)
- **Duplicate Detection**: No performance impact (<100ms per check)

## 🔍 What's New

### Latest Version (v2.1)

✅ **S3 Image Storage** - All images uploaded directly to AWS S3 (no local storage)
✅ **S3 URL Integration** - Artifacts field contains public S3 URLs
✅ **Automatic Bucket Verification** - Validates S3 access before uploading
✅ **CDN-Ready URLs** - Public URLs ready for web applications
✅ **Batch Generation System** - Automatic batching for requests >15 questions
✅ **Capacity Preview** - See available questions before generating
✅ **Interactive Mode** - Choose number of questions interactively
✅ **Code Quality** - All ruff checks passing, formatted code
✅ **Testing Verified** - All integration tests passing
✅ **Clean Structure** - Removed archived scripts and docs
✅ **Improved Logging** - Better progress tracking and error messages
✅ **Dual-Layer Metadata** - Simple CSV + enriched Qdrant storage
✅ **Correct Field Mapping** - `file_path`=PDF source, `artifacts`=S3 URLs

### Bug Fixes

✅ Fixed Path type issue in content extraction
✅ Fixed method name mismatch (embed_content_chunks)
✅ Improved JSON parsing robustness
✅ Fixed markdown code block handling
✅ **Auto-create logs directory** - Scripts now automatically create logs/ directory to prevent crashes

## 📚 Examples

### Python - Access Generated Images

```python
import csv
import json
import requests
from PIL import Image
from io import BytesIO

with open('output/questions_grade5_20251127_001246.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Q{row['question_number']}: {row['question_text']}")
        print(f"Source: {row['file_path']} (page {row['page_number']})")

        # Access generated images from S3
        artifacts = json.loads(row['artifacts']) if row['artifacts'] else []
        for s3_url in artifacts:
            if s3_url:  # Check if URL exists
                # Download and display image from S3
                response = requests.get(s3_url)
                img = Image.open(BytesIO(response.content))
                img.show()
                print(f"Image URL: {s3_url}")
```

### Command Line - View Capacity

```bash
# Just check capacity without generating
python generate_questions.py --grade grade5 --num 0
```

### Bash - Automated Batch Processing

```bash
#!/bin/bash
# Generate 100 questions in batches automatically
python generate_questions.py \
  --grade grade5 \
  --num 100 \
  --generate-images \
  --no-preview

# System will:
# - Split into 7 batches (15+15+15+15+15+15+10)
# - Prevent duplicates across batches
# - Handle content exhaustion gracefully
```

## 🤝 Contributing

This project is designed for educational content generation. When contributing:

1. Maintain the anti-hallucination safeguards
2. Test with real PDFs before submitting
3. Update documentation for any new features
4. Follow the existing code structure
5. Run `ruff check .` before committing

## 📄 License

Proprietary - For educational use only.

---

**Questions or Issues?** Check the logs in `logs/` directory for detailed debugging information.

**Happy Question Generating! 🎓✨**
# eduTech
