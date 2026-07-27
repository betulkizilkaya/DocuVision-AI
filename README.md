# 📄✨ DocuVision AI

<div align="center">

### A Multimodal AI-Powered Document Analysis System

**PDF Processing · NLP · Computer Vision · OCR · Document Classification · Chess Analysis**

</div>

---

## 🌟 About the Project

**DocuVision AI** is a multimodal document analysis platform designed to extract, analyze, classify, and visualize textual and visual content from PDF documents.

The system combines natural language processing, machine learning, deep learning, optical character recognition, and image processing techniques within a single platform.

Although the current implementation focuses primarily on chess-related documents, the architecture can be adapted to different document analysis scenarios.

DocuVision AI can:

* Extract text and embedded images from PDF documents
* Classify documents according to their content
* Calculate text and image similarity
* Detect and crop chessboards
* Generate FEN notation from chessboard images
* Extract chess game notation using OCR
* Detect people, logos, and notation content in images
* Extract named entities from document text
* Store analysis results in a relational database
* Present all results through an interactive web dashboard

---

## 🎯 Project Objectives

PDF documents may contain multiple types of information, including text, images, tables, diagrams, logos, people, chessboards, and game notation.

Analyzing only the textual or visual content is therefore not sufficient for many real-world document analysis tasks.

DocuVision AI provides an integrated solution by combining:

* 📄 PDF processing
* 🧠 Natural language processing
* 🖼️ Computer vision
* 🤖 Machine learning
* 🔍 Similarity analysis
* 🔤 Optical character recognition
* ♟️ Chess-specific content analysis
* 🌐 Web-based result visualization

---

## 🚀 Main Features

### 📄 PDF Processing

The system automatically processes PDF documents and extracts their textual and visual content.

Main operations include:

* PDF file indexing
* SHA-256 duplicate detection
* Page-based text extraction
* Embedded image extraction
* Image and text storage in SQLite
* Document and page relationship tracking

---

### 🗂️ Document Classification

Documents are classified according to their textual content.

Supported document classes include:

* `educational_chess`
* `tournament_report`
* `image_only_or_empty`

The classification pipeline uses:

* TF-IDF feature extraction
* Linear Support Vector Machine
* Chunk-based document processing
* PDF-level training and test separation
* Keyword-based control for low-text documents

The final document classification model achieved approximately **96% accuracy**.

---

### 📝 Text Similarity Analysis

DocuVision AI uses a hybrid approach to compare text extracted from PDF documents.

Supported similarity methods include:

* Levenshtein similarity
* Jaro-Winkler similarity
* Sørensen-Dice coefficient
* Jaccard similarity
* TF-IDF cosine similarity

The text processing pipeline also includes:

* Unicode NFKC normalization
* Turkish character support
* Text cleaning
* Custom tokenization
* Noise filtering

Using multiple similarity metrics provides more reliable results than relying on a single method.

---

### 🖼️ Image Similarity Analysis

Images extracted from PDF documents are compared using multiple structural, perceptual, and feature-based methods.

Supported methods include:

* SSIM
* Perceptual Hash — pHash
* ORB
* AKAZE

The final similarity decision is produced by evaluating multiple scores together.

This hybrid structure makes the system more resistant to:

* Resolution differences
* Small cropping changes
* Alignment differences
* Contrast changes
* Scaling
* Minor visual transformations

---

### ♟️ Chessboard Detection

The system automatically identifies chessboard images extracted from PDF documents.

The detection pipeline includes:

* CNN-based image classification
* YOLO-based object detection
* Confidence score calculation
* Threshold-based decision logic
* Hard-negative sample training
* Real PDF image evaluation

The final CNN model was trained using a balanced dataset containing:

* 4,000 chessboard images
* 4,000 non-chessboard images

The model achieved approximately:

* **98.3% accuracy on the training evaluation**
* **94.2% accuracy on unseen PDF images**

---

### ✂️ Chessboard Detection and Cropping

A PDF image may contain one or multiple chessboards.

DocuVision AI uses several techniques to locate and crop these chessboards:

* YOLO object detection
* Corner detection
* Region of interest analysis
* Hough-Line detection
* Border-based detection
* Contour analysis
* Fallback cropping

The cropped chessboards are stored and prepared for FEN generation.

---

### 🧩 FEN Generation

The system attempts to convert cropped chessboard images into **Forsyth–Edwards Notation — FEN**.

Each chessboard is divided into an 8×8 grid.

Every square is classified as one of the following:

* White king
* White queen
* White rook
* White bishop
* White knight
* White pawn
* Black king
* Black queen
* Black rook
* Black bishop
* Black knight
* Black pawn
* Empty square

The predicted board position is then converted into a FEN string.

---

### 🔤 Chess Notation OCR

DocuVision AI extracts chess game notation from scanned pages and document images.

The OCR pipeline includes:

1. Image preprocessing
2. Grayscale conversion
3. Thresholding
4. Noise reduction
5. Chessboard masking
6. Text region detection
7. Column separation
8. Line-based OCR
9. Character normalization
10. Chess notation correction
11. Candidate block scoring
12. Move validation with `python-chess`
13. Best candidate selection

The system uses **Tesseract OCR** with Turkish and English language support.

---

### 🧑 Named Entity Recognition

The system performs Named Entity Recognition on extracted document text.

Supported entity types include:

* `PERSON`
* `ORG`
* `LOC`
* `MISC`

Detected person names are:

* Normalized
* Deduplicated
* Connected to their source documents
* Stored in relational database tables

The NER module uses a multilingual spaCy model.

---

### 👤 Visual Content Analysis

Extracted images are analyzed according to their semantic content.

The system can identify:

* People
* Logos
* Chessboards
* Chess game notation
* Other visual content

The visual analysis pipeline uses:

* YOLOv8 for person detection
* OCR and regular expressions for notation detection
* Rule-based and feature-based methods for logo detection
* Confidence-based final labeling

---

## 🏗️ System Architecture

DocuVision AI consists of three main layers.

### 1. Data Processing Layer

Responsible for:

* PDF indexing
* Text extraction
* Image extraction
* Image preprocessing
* Text normalization
* Feature extraction

### 2. Analysis Layer

Responsible for:

* Document classification
* Text similarity
* Image similarity
* Chessboard detection
* Chessboard cropping
* FEN generation
* OCR
* Named Entity Recognition
* Visual content analysis

### 3. Presentation Layer

Responsible for:

* Flask web application
* Dashboard metrics
* Document detail pages
* Searchable tables
* Image preview modals
* Analysis result visualization
* Document and image filtering

---

## 🛠️ Technologies

### Backend

* Python
* Flask
* SQLite
* Jinja2

### Machine Learning and AI

* Scikit-learn
* TensorFlow / Keras
* YOLOv8
* Ultralytics
* spaCy
* python-chess

### PDF Processing

* pdfPlumber
* PyMuPDF

### Image Processing

* OpenCV
* Pillow
* NumPy
* scikit-image
* ImageHash

### Text Processing

* RapidFuzz
* TextDistance
* TF-IDF
* Unicode normalization

### OCR

* Tesseract OCR
* PyTesseract

### Frontend

* HTML
* CSS
* JavaScript
* Bootstrap
* DataTables
* Chart.js

---

## 📁 Project Structure

```text
DocuVision-AI/
│
├── app/
│   ├── core/          # Database operations, paths, and pipeline utilities
│   ├── image/         # OCR, image similarity, feature extraction, and preprocessing
│   ├── ml/            # Dataset creation and model training scripts
│   ├── model/         # Chessboard detection, FEN generation, and legacy models
│   ├── namedEntity/   # Named entity recognition modules
│   ├── script/        # End-to-end PDF and chess OCR workflows
│   ├── text/          # Document classification and text similarity
│   └── web/           # Flask web application, templates, and static files
│
├── data/              # Datasets, sample PDFs, and trained models
├── db/                # Database files
├── temp/              # Temporary processing files
├── semantic_labeled_images/
├── requirements.txt
├── yolov8s.pt
└── README.md```


```markdown
> The repository contains experimental scripts, trained model files, datasets, legacy implementations, and intermediate outputs developed during the project lifecycle. Some generated or large files may be excluded from version control.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/betulkizilkaya/DocuVision-AI.git
cd DocuVision-AI
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux or macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔤 Tesseract OCR Installation

Tesseract OCR must be installed separately because it is not installed through `pip`.

The following language packages are recommended:

* English — `eng`
* Turkish — `tur`

On Windows, the Tesseract executable path may need to be configured manually:

```python
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
```

---

## 🧠 spaCy Model Installation

Install the multilingual NER model:

```bash
python -m spacy download xx_ent_wiki_sm
```

---

## ▶️ Usage

The project has a modular structure. Individual analysis stages can be executed separately.

### Extract Text and Images from PDF Documents

```bash
python app/script/pdf_extract.py
```

### Classify PDF Documents

```bash
python app/script/classify_pdfs.py
```

### Detect Chessboard Images

```bash
python app/script/compute_chessboard_flags.py
```

### Run YOLO-Based Chessboard Detection

```bash
python app/model/yolo_run.py
```

### Run Chess Notation OCR

```bash
python app/script/run_chess_notation_ocr.py
```

### Run the Full Chess OCR Pipeline

```bash
python app/script/run_full_chess_ocr.py
```

### Run Named Entity Recognition

```bash
python app/namedEntity/run_ner.py
```

### Generate and Store FEN Results

```bash
python app/model/add_multi_fen_to_db.py
```

### Start the Web Application

```bash
python app/web/app.py
```

Open the local address displayed in the terminal, usually:

```text
http://127.0.0.1:5000
```

Some scripts require trained model files, input PDF documents, dataset folders, or configured database paths.

---

## 🌐 Web Dashboard

The Flask-based dashboard provides a central interface for reviewing analysis results.

Users can:

* View processed PDF documents
* Review document classification results
* Browse extracted images
* Filter chessboard and non-chessboard images
* Inspect OCR outputs
* View generated FEN notation
* Review person, logo, and notation labels
* Compare text similarity results
* Compare image similarity results
* Open images in preview modals
* Search, filter, sort, and paginate result tables
* Review summary statistics through dashboard cards

---

## 🗄️ Database Structure

DocuVision AI uses SQLite to store extracted data and analysis results.

Main database tables include:

| Table              | Description                                        |
| ------------------ | -------------------------------------------------- |
| `file_index`       | Stores indexed PDF documents                       |
| `text_lines`       | Stores text extracted from PDF pages               |
| `pdf_images`       | Stores images extracted from PDF documents         |
| `image_features`   | Stores image properties and classification results |
| `text_similarity`  | Stores text similarity scores                      |
| `image_similarity` | Stores image similarity scores                     |
| `entities_raw`     | Stores raw entities detected by the NER model      |
| `persons`          | Stores normalized and unique person names          |
| `person_mentions`  | Stores person-to-document relationships            |
| `chess_ocr`        | Stores chess notation OCR results                  |
| `fen_results`      | Stores generated FEN notation                      |

Database table names may differ slightly between experimental project versions.

---

## 📊 Experimental Results

### Document Classification

The final document classification dataset contained:

* 66 PDF documents
* 5,598 text chunks
* PDF-level training and test separation

Final accuracy:

```text
Approximately 96%
```

### Chessboard Detection

The final CNN dataset contained:

* 4,000 chessboard images
* 4,000 non-chessboard images
* 8,000 images in total

Results:

```text
Training evaluation accuracy: approximately 98.3%
Unseen PDF image accuracy: approximately 94.2%
```

### Image Similarity

The improved SSIM, pHash, ORB, and AKAZE decision mechanism detected significantly more valid similar image pairs than the original single-threshold approach.

---

## ⚠️ Limitations

* OCR performance may decrease on low-resolution or highly compressed images.
* Complex page layouts may make notation region detection more difficult.
* Decorative or partially visible chessboards may produce false-positive results.
* FEN accuracy depends on chess piece classification performance.
* Image-only PDF documents require OCR before text-based classification.
* General-purpose NER models may incorrectly classify some chess terminology.
* Some datasets and trained model files may not be included because of their size.
* Local file paths or model paths may require configuration before execution.

---

## 👩‍💻 Project Team

<p>
  <strong>Betül Kızılkaya</strong> &nbsp;•&nbsp;
  <strong>Hiranur Akpınar</strong> &nbsp;•&nbsp;
  <strong>Selin Kübra Şimşek</strong>
</p>

### 🎓 Academic Advisor

* **Prof. Dr. Selçuk Kavut**

### 🏫 Institution

**Balıkesir University**
Faculty of Engineering
Department of Computer Engineering

---

## 📚 Academic Project

This project was developed as a Computer Engineering graduation project at Balıkesir University.

### Project Title

> **Multimodule AI-Powered Document Analysis System**

---

## 📜 License

This project was developed for educational and academic research purposes.

Please contact the repository owners before using, distributing, or modifying the project for commercial purposes.
