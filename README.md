# ♟️ Chessboard Analysis & Web Interface

Welcome to the **Chessboard Analysis & Web Interface** module.  
This component is part of a larger intelligent PDF analysis system and focuses on **detecting chessboard images** inside PDF documents and **visualizing the results through an interactive web dashboard**.

---

## 🎯 Module Objectives

The main goals of this module are:

- Detect **chessboard patterns** in images extracted from PDF files  
- Use a **machine learning model** to classify images automatically  
- Store detection results in a structured database  
- Provide a **user-friendly web interface** to explore and filter results  

This enables users to quickly identify which PDFs and pages contain chessboard images.

---

## 🤖 Chessboard Detection (Machine Learning Inference)

Chessboard detection is implemented in the script:

📄 `compute_chessboard_flags.py`

### 🔍 How It Works

1. Images are read from the database (`pdf_images`) as **BLOB data**
2. Only images that have not been processed yet  
   (`image_features.is_chessboard IS NULL`) are selected
3. Each image is preprocessed:
   - Converted to RGB  
   - Resized to **64 × 64**  
   - Normalized to the **[0–1]** pixel range
4. A **pre-trained ML classification model** predicts the probability of a chessboard
5. Based on a threshold value:
   - `is_chessboard = 1` → Chessboard detected  
   - `is_chessboard = 0` → No chessboard detected  

### 📊 Stored Outputs

The following fields are updated in the database:

- `image_features.is_chessboard`
- `image_features.chessboard_score`

No new images are created — existing images are **only labeled**.

---

## 🌐 Web Interface (Flask Dashboard)

The web interface is built using **Flask** and provides a clean, interactive dashboard for exploring analysis results.

### 🧩 Main Components

- **Flask Backend**
  - Database connection handling
  - Route definitions
  - Data preparation for the frontend

- **Dashboard View**
  - Dynamic listing of all database tables
  - Summary metrics (PDF count, image count, similarity statistics)
  - Interactive tables with search, sort, and pagination

- **PDF Detail Page**
  - Displays all images extracted from a selected PDF
  - Chessboard-based filters:
    - Only chessboard images
    - Non-chessboard images
  - Thumbnail previews for fast visual inspection

---

## ✨ Interface Features

- DataTables integration for advanced table interaction  
- Chart.js visualizations for summary insights  
- Responsive layout  
- Dark mode / Light mode toggle  
- Image preview modal with zoom and pan support  

---

## 🔄 Pipeline Integration

This module is fully integrated into the project pipeline.

Through `pipeline.py`, the **Chessboard Detection** step is executed automatically as part of the full analysis flow, ensuring seamless cooperation with other components such as OCR, text similarity, and image similarity analysis.

---

## ▶️ How to Run

### Run Chessboard Detection

```bash
python app/script/compute_chessboard_flags.py
