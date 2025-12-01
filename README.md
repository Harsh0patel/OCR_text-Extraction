# 🔍 OCR Extraction System with Preprocessing, Pattern Correction & Streamlit UI

This project is a complete and production-ready OCR (Optical Character Recognition) solution built using:

- **Streamlit** (UI)
- **OpenCV** (preprocessing)
- **PaddleOCR / PPStructure** (OCR engine)
- **Custom Regex-based Pattern Correction**
- **JSON Export**
- **Real-time Visualization without Saving Images**

The system is optimized for **waybills, courier labels, invoices**, and similar documents where a strict pattern-based extraction is required.

---

# 🚀 Features

### ✔ 1. Streamlit User Interface
- Upload images (JPG/PNG)
- Real-time OCR execution
- View detection overlay **without saving images**
- Display extracted text, confidence & bounding boxes
- JSON results downloadable

---

### ✔ 2. Preprocessing Pipeline
The image preprocessing module supports:

- Resize (512 / 1024 or dynamic)
- Smoothing (Gaussian blur)
- Sharpening (custom kernel)
- Morphological opening, closing 
- Noise reduction
- Rotation correction (from PaddleOCR)

This improves OCR accuracy dramatically.

---

### ✔ 3. PaddleOCR Integration
We use PP-OCRv4 + PPStructure:

- Document preprocessing  
  (orientation classification, unwarp)
- Text detection
- Text recognition
- Textline orientation
- Visual overlay generation (`visualize()`)

The system extracts:
- `rec_texts`  
- `rec_scores`  
- `rec_boxes`  

and formats them into clean JSON.

---

### ✔ 4. Pattern Fixing (Custom Algorithm)

The model sometimes outputs patterns like:
"156387426414724544 1 wni"
But the expected format is:
[14–20 uppercase alphanumeric][1–2 digits][2–5 letters]
Pattern fixer converts noisy OCR output into:
"156387426414724544_1_WNI"


Handled cases:

- Spaces → underscore  
- Lowercase → uppercase  
- Removing unwanted characters  
- Normalizing digits  
- Validating final format  

---

### ✔ 5. JSON Output
Results are formatted as:

```json
{
  "text": "156387426414724544_1_WNI",
  "confidence": 0.98,
  "box": [x1, y1, x2, y2]
}
```
This JSON is displayed in Streamlit and can be downloaded.

✔ 6. Visual Output Without Saving

OCR detection visualization is shown using:
```python
vis = result[0].visualize(img)
vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
st.image(vis_rgb)
```

📂 Project Structure
```
project/
│
├── app.py                        # Streamlit UI
├── preprocessing.py              # All preprocessing functions
├── ocr_engine.py                 # PaddleOCR wrapper + visualization
├── pattern_fixer.py              # fix_underscore_pattern() & regex logic
├── utils.py                      # Helper functions (conversion, boxes)
├── test_images/                  # Sample images
├── requirements.txt
└── README.md
```

🔧 Installation

1️⃣ Clone repository

```bash
git clone https://github.com/yourname/ocr-system.git
cd ocr-system
```

2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

▶️ Running the App
```bash
streamlit run app.py
```

The UI will open automatically at:
```bash
http://localhost:8501
```

🧠 OCR Pipeline (Simplified)
```aiignore
(1) Upload Image (Streamlit)
         ↓
(2) Preprocessing (OpenCV)
         ↓
(3) PaddleOCR Document Processing
         ↓
(4) Text Detection + Recognition
         ↓
(5) Pattern Fixing & Validation
         ↓
(6) JSON Prepared
         ↓
(7) Visualization (No Saving)
         ↓
(8) Streamlit Output
```

💡 Example
Input OCR result:
```aiignore
"156387426414724544 1 wni"
```

Fixed Output:
```aiignore
"156387426414724544_1_WNI"
```
JSON Result
```json
[
  {
    "text": "156387426414724544_1_WNI",
    "confidence": 0.9803,
    "box": [1314, 1075, 1823, 1104]
  }
]
```

📌 Notes & Limitations

Very low-resolution or highly noisy images may require more preprocessing.

Pattern matcher is strict by design (built for logistics use-case).

Works best with PP-OCRv4 or PP-OCRv3 recognition models.

Model sometime fails on image which have pattern but don't recognize.

📞 Contact

harsh Patel

hp333854@gmail.com

+91-9157752911
