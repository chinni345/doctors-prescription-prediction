# doctors-prescription-prediction
Automatic image-based prediction and extraction of physicians' medical prescriptions is a key  development in the medical practice, enhancing productivity and reducing mistakes in medical  records.

Problem Statement

- Manual interpretation of handwritten prescriptions is prone to human error, misinterpretation, and delays. This system aims to solve that by:
- Automatically recognizing medications, dosages, and instructions from prescription images.
- Assisting healthcare professionals with faster, more reliable prescription digitizatio
  
Features

 - Upload handwritten or scanned prescriptions.
 - Automatically extract medication names, dosages, and instructions.
 - Displays extracted data with medication prices.
 - Converts prescriptions into structured digital formats for hospitals, pharmacies, and patients.

Technologies & Libraries Used

- Frontend: HTML, CSS (via Flask templates)
- Backend: Python, Flask
- Image Processing: OpenCV, PIL
- OCR: Tesseract OCR via PyTesseract
- NLP & ML:
- SpaCy (Named Entity Recognition)
- TF-IDF Vectorizer (for text vectorization)
- Cosine Similarity (for drug name matching)
- Data Handling: Pandas, NumPy
- Utilities: Regex, werkzeug, OS module
