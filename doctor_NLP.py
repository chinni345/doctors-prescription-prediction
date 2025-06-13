from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os
import pytesseract
from PIL import Image
import re
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configure file upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

# Set Tesseract path (Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load NLP Model (SpaCy)
nlp = spacy.load("en_core_web_sm")

# Load & Preprocess Medicine Data
def load_medicine_data():
    df = pd.read_csv('updated_indian_medicine_data.csv', encoding='utf-8')
    
    # Clean column names
    df.columns = df.columns.str.strip()  # Remove spaces from column names
    
    # Print column names to check for the issue
    print(df.columns)  # Add this line to inspect the columns
    
    # Check if the necessary columns exist
    if 'name' not in df.columns or 'price' not in df.columns:
        raise KeyError("Required columns 'name' or 'price' are missing.")
    
    df = df[['name', 'price']].dropna()  # Keep only relevant columns
    df['name'] = df['name'].str.lower().str.strip()
    return df

medicine_df = load_medicine_data()

# Train TF-IDF for better matching
vectorizer = TfidfVectorizer()
medicine_tfidf = vectorizer.fit_transform(medicine_df['name'])

# Check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Extract text using Tesseract OCR
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img)
        return extracted_text
    except Exception as e:
        return str(e)

# Clean extracted text
def clean_extracted_text(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip().lower()
    return cleaned_text

# Detect medicines using NLP (NER + TF-IDF Matching)
def detect_medicines(text):
    detected_medicines = []
    doc = nlp(text)  # Run NLP on extracted text
    extracted_names = [ent.text.lower() for ent in doc.ents]

    # Match extracted names with medicine dataset using TF-IDF
    for name in extracted_names:
        name_tfidf = vectorizer.transform([name])
        similarity = cosine_similarity(name_tfidf, medicine_tfidf)
        best_match_index = similarity.argmax()
        best_match_score = similarity.max()

        if best_match_score > 0.7:  # Threshold for accuracy
            matched_medicine = medicine_df.iloc[best_match_index]
            detected_medicines.append((matched_medicine['name'].capitalize(), f"₹{matched_medicine['price']}"))

    return detected_medicines

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Image upload and processing route
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract text
            extracted_text = extract_text_from_image(filepath)
            cleaned_text = clean_extracted_text(extracted_text)

            # Detect medicines using NLP
            detected_medicines = detect_medicines(cleaned_text)

            # Save extracted text
            text_filename = f"extracted_text_{filename.split('.')[0]}.txt"
            output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], text_filename)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)

            # Render results
            return render_template('extracted_text.html', extracted_text=cleaned_text, detected_medications=detected_medicines, file_path=text_filename)

    return render_template('upload.html')

# Route for downloading extracted text file
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
