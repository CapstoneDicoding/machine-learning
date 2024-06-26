# -*- coding: utf-8 -*-
"""ocr_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tfDMU414-BXAPo8EZvbdbhiX6GTWRmNs
"""

!apt-get update
!apt-get install -y tesseract-ocr

pip install pytesseract

!pip install wordcloud matplotlib

pip install Sastrawi

pip install pdf2image

!apt-get update
!apt-get install -y poppler-utils

"""# Simpan File di Drive"""

from google.colab import drive
drive.mount('/content/drive')

# Create a folder in the root directory
!mkdir -p "/content/drive/My Drive/OCR Data"

from google.colab import drive
import shutil
import os

def mount_drive():
    drive.mount('/content/drive')

def save_to_drive(source_path, target_path):
    if not os.path.isdir('/content/drive'):
        mount_drive()

    if os.path.isfile(source_path):
        shutil.copy(source_path, target_path)
        print(f"File '{source_path}' has been copied to '{target_path}'.")
    elif os.path.isdir(source_path):
        shutil.copytree(source_path, target_path)
        print(f"Directory '{source_path}' has been copied to '{target_path}'.")
    else:
        print(f"Source path '{source_path}' does not exist.")

# Mount Google Drive
mount_drive()

# Save a single file to Google Drive
# save_to_drive('/content/cv_data.csv', '/content/drive/My Drive/OCR Data/cv_data.csv')

# Save an entire directory to Google Drive
# save_to_drive('/content/clean_images', '/content/drive/My Drive/OCR Data/clean_images')

"""# Simpan File di Lokal"""

from google.colab import files
import shutil
import os

def download_from_colab(source_path):
    if os.path.isfile(source_path):
        files.download(source_path)
        print(f"File '{source_path}' is being downloaded.")
    elif os.path.isdir(source_path):
        zip_path = source_path + '.zip'
        shutil.make_archive(source_path, 'zip', source_path)
        files.download(zip_path)
        print(f"Directory '{source_path}' is being downloaded as '{zip_path}'.")
    else:
        print(f"Source path '{source_path}' does not exist.")

# Download a single file
download_from_colab('/content/cv_data.csv')

# Download an entire directory
download_from_colab('/content/clean_images')

"""# Import Library"""

from PIL import Image
from google.colab import files
import pytesseract
from pdf2image import convert_from_path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import shutil
import re
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

"""# Remove Folder"""

def remove_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f'Folder {folder_path} berhasil dihapus.')
    else:
        print(f'Folder {folder_path} tidak ditemukan.')

# Contoh pemanggilan fungsi
folder_path = '/content/cv_raw_data'
remove_folder(folder_path)

"""# Upload File"""

def upload_and_save_file(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Direktori {directory} berhasil dibuat.')

    uploaded = files.upload()
    for filename, file_content in uploaded.items():
        full_path = os.path.join(directory, filename)
        with open(full_path, 'wb') as f:
            f.write(file_content)

        print(f'File {filename} berhasil disimpan di {full_path}.')

    # Hapus file yang secara otomatis disimpan di direktori Colab
    for filename in uploaded.keys():
        os.remove(filename)

upload_and_save_file('/content/cv_raw_data')

"""# Convert pdf to image"""

def convert_pdf_to_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            images = convert_from_path(pdf_path)
            pdf_name = os.path.splitext(filename)[0]
            pdf_output_dir = os.path.join(output_dir, pdf_name)
            if not os.path.exists(pdf_output_dir):
                os.makedirs(pdf_output_dir)
            for i, image in enumerate(images):
                image_path = os.path.join(pdf_output_dir, f'{pdf_name}_page_{i + 1}.jpg')
                image.save(image_path, 'JPEG')
                print(f'Page {i + 1} dari file {filename} berhasil dikonversi dan disimpan di {image_path}.')

convert_pdf_to_images('/content/cv_raw_data', '/content/cv_images')

"""# Image Preprocessing"""

def image_preprocessing(img_path, cv_name, output_dir):
    # Load image
    img = cv2.imread(img_path)

    # Rescaling the image
    img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

    # Converting image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Applying dilation and erosion to remove noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Applying blur
    img = cv2.medianBlur(img, 3)

    # Thresholding
    img = cv2.threshold(img, 65, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Create CV directory if not exists
    cv_dir = os.path.join(output_dir, cv_name)
    if not os.path.exists(cv_dir):
        os.makedirs(cv_dir)

    # Save processed image
    image_name = os.path.splitext(os.path.basename(img_path))[0] + '_processed.jpg'
    output_path = os.path.join(cv_dir, image_name)
    cv2.imwrite(output_path, img)

    return output_path

# Fungsi untuk memproses semua gambar CV dalam direktori input dan menyimpan hasilnya ke direktori output
def process_cv_images(input_dir, output_dir):
    output_clean_images_dir = os.path.join(output_dir, 'clean_images')
    if not os.path.exists(output_clean_images_dir):
        os.makedirs(output_clean_images_dir)

    for cv_name in os.listdir(input_dir):
        cv_dir = os.path.join(input_dir, cv_name)
        if os.path.isdir(cv_dir):
            for filename in os.listdir(cv_dir):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(cv_dir, filename)
                    processed_image_path = image_preprocessing(image_path, cv_name, output_clean_images_dir)
                    print(f'Image {filename} dari CV {cv_name} telah diproses dan disimpan di {processed_image_path}.')

output_dir = '/content'
process_cv_images('/content/cv_images', output_dir)

"""# Extract text from image"""

def extract_text_from_image(image_path):
    # Load preprocessed image
    img = cv2.imread(image_path)

    # Extract text using Tesseract OCR
    extracted_text = pytesseract.image_to_string(img)

    return extracted_text

def process_images_to_dataframe(input_dir):
    data = []
    for cv_name in os.listdir(input_dir):
        cv_dir = os.path.join(input_dir, cv_name)
        if os.path.isdir(cv_dir):
            cv_text = ""
            for filename in os.listdir(cv_dir):
                if filename.endswith('_processed.jpg'):
                    image_path = os.path.join(cv_dir, filename)
                    text = extract_text_from_image(image_path)
                    cv_text += text + "\n\n"  # Gabungkan teks dari setiap halaman
            data.append({'CV Name': cv_name, 'Text': cv_text})

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

input_dir = '/content/clean_images'
df = process_images_to_dataframe(input_dir)

df

"""# Simpan Data dalam CSV"""

def dataframe_to_csv(df, output_dir, filename):
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f'DataFrame berhasil disimpan sebagai CSV di: {output_path}')

output_dir = '/content'
filename = 'cv_data.csv'
dataframe_to_csv(df, output_dir, filename)

"""# Persebaran Kata"""

def create_word_cloud(df, text_column):
    # Gabungkan semua teks dalam kolom text_column menjadi satu string
    text = ' '.join(df[text_column].tolist())

    # Buat word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Tampilkan word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

df = pd.read_csv('/content/cv_data.csv')
create_word_cloud(df, 'Text')

"""# Text Preprocessing"""

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords_english = set(stopwords.words('english'))
stopwords_indonesian = set(stopwords.words('indonesian'))
all_stopwords = stopwords_english.union(stopwords_indonesian) # gabung stopwords dari english dan indonesian

lemmatizer = WordNetLemmatizer() # ubah ke bentuk dasar (english)

factory = StemmerFactory()
stemmer = factory.create_stemmer() # ubah ke bentuk dasar (indonesian)

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove extra spaces
    text = text.strip()

    # Tokenize the text
    tokens = nltk.word_tokenize(text)

   # Remove stopwords for English and lemmatize
    if stopwords_english:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stopwords_english]

    # Remove stopwords for Indonesian
    if stopwords_indonesian:
        tokens = [token for token in tokens if token.lower() not in stopwords_indonesian]
        # Stem the tokens for Indonesian
        tokens = [stemmer.stem(token) for token in tokens]

    # Remove extra spaces or lines
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

df['Preprocessed_Text'] = df['Text'].apply(preprocess_text)

df['Preprocessed_Text'].tail()

create_word_cloud(df, 'Preprocessed_Text')