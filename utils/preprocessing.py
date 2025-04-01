import os
import json
import numpy as np
from PIL import Image
import cv2
from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def preprocess_dataset(data_dir):
    os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'pdfs'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'transcriptions'), exist_ok=True)

    pdf_dir = os.path.join(data_dir, 'pdfs')
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    metadata = {
        'sources': [],
        'total_pages': 0,
        'processed_pages': 0
    }
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        source_info = {
            'filename': pdf_file,
            'pages': [],
            'transcription_file': pdf_file.replace('.pdf', '_transcription.txt')
        }
        pages = convert_from_path(pdf_path, first_page=1, last_page=3,dpi=150)
        metadata['total_pages'] += len(pages)
        for i, page in enumerate(pages):
            page_array = np.array(page)
            processed_page = preprocess_page(page_array)
            page_filename = f"{pdf_file.replace('.pdf', '')}_page_{i+1}.png"
            page_path = os.path.join(data_dir, 'processed', page_filename)
            processed_image = Image.fromarray(processed_page)
            processed_image.save(page_path)
            ocr_text = pytesseract.image_to_string(processed_image, lang='spa')
            
            source_info['pages'].append({
                'page_number': i + 1,
                'image_file': page_filename,
                'ocr_text': ocr_text
            })
        
        metadata['sources'].append(source_info)
        metadata['processed_pages'] += len(pages)
        transcription_file = os.path.join(data_dir, 'transcriptions', source_info['transcription_file'])
        with open(transcription_file, 'w', encoding='utf-8') as f:
            for page in source_info['pages']:
                f.write(page['ocr_text'] + '\n\n')
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata

def preprocess_page(page_array):
    if len(page_array.shape) == 3:
        gray = cv2.cvtColor(page_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = page_array
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    denoised = cv2.fastNlMeansDenoising(binary)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    resized = cv2.resize(enhanced, (1000, 1000), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    return rgb

def create_style_reference(image_array, style_type='ink_bleed'):
    if style_type == 'ink_bleed':
        #Ink based effect
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(image_array, kernel, iterations=1)
        return cv2.addWeighted(image_array, 0.7, dilated, 0.3, 0)
    
    elif style_type == 'smudging':
        # Smudging effect
        blur = cv2.GaussianBlur(image_array, (5,5), 0)
        return cv2.addWeighted(image_array, 0.8, blur, 0.2, 0)
    
    elif style_type == 'faded':
        # Faded text effect
        return cv2.addWeighted(image_array, 0.6, np.ones_like(image_array) * 255, 0.4, 0)
    else:
        return image_array 