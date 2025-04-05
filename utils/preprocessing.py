<<<<<<< HEAD
import os
import json
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import logging

logger = logging.getLogger(__name__)

def preprocess_dataset(data_dir):
    """Preprocess the dataset for both OCR and text generation tasks."""
    logger.info(f"Preprocessing dataset in {data_dir}")
    
    # Create necessary directories
    processed_dir = os.path.join(data_dir, 'processed')
    style_dir = os.path.join(data_dir, 'style_references')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)
    
    metadata = []
    
    # Process each PDF file
    for filename in os.listdir(data_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(data_dir, filename)
            logger.info(f"Processing PDF: {filename}")
            
            try:
                # Convert PDF to images
                images = convert_from_path(pdf_path)
                
                for i, image in enumerate(images):
                    # Convert to numpy array
                    image_array = np.array(image)
                    
                    # Preprocess for OCR
                    processed_image = preprocess_for_ocr(image_array)
                    
                    # Create style reference
                    style_reference = create_style_reference(processed_image)
                    
                    # Save processed image
                    processed_path = os.path.join(processed_dir, f"{filename[:-4]}_page_{i+1}.png")
                    Image.fromarray(processed_image).save(processed_path)
                    
                    # Save style reference
                    style_path = os.path.join(style_dir, f"{filename[:-4]}_page_{i+1}_style.png")
                    Image.fromarray(style_reference).save(style_path)
                    
                    # Extract text using OCR
                    text = pytesseract.image_to_string(Image.fromarray(processed_image))
                    
                    # Add to metadata
                    metadata.append({
                        'real_text_path': os.path.relpath(processed_path, data_dir),
                        'style_ref_path': os.path.relpath(style_path, data_dir),
                        'text_prompt': f"Generate Renaissance-style text: {text[:100]}...",
                        'real_text_string': text.strip(),
                        'source': filename,
                        'page': i + 1
                    })
                    
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
    
    # Save metadata
    metadata_path = os.path.join(data_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Preprocessing complete. Processed {len(metadata)} pages.")

def preprocess_for_ocr(image_array):
    """Preprocess image for OCR task."""
    try:
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
            
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Resize to standard size
        resized = cv2.resize(enhanced, (1000, 1000))
        
        return resized
        
    except Exception as e:
        logger.error(f"Error in OCR preprocessing: {str(e)}")
        raise

def create_style_reference(image_array, style_type='ink_bleed'):
    """Create style reference image for text generation."""
    try:
        # Convert to float for processing
        image = image_array.astype(float) / 255.0
        
        if style_type == 'ink_bleed':
            # Simulate ink bleed effect
            kernel = np.ones((3,3), np.float32) / 9
            blurred = cv2.filter2D(image, -1, kernel)
            style = cv2.addWeighted(image, 0.7, blurred, 0.3, 0)
            
        elif style_type == 'smudged':
            # Simulate smudging effect
            kernel = np.ones((5,5), np.float32) / 25
            style = cv2.filter2D(image, -1, kernel)
            
        elif style_type == 'faded':
            # Simulate faded text effect
            style = cv2.addWeighted(image, 0.8, np.ones_like(image) * 0.2, 0.2, 0)
            
        else:
            style = image
            
        # Convert back to uint8
        style = (style * 255).astype(np.uint8)
        
        # Ensure 3 channels for RGB
        if len(style.shape) == 2:
            style = cv2.cvtColor(style, cv2.COLOR_GRAY2RGB)
            
        return style
        
    except Exception as e:
        logger.error(f"Error creating style reference: {str(e)}")
        raise 
=======
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
>>>>>>> 0577fbaf673c3f23bd9da458071ca83cb5589cae
