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