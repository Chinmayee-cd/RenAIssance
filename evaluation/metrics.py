import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
import pytesseract
from PIL import Image
import cv2

class RenaissanceTextEvaluator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def calculate_ssim(self, generated, real):
        """Calculate Structural Similarity Index between generated and real text."""
        # Convert to grayscale if needed
        if len(generated.shape) == 3:
            generated = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
        if len(real.shape) == 3:
            real = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
            
        return ssim(generated, real, data_range=255)
    
    def calculate_fid(self, generated, real):
        """Calculate Fréchet Inception Distance between generated and real text."""
        # Convert to numpy arrays if needed
        if torch.is_tensor(generated):
            generated = generated.cpu().numpy()
        if torch.is_tensor(real):
            real = real.cpu().numpy()
            
        # Calculate mean and covariance
        mu1 = np.mean(generated, axis=1)
        mu2 = np.mean(real, axis=1)
        sigma1 = np.cov(generated, rowvar=False)
        sigma2 = np.cov(real, rowvar=False)
        
        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    
    def calculate_ocr_accuracy(self, generated, real_text):
        """Calculate OCR accuracy of generated text compared to real text."""
        # Convert to PIL Image if needed
        if not isinstance(generated, Image.Image):
            generated = Image.fromarray(generated)
            
        # Perform OCR
        generated_text = pytesseract.image_to_string(generated, lang='spa')
        
        # Calculate accuracy (simple character-level accuracy)
        correct = sum(1 for a, b in zip(generated_text, real_text) if a == b)
        total = max(len(generated_text), len(real_text))
        return correct / total
    
    def calculate_style_consistency(self, generated, reference):
        """Calculate how well the generated text matches Renaissance style."""
        # Convert to grayscale if needed
        if len(generated.shape) == 3:
            generated = cv2.cvtColor(generated, cv2.COLOR_RGB2GRAY)
        if len(reference.shape) == 3:
            reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
            
        # Calculate histogram similarity
        hist1 = cv2.calcHist([generated], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([reference], [0], None, [256], [0, 256])
        
        # Normalize histograms
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        
        # Calculate histogram intersection
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
        return similarity
    
    def evaluate(self, generated, real, reference, real_text):
        """Evaluate all metrics for the generated text."""
        results = {
            'ssim': self.calculate_ssim(generated, real),
            'fid': self.calculate_fid(generated, real),
            'ocr_accuracy': self.calculate_ocr_accuracy(generated, real_text),
            'style_consistency': self.calculate_style_consistency(generated, reference)
        }
        
        # Calculate overall score (weighted average)
        weights = {
            'ssim': 0.3,
            'fid': 0.3,
            'ocr_accuracy': 0.2,
            'style_consistency': 0.2
        }
        
        overall_score = sum(results[metric] * weights[metric] for metric in weights)
        results['overall_score'] = overall_score
        
        return results 