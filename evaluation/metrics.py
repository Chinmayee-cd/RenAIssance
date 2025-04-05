import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import sqrtm
import pytesseract
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr

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

class OCRMetrics:
    def __init__(self):
        self.cer = nn.CTCLoss(blank=0, zero_infinity=True)
    
    def calculate_metrics(self, predicted_text, ground_truth):
        """
        Calculate OCR evaluation metrics
        
        Args:
            predicted_text (str): Text predicted by the model
            ground_truth (str): Ground truth text
            
        Returns:
            dict: Dictionary containing various OCR metrics
        """
        # Character Error Rate (CER)
        cer = self._calculate_cer(predicted_text, ground_truth)
        
        # Word Error Rate (WER)
        wer = self._calculate_wer(predicted_text, ground_truth)
        
        # Accuracy
        accuracy = self._calculate_accuracy(predicted_text, ground_truth)
        
        return {
            'cer': cer,
            'wer': wer,
            'accuracy': accuracy
        }
    
    def _calculate_cer(self, pred, target):
        """Calculate Character Error Rate"""
        if not pred or not target:
            return 1.0
        distance = self._levenshtein_distance(pred, target)
        return distance / len(target)
    
    def _calculate_wer(self, pred, target):
        """Calculate Word Error Rate"""
        pred_words = pred.split()
        target_words = target.split()
        if not pred_words or not target_words:
            return 1.0
        distance = self._levenshtein_distance(pred_words, target_words)
        return distance / len(target_words)
    
    def _calculate_accuracy(self, pred, target):
        """Calculate text accuracy"""
        if not pred or not target:
            return 0.0
        correct = sum(1 for a, b in zip(pred, target) if a == b)
        return correct / max(len(pred), len(target))
    
    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two sequences"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

class TextGenerationMetrics:
    def __init__(self):
        self.mse = nn.MSELoss()
    
    def calculate_metrics(self, generated_image, reference_image):
        """
        Calculate text generation evaluation metrics
        
        Args:
            generated_image (torch.Tensor): Generated image
            reference_image (torch.Tensor): Reference image
            
        Returns:
            dict: Dictionary containing various image quality metrics
        """
        # Convert to numpy for skimage metrics
        gen_np = generated_image.cpu().numpy().transpose(1, 2, 0)
        ref_np = reference_image.cpu().numpy().transpose(1, 2, 0)
        
        # Structural Similarity Index (SSIM)
        ssim_score = ssim(gen_np, ref_np, multichannel=True)
        
        # Peak Signal-to-Noise Ratio (PSNR)
        psnr_score = psnr(ref_np, gen_np)
        
        # Mean Squared Error (MSE)
        mse_score = self.mse(generated_image, reference_image).item()
        
        # Style Consistency Score
        style_score = self._calculate_style_consistency(generated_image, reference_image)
        
        # Printing Artifact Score
        artifact_score = self._calculate_artifact_score(generated_image)
        
        return {
            'ssim': ssim_score,
            'psnr': psnr_score,
            'mse': mse_score,
            'style_consistency': style_score,
            'artifact_score': artifact_score
        }
    
    def _calculate_style_consistency(self, gen_img, ref_img):
        """Calculate how well the generated image maintains the style of the reference"""
        # Extract style features using simple statistics
        gen_mean = torch.mean(gen_img, dim=[2, 3])
        ref_mean = torch.mean(ref_img, dim=[2, 3])
        gen_std = torch.std(gen_img, dim=[2, 3])
        ref_std = torch.std(ref_img, dim=[2, 3])
        
        mean_diff = torch.mean(torch.abs(gen_mean - ref_mean))
        std_diff = torch.mean(torch.abs(gen_std - ref_std))
        
        return 1.0 - (mean_diff + std_diff) / 2
    
    def _calculate_artifact_score(self, image):
        """Calculate how well the image simulates printing artifacts"""
        # Check for ink bleed
        kernel = torch.ones(3, 3).to(image.device) / 9
        kernel = kernel.view(1, 1, 3, 3)
        bleed = nn.functional.conv2d(image, kernel, padding=1)
        bleed_score = torch.mean(torch.abs(bleed - image))
        
        # Check for smudging
        blur = nn.functional.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        smudge_score = torch.mean(torch.abs(blur - image))
        
        # Combine scores (higher is better for artifacts)
        return (bleed_score + smudge_score) / 2 