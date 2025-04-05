import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import logging
from typing import List, Dict, Union
import re

logger = logging.getLogger(__name__)

class LanguageProcessor:
    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """Initialize the language processor with a multilingual BERT model."""
        logger.info(f"Initializing language processor with model: {model_name}")
        try:
            # Try to load the model with offline fallback
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
                self.model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=False)
            except Exception as e:
                logger.warning(f"Failed to load model from HuggingFace: {str(e)}")
                logger.info("Falling back to local processing only")
                self.tokenizer = None
                self.model = None
            
            if self.model:
                self.model.eval()
                logger.info("BERT model loaded successfully")
            else:
                logger.info("Using rule-based processing only")
            
        except Exception as e:
            logger.error(f"Error initializing language processor: {str(e)}")
            self.tokenizer = None
            self.model = None

    def correct_ocr_errors(self, text: str, confidence_threshold: float = 0.8) -> str:
        """Correct common OCR errors using BERT if available, otherwise use rule-based correction."""
        try:
            if self.model and self.tokenizer and text.strip():
                # Use BERT for correction
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = outputs.logits
                
                predicted_tokens = torch.argmax(predictions, dim=-1)
                corrected_text = self.tokenizer.decode(predicted_tokens[0], skip_special_tokens=True)
                
                if outputs.logits.max() < confidence_threshold:
                    logger.warning(f"Low confidence in correction for text: {text}")
                    return self._rule_based_correction(text)
                
                return corrected_text
            else:
                # Fallback to rule-based correction
                return self._rule_based_correction(text)
            
        except Exception as e:
            logger.error(f"Error in OCR correction: {str(e)}")
            return self._rule_based_correction(text)

    def _rule_based_correction(self, text: str) -> str:
        """Apply rule-based corrections for common OCR errors."""
        try:
            if not text.strip():
                return text
                
            # Common OCR error patterns
            correction_patterns = {
                r'[0oO]': 'o',  # Common number/letter confusion
                r'[1lI]': 'l',  # Common number/letter confusion
                r'[5sS]': 's',  # Common number/letter confusion
                r'[8B]': 'B',   # Common number/letter confusion
                r'[2Z]': 'Z',   # Common number/letter confusion
                r'\s+': ' ',    # Normalize whitespace
                r'[.,;:]+': '.', # Normalize punctuation
            }
            
            corrected_text = text
            for pattern, replacement in correction_patterns.items():
                corrected_text = re.sub(pattern, replacement, corrected_text)
            
            return corrected_text.strip()
            
        except Exception as e:
            logger.error(f"Error in rule-based correction: {str(e)}")
            return text

    def normalize_historical_spanish(self, text: str) -> str:
        """Normalize historical Spanish text to modern Spanish."""
        try:
            if not text.strip():
                return text
                
            # Common historical Spanish patterns
            historical_patterns = {
                r'ç': 'c',  # Replace ç with c
                r'ſ': 's',  # Replace long s with regular s
                r'[ſs]+': 's',  # Normalize multiple s
                r'[uú]': 'u',  # Normalize u variations
                r'[ií]': 'i',  # Normalize i variations
                r'[j]': 'i',  # Replace j with i where appropriate
                r'[h]': '',   # Remove silent h
                r'[ñ]': 'n',  # Replace ñ with n
                r'[áéíóú]': lambda m: m.group(0)[0],  # Remove accents
            }
            
            normalized_text = text
            for pattern, replacement in historical_patterns.items():
                if callable(replacement):
                    normalized_text = re.sub(pattern, replacement, normalized_text)
                else:
                    normalized_text = re.sub(pattern, replacement, normalized_text)
            
            return normalized_text
            
        except Exception as e:
            logger.error(f"Error in historical text normalization: {str(e)}")
            return text

    def post_process_text(self, text: str, 
                         correct_ocr: bool = True,
                         normalize_historical: bool = True,
                         confidence_threshold: float = 0.8) -> str:
        """Apply all post-processing steps to the text."""
        try:
            if not text or not text.strip():
                return text
                
            processed_text = text
            
            if normalize_historical:
                processed_text = self.normalize_historical_spanish(processed_text)
            
            if correct_ocr:
                processed_text = self.correct_ocr_errors(processed_text, confidence_threshold)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error in text post-processing: {str(e)}")
            return text

    def batch_process(self, texts: List[str], 
                     correct_ocr: bool = True,
                     normalize_historical: bool = True) -> List[str]:
        """Process a batch of texts."""
        try:
            if not texts:
                return texts
                
            processed_texts = []
            for text in texts:
                processed_text = self.post_process_text(
                    text,
                    correct_ocr=correct_ocr,
                    normalize_historical=normalize_historical
                )
                processed_texts.append(processed_text)
            return processed_texts
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return texts 