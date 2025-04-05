import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import numpy as np
from pdf2image import convert_from_path
import pytesseract
import logging

logger = logging.getLogger(__name__)

class RenaissanceTextDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.metadata_path = os.path.join(data_dir, 'metadata.json')
        
        # Create style references directory if it doesn't exist
        self.style_dir = os.path.join(data_dir, 'style_references')
        os.makedirs(self.style_dir, exist_ok=True)
        
        logger.info(f"Loading dataset from {data_dir}")
        self._load_metadata()
        
    def _load_metadata(self):
        try:
            with open(self.metadata_path, 'r') as f:
                raw_metadata = json.load(f)
                
            # Handle both list and dict metadata formats
            if isinstance(raw_metadata, dict):
                if 'sources' in raw_metadata:
                    # Convert old format to new format
                    self.metadata = []
                    for source in raw_metadata['sources']:
                        for page in source['pages']:
                            self.metadata.append({
                                'real_text_path': os.path.join('processed', page['image_file']),
                                'style_ref_path': os.path.join('style_references', page['image_file'].replace('.png', '_style.png')),
                                'text_prompt': f"Generate Renaissance-style text: {page['ocr_text'][:100]}...",
                                'real_text_string': page['ocr_text']
                            })
                else:
                    self.metadata = []
            else:
                # Assume new format (list of samples)
                self.metadata = raw_metadata
                
            logger.info(f"Loaded metadata with {len(self.metadata)} samples")
            
            # Validate metadata
            valid_samples = []
            for idx, sample in enumerate(self.metadata):
                try:
                    # Check if real text file exists
                    real_text_path = os.path.join(self.data_dir, sample['real_text_path'])
                    if not os.path.exists(real_text_path):
                        logger.warning(f"Missing real text file: {real_text_path}")
                        continue
                        
                    # Check style reference path
                    style_ref_path = os.path.join(self.data_dir, sample['style_ref_path'])
                    if not os.path.exists(style_ref_path):
                        # Create a default style reference if missing
                        logger.warning(f"Missing style reference file: {style_ref_path}")
                        os.makedirs(os.path.dirname(style_ref_path), exist_ok=True)
                        # Copy the real text image as style reference
                        real_text = Image.open(real_text_path)
                        real_text.save(style_ref_path)
                        logger.info(f"Created default style reference: {style_ref_path}")
                        
                    valid_samples.append(sample)
                except Exception as e:
                    logger.error(f"Error validating sample {idx}: {str(e)}")
                    continue
                    
            self.metadata = valid_samples
            logger.info(f"Found {len(self.metadata)} valid samples")
            
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            self.metadata = []
            
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        try:
            if idx >= len(self.metadata):
                raise IndexError(f"Index {idx} out of range for dataset with {len(self.metadata)} samples")
                
            sample = self.metadata[idx]
            
            # Load real text image
            real_text_path = os.path.join(self.data_dir, sample['real_text_path'])
            real_text = Image.open(real_text_path).convert('RGB')
            real_text = np.array(real_text)
            real_text = torch.from_numpy(real_text).float() / 255.0
            real_text = real_text.permute(2, 0, 1)  # HWC to CHW
            
            # Load style reference
            style_ref_path = os.path.join(self.data_dir, sample['style_ref_path'])
            style_ref = Image.open(style_ref_path).convert('RGB')
            style_ref = np.array(style_ref)
            style_ref = torch.from_numpy(style_ref).float() / 255.0
            style_ref = style_ref.permute(2, 0, 1)  # HWC to CHW
            
            # Get text content
            text_prompt = sample.get('text_prompt', '')
            real_text_string = sample.get('real_text_string', '')
            
            return {
                'real_texts': real_text,
                'style_references': style_ref,
                'text_prompts': text_prompt,
                'real_text_strings': real_text_string
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return a dummy sample to prevent training from crashing
            return {
                'real_texts': torch.zeros((3, 1000, 1000)),
                'style_references': torch.zeros((3, 1000, 1000)),
                'text_prompts': '',
                'real_text_strings': ''
            } 