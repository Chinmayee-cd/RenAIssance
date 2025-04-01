import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import json
import numpy as np
from pdf2image import convert_from_path
import pytesseract

class RenaissanceTextDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.samples = []
        
        metadata_path = os.path.join(data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self._load_samples()
        
    def _load_samples(self):
        for source in self.metadata['sources']:
            for page_info in source['pages']:
                image_path = os.path.join(self.data_dir, 'processed', page_info['image_file'])
                if os.path.exists(image_path):
                    image = Image.open(image_path)
                    image_array = np.array(image)
                    
                    self.samples.append({
                        'image': image_array,
                        'text': page_info['ocr_text'],
                        'source': source['filename'],
                        'page': page_info['page_number']
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = torch.from_numpy(sample['image']).float()
        image = image.permute(2, 0, 1)  # Convert from HWC to CHW format
        image = image / 255.0  
        
        text = sample['text'].strip()
        if not text:
            text = "Generate Renaissance-style text"
        text_prompt = f"Generate Renaissance-style text: {text[:100]}..."

        style_reference = image.clone()
        
        return {
            'text_prompts': text_prompt, 
            'style_references': style_reference,
            'real_texts': image,
            'real_text_strings': text,
            'source': sample['source'],
            'page': sample['page']
        } 