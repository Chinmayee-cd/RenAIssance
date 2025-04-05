<<<<<<< HEAD
import torch
import torch.nn as nn
import logging
import traceback
from transformers import ViTFeatureExtractor, ViTModel, ViTConfig
from utils.language_processing import LanguageProcessor

logger = logging.getLogger(__name__)

class OCRModel(nn.Module):
    def __init__(self, num_classes=128, hidden_size=768):
        super().__init__()
        logger.info("Initializing OCR model...")
        
        try:
            # Vision Transformer for feature extraction
            self.vit_config = ViTConfig(
                image_size=1000,
                patch_size=16,
                num_channels=3,
                hidden_size=hidden_size,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=0.02,
                layer_norm_eps=1e-12,
            )
            self.vit = ViTModel(self.vit_config)
            
            # Bidirectional LSTM for text recognition
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size // 2,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            
            # Character prediction head
            self.classifier = nn.Linear(hidden_size, num_classes)
            
            # Language processor for post-processing
            self.language_processor = LanguageProcessor()
            
            logger.info("OCR model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing OCR model: {str(e)}")
            raise

    def forward(self, x):
        try:
            # Extract features using ViT
            features = self.vit(x).last_hidden_state
            
            # Process through LSTM
            lstm_out, _ = self.lstm(features)
            
            # Predict characters
            logits = self.classifier(lstm_out)
            
            # Post-process predictions
            predictions = torch.argmax(logits, dim=-1)
            processed_texts = self.language_processor.batch_process(
                [self._decode_predictions(pred) for pred in predictions]
            )
            
            return processed_texts
            
        except Exception as e:
            logger.error(f"Error in OCR forward pass: {str(e)}")
            raise

    def _decode_predictions(self, predictions):
        """Convert model predictions to text."""
        try:
            # Convert predictions to characters
            char_map = {i: chr(i) for i in range(128)}  # ASCII characters
            text = ''.join([char_map[p.item()] for p in predictions])
            return text
        except Exception as e:
            logger.error(f"Error decoding predictions: {str(e)}")
            return ""

class RenaissanceTextGenerator(nn.Module):
    def __init__(self, config={}):
        super().__init__()
        logger.info("Initializing RenaissanceTextGenerator...")
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Decoder with style transfer
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Style transfer layers
        self.style_transfer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        logger.info("RenaissanceTextGenerator initialized successfully")
    
    def forward(self, text_prompt, style_reference):
        try:
            logger.info("Starting text generation forward pass...")
            
            # Encode input
            encoded = self.encoder(style_reference)
            
            # Decode with style
            decoded = self.decoder(encoded)
            
            # Apply style transfer
            stylized = self.style_transfer(decoded)
            
            # Add printing artifacts
            artifacts = self._add_printing_artifacts(stylized)
            
            logger.info("Text generation forward pass completed successfully")
            return artifacts
            
        except Exception as e:
            logger.error(f"Error in text generation forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _add_printing_artifacts(self, image):
        # Simulate ink bleed
        kernel = torch.ones(3, 3).to(image.device) / 9
        kernel = kernel.view(1, 1, 3, 3)
        ink_bleed = nn.functional.conv2d(image, kernel, padding=1)
        
        # Simulate smudging
        blur = nn.functional.avg_pool2d(image, kernel_size=3, stride=1, padding=1)
        
        # Combine effects
        result = 0.7 * image + 0.2 * ink_bleed + 0.1 * blur
        return torch.clamp(result, 0, 1) 
=======
import torch
import torch.nn as nn
import logging
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RenaissanceTextGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        try:
            logger.info("Initializing style transfer network...")
            self.style_transfer = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
            logger.info("Style transfer network initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    def forward(self, text_prompt, style_reference):
        try:
            logger.info("Starting forward pass...")
            if len(style_reference.shape) == 3:
                style_reference = style_reference.unsqueeze(0)
            styled_output = self.style_transfer(style_reference)
            logger.info("Forward pass completed successfully")
            return styled_output
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise 
>>>>>>> 0577fbaf673c3f23bd9da458071ca83cb5589cae
