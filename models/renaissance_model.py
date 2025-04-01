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