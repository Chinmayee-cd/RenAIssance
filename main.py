import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
import logging
import traceback

from models.renaissance_model import OCRModel, RenaissanceTextGenerator
from evaluation.metrics import OCRMetrics, TextGenerationMetrics
from utils.preprocessing import preprocess_dataset
from utils.dataset import RenaissanceTextDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train OCR and text generation models')
    parser.add_argument('--mode', choices=['train_ocr', 'train_gen', 'evaluate'], required=True,
                        help='Mode to run the model in')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run the model on')
    return parser.parse_args()

def train_ocr(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    metrics = OCRMetrics()
    logger.info(f"Training OCR model on device: {device}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        epoch_metrics = {
            'cer': 0,
            'wer': 0,
            'accuracy': 0
        }

        for batch_idx, batch in enumerate(progress_bar):
            try:
                images = batch['real_texts'].to(device)
                text_targets = batch['real_text_strings']
                
                optimizer.zero_grad()
                predictions = model(images)
                
                # Calculate loss
                loss = criterion(predictions, text_targets)
                loss.backward()
                optimizer.step()
                
                # Calculate metrics
                batch_metrics = metrics.calculate_metrics(predictions, text_targets)
                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': total_loss / num_batches,
                    'cer': epoch_metrics['cer'] / num_batches,
                    'wer': epoch_metrics['wer'] / num_batches
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        if num_batches > 0:
            avg_metrics = {k: v/num_batches for k, v in epoch_metrics.items()}
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            logger.info(f'Average Loss: {total_loss/num_batches:.4f}')
            logger.info(f'Metrics: {avg_metrics}')
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': avg_metrics,
            }, f'outputs/ocr_checkpoint_epoch_{epoch+1}.pth')
        else:
            logger.warning(f'Epoch {epoch+1}/{num_epochs} had no valid batches')

def train_generator(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    metrics = TextGenerationMetrics()
    logger.info(f"Training text generator on device: {device}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        epoch_metrics = {
            'ssim': 0,
            'psnr': 0,
            'mse': 0,
            'style_consistency': 0,
            'artifact_score': 0
        }

        for batch_idx, batch in enumerate(progress_bar):
            try:
                text_prompts = batch['text_prompts']
                style_references = batch['style_references'].to(device)
                real_texts = batch['real_texts'].to(device)
                
                optimizer.zero_grad()
                batch_size = len(text_prompts)
                generated_texts = []
                
                for i in range(batch_size):
                    try:
                        generated_text = model(text_prompts[i], style_references[i].unsqueeze(0))
                        if generated_text.device != device:
                            generated_text = generated_text.to(device)
                        generated_texts.append(generated_text)
                    except Exception as e:
                        logger.error(f"Error processing sample {i}: {str(e)}")
                        continue
                
                if not generated_texts:
                    logger.warning(f"No valid samples in batch {batch_idx}, skipping...")
                    continue
                
                generated_texts = torch.cat(generated_texts, dim=0)
                real_texts = real_texts[:len(generated_texts)]
                
                # Calculate losses
                content_loss = criterion(generated_texts, real_texts)
                style_loss = torch.mean(torch.abs(generated_texts - style_references[:len(generated_texts)]))
                total_loss = content_loss + 0.5 * style_loss
                
                total_loss.backward()
                optimizer.step()
                
                # Calculate metrics
                batch_metrics = metrics.calculate_metrics(generated_texts, real_texts)
                for k, v in batch_metrics.items():
                    epoch_metrics[k] += v
                
                num_batches += 1
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'ssim': epoch_metrics['ssim'] / num_batches,
                    'psnr': epoch_metrics['psnr'] / num_batches
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        if num_batches > 0:
            avg_metrics = {k: v/num_batches for k, v in epoch_metrics.items()}
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            logger.info(f'Average Loss: {total_loss/num_batches:.4f}')
            logger.info(f'Metrics: {avg_metrics}')
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': avg_metrics,
            }, f'outputs/gen_checkpoint_epoch_{epoch+1}.pth')
        else:
            logger.warning(f'Epoch {epoch+1}/{num_epochs} had no valid batches')

def evaluate_models(ocr_model, gen_model, test_loader, device):
    ocr_model.eval()
    gen_model.eval()
    
    ocr_metrics = OCRMetrics()
    gen_metrics = TextGenerationMetrics()
    
    results = {
        'ocr': {'cer': 0, 'wer': 0, 'accuracy': 0},
        'generation': {'ssim': 0, 'psnr': 0, 'mse': 0, 'style_consistency': 0, 'artifact_score': 0}
    }
    
    num_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            try:
                # OCR evaluation
                images = batch['real_texts'].to(device)
                text_targets = batch['real_text_strings']
                ocr_predictions = ocr_model(images)
                ocr_batch_metrics = ocr_metrics.calculate_metrics(ocr_predictions, text_targets)
                
                # Text generation evaluation
                text_prompts = batch['text_prompts']
                style_references = batch['style_references'].to(device)
                real_texts = batch['real_texts'].to(device)
                
                generated_texts = []
                for i in range(len(text_prompts)):
                    generated_text = gen_model(text_prompts[i], style_references[i].unsqueeze(0))
                    generated_texts.append(generated_text)
                
                generated_texts = torch.cat(generated_texts, dim=0)
                gen_batch_metrics = gen_metrics.calculate_metrics(generated_texts, real_texts[:len(generated_texts)])
                
                # Accumulate metrics
                for k, v in ocr_batch_metrics.items():
                    results['ocr'][k] += v
                for k, v in gen_batch_metrics.items():
                    results['generation'][k] += v
                
                num_samples += len(text_prompts)
                
            except Exception as e:
                logger.error(f"Error in evaluation batch: {str(e)}")
                continue
    
    # Calculate averages
    if num_samples > 0:
        for task in results:
            for metric in results[task]:
                results[task][metric] /= num_samples
    
    return results

def main():
    args = parse_args()
    logger.info("Starting main function...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize models
        logger.info("Initializing models...")
        ocr_model = OCRModel()
        gen_model = RenaissanceTextGenerator()
        
        # Move models to device
        logger.info(f"Moving models to device: {args.device}")
        ocr_model = ocr_model.to(args.device)
        gen_model = gen_model.to(args.device)
        
        # Set up optimizers and criteria
        logger.info("Setting up optimizers and criteria...")
        ocr_optimizer = optim.Adam(ocr_model.parameters(), lr=args.learning_rate)
        gen_optimizer = optim.Adam(gen_model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        
        # Load dataset
        logger.info("Loading dataset...")
        train_dataset = RenaissanceTextDataset(args.data_dir)
        logger.info(f"Dataset size: {len(train_dataset)} samples")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        if args.mode == 'train_ocr':
            logger.info("Starting OCR training...")
            train_ocr(ocr_model, train_loader, ocr_optimizer, criterion, args.device, args.num_epochs)
            torch.save(ocr_model.state_dict(), os.path.join(args.output_dir, 'ocr_model.pth'))
            
        elif args.mode == 'train_gen':
            logger.info("Starting text generation training...")
            train_generator(gen_model, train_loader, gen_optimizer, criterion, args.device, args.num_epochs)
            torch.save(gen_model.state_dict(), os.path.join(args.output_dir, 'gen_model.pth'))
            
        elif args.mode == 'evaluate':
            logger.info("Loading model checkpoints...")
            ocr_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'ocr_model.pth')))
            gen_model.load_state_dict(torch.load(os.path.join(args.output_dir, 'gen_model.pth')))
            
            logger.info("Starting evaluation...")
            results = evaluate_models(ocr_model, gen_model, train_loader, args.device)
            
            logger.info("Evaluation Results:")
            logger.info(f"OCR Metrics: {results['ocr']}")
            logger.info(f"Generation Metrics: {results['generation']}")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == '__main__':
    main()