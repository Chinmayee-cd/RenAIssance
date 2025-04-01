import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

from models.renaissance_model import RenaissanceTextGenerator
from evaluation.metrics import RenaissanceTextEvaluator
from utils.preprocessing import preprocess_dataset
from utils.dataset import RenaissanceTextDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Train or generate Renaissance-style text')
    parser.add_argument('--mode', choices=['train', 'generate', 'evaluate'], required=True,
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

def train(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, batch in enumerate(progress_bar):
            try:
                text_prompts = batch['text_prompts']  # List of strings
                style_references = batch['style_references'].to(device)  # [B, C, H, W]
                real_texts = batch['real_texts'].to(device)  # [B, C, H, W]

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
                        print(f"Error processing sample {i}: {str(e)}")
                        continue
                
                if not generated_texts:
                    print(f"No valid samples in batch {batch_idx}, skipping...")
                    continue
            
                generated_texts = torch.cat(generated_texts, dim=0)

                real_texts = real_texts[:len(generated_texts)].to(device)
                style_references = style_references[:len(generated_texts)].to(device)

                content_loss = criterion(generated_texts, real_texts)
                
                style_loss = torch.mean(torch.abs(generated_texts - style_references))
                
                total_loss = content_loss + 0.5 * style_loss
                total_loss.backward()
                optimizer.step()

                num_batches += 1
                avg_loss = total_loss.item() / num_batches
                progress_bar.set_postfix({
                    'loss': avg_loss,
                    'content_loss': content_loss.item(),
                    'style_loss': style_loss.item()
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        if num_batches > 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'outputs/checkpoint_epoch_{epoch+1}.pth')
        else:
            print(f'Epoch {epoch+1}/{num_epochs} had no valid batches, skipping checkpoint save')

def generate(model, text_prompts, style_references, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (prompt, reference) in enumerate(zip(text_prompts, style_references)):
            generated_text = model(prompt, reference)

            output_path = os.path.join(output_dir, f'generated_{i}.png')
            generated_text.save(output_path)

def evaluate(model, test_loader, evaluator, device):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            text_prompts = batch['text_prompts'].to(device)
            style_references = batch['style_references'].to(device)
            real_texts = batch['real_texts']

            generated_texts = model(text_prompts, style_references)

            for gen, real, ref, real_text in zip(generated_texts, real_texts, style_references, batch['real_text_strings']):
                metrics = evaluator.evaluate(gen, real, ref, real_text)
                results.append(metrics)

    avg_metrics = {}
    for metric in results[0].keys():
        avg_metrics[metric] = sum(r[metric] for r in results) / len(results)

    return avg_metrics

def main():
    args = parse_args()
    
    print("Starting main function...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        print("Initializing model...")
        model = RenaissanceTextGenerator(config={})
        print("Moving model to device:", args.device)
        model = model.to(args.device)
        
        print("Setting up optimizer and criterion...")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.MSELoss()
        
        print("Setting up evaluator...")
        evaluator = RenaissanceTextEvaluator()
        
        if args.mode == 'train':
            print("Preprocessing dataset...")
            preprocess_dataset(args.data_dir)
            
            print("Loading dataset...")
            train_dataset = RenaissanceTextDataset(args.data_dir)
            print(f"Dataset size: {len(train_dataset)} samples")
            
            print("Creating data loader...")
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True,
                num_workers=0,  # Disable multiprocessing to avoid memory issues
                pin_memory=True if torch.cuda.is_available() else False
            )
            print(f"Number of batches: {len(train_loader)}")
            
            print("Starting training...")
            train(model, train_loader, optimizer, criterion, args.device, args.num_epochs)
            
            print("Saving final model...")
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))
            
        elif args.mode == 'generate':
            print("Loading model checkpoint...")
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.pth')))
            
            text_prompts = ["Sample Spanish Renaissance text prompt 1", "Sample Spanish Renaissance text prompt 2"]
            style_references = [torch.randn(3, 256, 256) for _ in range(len(text_prompts))]
            
            print("Generating text...")
            generate(model, text_prompts, style_references, os.path.join(args.output_dir, 'generated'))
            
        elif args.mode == 'evaluate':
            print("Loading model checkpoint...")
            model.load_state_dict(torch.load(os.path.join(args.output_dir, 'model.pth')))
            
            print("Loading test dataset...")
            test_dataset = RenaissanceTextDataset(args.data_dir, split='test')
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size,
                num_workers=0,  # Disable multiprocessing to avoid memory issues
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            print("Starting evaluation...")
            metrics = evaluate(model, test_loader, evaluator, args.device)
            
            print("Saving metrics...")
            with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print("Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()