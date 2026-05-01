# src/training/engine.py
import torch
import torch.nn as nn
from tqdm import tqdm

class PrismEngine:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        
        # Standard classification loss
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # tqdm gives us a beautiful progress bar in the terminal
        loop = tqdm(self.train_loader, leave=True, desc=f"Epoch {epoch} [Train]")
        
        for batch in loop:
            # 1. Move all multimodal data to the GPU/CPU
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_frame = batch['video_frame'].to(self.device)
            labels = batch['label'].to(self.device)
            dataset_ids = batch['dataset_id'].to(self.device)
            
            # 2. Clear old gradients
            self.optimizer.zero_grad()
            
            # 3. Forward Pass!
            outputs = self.model(
                text_features=input_ids, # Note: We pass raw text to the model, it handles embedding internally
                audio_features=audio_features, 
                video_features=video_frame, 
                dataset_ids=dataset_ids
            )
            
            # Unpack the 8 items returned by your Fusion Layer
            (logits, text_logits, audio_logits, video_logits, 
             text_unc, audio_unc, video_unc, con_loss) = outputs
             
            # 4. Calculate the Advanced Loss
            # A. Main fusion classification loss
            cls_loss = self.criterion(logits, labels)
            
            # B. Individual modality losses (forces each encoder to learn properly)
            t_loss = self.criterion(text_logits, labels)
            a_loss = self.criterion(audio_logits, labels)
            v_loss = self.criterion(video_logits, labels)
            
            # C. Uncertainty Regularization (penalizes the model for being too uncertain)
            unc_loss = text_unc.mean() + audio_unc.mean() + video_unc.mean()
            
            # Combine everything based on your mentor's theoretical weighting
            final_loss = cls_loss + 0.3 * (t_loss + a_loss + v_loss) + 0.1 * con_loss + 0.1 * unc_loss
            
            # 5. Backward Pass (Learn!)
            final_loss.backward()
            self.optimizer.step()
            
            # Tracking metrics
            total_loss += final_loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            loop.set_postfix(loss=final_loss.item(), acc=correct/total)
            
        return total_loss / len(self.train_loader), correct / total

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(self.val_loader, leave=True, desc=f"Epoch {epoch} [Valid]")
        
        with torch.no_grad(): # Don't calculate gradients during evaluation
            for batch in loop:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                video_frame = batch['video_frame'].to(self.device)
                labels = batch['label'].to(self.device)
                dataset_ids = batch['dataset_id'].to(self.device)
                
                outputs = self.model(
                    text_features=input_ids,
                    audio_features=audio_features, 
                    video_features=video_frame, 
                    dataset_ids=dataset_ids
                )
                
                # Unpack and calculate main validation loss
                logits = outputs[0]
                cls_loss = self.criterion(logits, labels)
                
                total_loss += cls_loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                loop.set_postfix(val_loss=cls_loss.item(), val_acc=correct/total)
                
        return total_loss / len(self.val_loader), correct / total