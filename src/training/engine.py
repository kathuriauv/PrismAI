import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np

torch.backends.cudnn.benchmark = True

class PrismEngine:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([0.5, 2.0, 2.0, 2.0]).to(device)
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        loop = tqdm(self.train_loader, leave=True, desc=f"Epoch {epoch} [Train]")

        for batch in loop:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            audio_features = batch['audio_features'].to(self.device)
            video_frame = batch['video_frame'].to(self.device)
            labels = batch['label'].to(self.device)
            dataset_ids = batch['dataset_id'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_features=audio_features,
                video_frame=video_frame,
                dataset_ids=dataset_ids
            )

            (logits, text_logits, audio_logits, video_logits,
             text_unc, audio_unc, video_unc, con_loss) = outputs

            cls_loss = self.criterion(logits, labels)
            t_loss = self.criterion(text_logits, labels)
            a_loss = self.criterion(audio_logits, labels)
            v_loss = self.criterion(video_logits, labels)
            unc_loss = text_unc.mean() + audio_unc.mean() + video_unc.mean()

            final_loss = cls_loss + 0.3 * (t_loss + a_loss + v_loss) + 0.1 * con_loss + 0.1 * unc_loss

            final_loss.backward()
            self.optimizer.step()

            total_loss += final_loss.item()
            loop.set_postfix(loss=final_loss.item())

        return total_loss / len(self.train_loader)

    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0

        all_labels = []
        all_preds = []
        all_probs = []

        loop = tqdm(self.val_loader, leave=True, desc=f"Epoch {epoch} [Valid]")

        with torch.no_grad():
            for batch in loop:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio_features = batch['audio_features'].to(self.device)
                video_frame = batch['video_frame'].to(self.device)
                labels = batch['label'].to(self.device)
                dataset_ids = batch['dataset_id'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_features=audio_features,
                    video_frame=video_frame,
                    dataset_ids=dataset_ids
                )

                logits = outputs[0]
                cls_loss = self.criterion(logits, labels)
                total_loss += cls_loss.item()

                probs = F.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                loop.set_postfix(val_loss=cls_loss.item())

        metrics = {}
        metrics['loss'] = total_loss / len(self.val_loader)
        metrics['acc'] = accuracy_score(all_labels, all_preds)

        p, r, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        metrics['precision'] = p
        metrics['recall'] = r
        metrics['f1'] = f1

        try:
            metrics['auc'] = roc_auc_score(
                all_labels, all_probs, multi_class='ovr', average='weighted'
            )
        except ValueError:
            metrics['auc'] = 0.0

        return metrics