import os
import torch
import csv
from torch.utils.data import DataLoader, ConcatDataset, random_split

from src.dataset.iemocap_dataset import IEMOCAPDataset
from src.dataset.meld_dataset import MELDDataset
from src.models.prism_model_v1 import PrismMasterModel
from src.training.engine import PrismEngine

def main():
    print("Starting PrismAI Trimodal Training")

    BATCH_SIZE = 4
    EPOCHS = 15
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    iemocap_dir = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\IEMOCAP"
    meld_csv = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\MELD-RAW\MELD.Raw\train\train_sent_emo.csv"
    meld_vid = r"C:\Users\kathu\OneDrive\Desktop\Projects\PrismAI_v1\data\raw\MELD-RAW\MELD.Raw\train\train_splits"

    iemocap_ds = IEMOCAPDataset(data_dir=iemocap_dir)
    meld_ds = MELDDataset(csv_path=meld_csv, video_dir=meld_vid)

    full_dataset = ConcatDataset([iemocap_ds, meld_ds])
    print(f"Total Samples: {len(full_dataset)}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = PrismMasterModel(num_classes=4, num_datasets=2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    engine = PrismEngine(model, train_loader, val_loader, optimizer, DEVICE)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)

    csv_file = open("logs/training_history_v2.csv", mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC'])

    best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss = engine.train_epoch(epoch)
        val_metrics = engine.evaluate(epoch)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"Accuracy:  {val_metrics['acc']:.4f}")
        print(f"Precision: {val_metrics['precision']:.4f}")
        print(f"Recall:    {val_metrics['recall']:.4f}")
        print(f"F1-Score:  {val_metrics['f1']:.4f}")
        print(f"AUC-ROC:   {val_metrics['auc']:.4f}")

        csv_writer.writerow([
            epoch, train_loss, val_metrics['loss'],
            val_metrics['acc'], val_metrics['precision'],
            val_metrics['recall'], val_metrics['f1'], val_metrics['auc']
        ])

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), "weights/best_prism_model_v2_weighted.pth")
            print(f"New best model saved with F1: {best_f1:.4f}")

    csv_file.close()
    print("Training Complete. Logs saved to logs/training_history.csv")

if __name__ == "__main__":
    main()