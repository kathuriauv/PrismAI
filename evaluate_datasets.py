import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

# Import your model and dataset loaders
from src.models.prism_model_v1 import PrismMasterModel
# Adjust these imports based on what you named them in your src/dataset folder
from src.dataset.data_loader import get_meld_test_loader, get_iemocap_test_loader 

def evaluate_on_dataset(model, dataloader, dataset_name, device):
    print(f"\n--- Starting Evaluation on {dataset_name} ---")
    model.eval() # Strictly lock the model weights
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad(): # Disable gradient calculation to save memory/speed
        for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_tensor = batch['audio'].to(device)
            video_tensor = batch['video'].to(device)
            labels = batch['label'].to(device)
            dataset_id = batch['dataset_id'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, audio_tensor, video_tensor, dataset_id)
            logits = outputs[0]
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n[{dataset_name} Results]")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nDetailed Breakdown:")
    # Assuming classes 0: Neutral, 1: Happy, 2: Sad, 3: Angry
    print(classification_report(all_labels, all_preds, target_names=["Neutral", "Happy", "Sad", "Angry"]))
    
    return acc, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Engine running on: {device}")
    
    # 1. Initialize and load the V1 model
    print("Loading best_prism_model.pth...")
    model = PrismMasterModel(num_classes=4, num_datasets=2)
    model.load_state_dict(torch.load("weights/best_prism_model.pth", map_location=device))
    model.to(device)
    
    # 2. Load the specific test sets
    # NOTE: Ensure these functions only load the TEST splits, not the training data
    meld_loader = get_meld_test_loader(batch_size=8)
    iemocap_loader = get_iemocap_test_loader(batch_size=8)
    
    # 3. Run Evaluations
    meld_acc, meld_f1 = evaluate_on_dataset(model, meld_loader, "MELD (TV Dialogue)", device)
    iemocap_acc, iemocap_f1 = evaluate_on_dataset(model, iemocap_loader, "IEMOCAP (Staged Lab)", device)
    
    # 4. Save the final report
    report_data = {
        "Dataset": ["MELD", "IEMOCAP"],
        "Environment": ["TV Show", "Lab Setup"],
        "Accuracy": [meld_acc, iemocap_acc],
        "F1-Score": [meld_f1, iemocap_f1]
    }
    df = pd.DataFrame(report_data)
    df.to_csv("logs/cross_dataset_evaluation.csv", index=False)
    print("\n✅ Evaluation complete. Results saved to logs/cross_dataset_evaluation.csv")

if __name__ == "__main__":
    main()