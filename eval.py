import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import csv
import numpy as np

# Import your model & dataset
from wav2vec_dcp import BowelModel, BowelDataset  # adjust filename

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load processor
processor = AutoProcessor.from_pretrained("C:/Users/lenovo/Downloads/wav2vec2_base/")

# 2. Rebuild model
model = BowelModel(num_classes=3).to(device)
model.load_state_dict(torch.load("best_wav2vec2_bowel4.pt", map_location=device))
model.eval()

# 3. Validation dataset (replace with your file)
val_dataset = BowelDataset(
    "C:/Users/lenovo/Downloads/tech test/Tech Test/23M74M.wav",
    "C:/Users/lenovo/Downloads/tech test/Tech Test/23M74M.txt",
    processor
)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 4. Collect predictions & labels
frame_preds, frame_labels_all = [], []
seg_preds, seg_labels = [], []
def predict_and_export_csv(model, dataset, processor, device, csv_path="predictions.csv", threshold=0.5, merge_gap=0.1):
    model.eval()
    results = []

    with torch.no_grad():
        for chunk_idx in range(len(dataset)):
            batch = dataset[chunk_idx]
            input_values = batch['input_values'].unsqueeze(0).to(device)
            attention_mask = batch['attention_mask'].unsqueeze(0).to(device)

            # Forward pass
            frame_logits, hidden = model(input_values, attention_mask)
            probs = torch.sigmoid(frame_logits).squeeze(0).cpu().numpy()

            # Frame duration (in seconds)
            frame_len_sec = dataset.frame_len / dataset.sample_rate
            chunk_start_time = chunk_idx * dataset.chunk_len / dataset.sample_rate

            # Detect segments where bowel sounds exist
            is_sound = probs > threshold
            start, end = None, None
            raw_segments = []
            for i, flag in enumerate(is_sound):
                if flag and start is None:
                    start = i
                elif not flag and start is not None:
                    end = i
                    raw_segments.append((start, end))
                    start, end = None, None
            if start is not None:
                raw_segments.append((start, len(is_sound)))

            # Merge close segments
            merged_segments = []
            for seg in raw_segments:
                seg_start, seg_end = seg
                seg_start_time = chunk_start_time + seg_start * frame_len_sec
                seg_end_time = chunk_start_time + seg_end * frame_len_sec

                if merged_segments and (seg_start_time - merged_segments[-1][1]) <= merge_gap:
                    # Extend last segment
                    merged_segments[-1] = (merged_segments[-1][0], seg_end_time, merged_segments[-1][2] + [(seg_start, seg_end)])
                else:
                    merged_segments.append((seg_start_time, seg_end_time, [(seg_start, seg_end)]))

            # Classify merged segments
            for seg_start_time, seg_end_time, seg_parts in merged_segments:
                # Collect hidden features from all sub-parts
                feats = []
                for seg_start, seg_end in seg_parts:
                    seg_feat = hidden[:, seg_start:seg_end, :].mean(dim=1)
                    feats.append(seg_feat)
                seg_feat = torch.stack(feats).mean(dim=0)  # average across merged parts

                logits = model.classify_head(seg_feat)
                pred_class = torch.argmax(logits, dim=-1).item()
                class_name = {0: "b", 1: "mb", 2: "h"}[pred_class]

                results.append([seg_start_time, seg_end_time, class_name])

    # Save to CSV
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "end_time", "predicted_class"])
        writer.writerows(results)

    print(f"✅ Predictions saved to {csv_path}")
with torch.no_grad():
    for batch in val_loader:
        input_values = batch["input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        frame_labels = batch["frame_labels"].to(device)
        frame_logits, hidden = model(input_values, attention_mask)
        min_len = min(frame_logits.size(1), frame_labels.size(1))
        frame_logits = frame_logits[:, :min_len]
        frame_labels = frame_labels[:, :min_len]

        # ---- Frame-level ----
        preds = (torch.sigmoid(frame_logits) > 0.5).long().cpu().numpy()
        frame_preds.extend(preds.flatten().tolist())
        frame_labels_all.extend(frame_labels.flatten().tolist())

        # ---- Segment-level ----
        for seg in batch["segments"]:
            seg_feat = hidden[:, seg['start']:seg['end'], :].mean(dim=1)
            logits = model.classify_head(seg_feat)
            pred_class = torch.argmax(logits, dim=1).item()
            seg_preds.append(pred_class)
            seg_labels.append(seg["class"])
predict_and_export_csv(model, val_dataset, processor, device, csv_path="predictions.csv")

# 5. Frame-level metrics
print("\n--- Frame-level Metrics ---")
print("F1:", f1_score(frame_labels_all, frame_preds, zero_division=0))
print("Precision:", precision_score(frame_labels_all, frame_preds, zero_division=0))
print("Recall:", recall_score(frame_labels_all, frame_preds, zero_division=0))
print("Accuracy:", accuracy_score(frame_labels_all, frame_preds))

# 6. Segment-level metrics
print("\n--- Segment-level Metrics ---")
print("Accuracy:", accuracy_score(seg_labels, seg_preds))
print("F1 (macro):", f1_score(seg_labels, seg_preds, average="macro", zero_division=0))
print("Precision (macro):", precision_score(seg_labels, seg_preds, average="macro", zero_division=0))
print("Recall (macro):", recall_score(seg_labels, seg_preds, average="macro", zero_division=0))
