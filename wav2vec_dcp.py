import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForPreTraining
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split
import matplotlib.pyplot as plt

class BowelDataset(Dataset):
    def __init__(self, audio_file, label_file, processor, frame_ms=20, sample_rate=16000, chunk_sec=5):
        # Load and resample audio
        audio, sr = torchaudio.load(audio_file)
        audio = torchaudio.functional.resample(audio, sr, sample_rate)
        self.audio = audio.squeeze(0)
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.chunk_len = chunk_sec * sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)

        self.processor = processor

        # Load segments
        self.segments = []
        with open(label_file, 'r') as f:
            for line in f:
                start, end, label = line.strip().split()
                if label in ['v','n']:
                    continue
                if label in ['sb','b']:
                    label='b'
                class_id = {'b':0,'mb':1,'h':2}[label]
                self.segments.append({
                    'start_sample': int(float(start) * sample_rate),
                    'end_sample': int(float(end) * sample_rate),
                    'class': class_id
                })

        # Split audio into chunks
        self.chunks = []
        num_chunks = int(np.ceil(len(self.audio) / self.chunk_len))
        for i in range(num_chunks):
            start = i * self.chunk_len
            end = min((i+1) * self.chunk_len, len(self.audio))
            # Frame labels for this chunk
            num_frames = (end - start) // self.frame_len
            frame_labels = torch.zeros(num_frames, dtype=torch.float32)
            # Segments within this chunk
            chunk_segments = []
            for seg in self.segments:
                if seg['end_sample'] < start or seg['start_sample'] > end:
                    continue  # segment not in this chunk
                seg_start = max(seg['start_sample'], start) - start
                seg_end = min(seg['end_sample'], end) - start
                start_frame = seg_start // self.frame_len
                end_frame = seg_end // self.frame_len
                frame_labels[start_frame:end_frame] = 1
                chunk_segments.append({'start': start_frame, 'end': end_frame, 'class': seg['class']})
            self.chunks.append({
                'audio': self.audio[start:end],
                'frame_labels': frame_labels,
                'segments': chunk_segments
            })

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        inputs = self.processor(
            chunk['audio'].numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        input_values = inputs['input_values'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        frame_labels = chunk['frame_labels']
        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'frame_labels': frame_labels,
            'segments': chunk['segments']
        }
class SoftIoULoss(nn.Module):
    def forward(self, preds, targets, eps=1e-6):
        preds = torch.sigmoid(preds.clamp(-10, 10))


        if preds.sum() == 0 and targets.sum() == 0:
            return torch.tensor(0.0, device=preds.device)
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection
        union = union.clamp(min=eps)
        return 1 - intersection / union



class BowelModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = AutoModelForPreTraining.from_pretrained("/wav2vec2_base").wav2vec2
        hidden_size = self.backbone.config.hidden_size

        self.frame_classifier = nn.Sequential(
    nn.Linear(hidden_size, 1),
     # single output per frame
)  # frame-level detection


        self.classify_head = nn.Sequential(
            nn.Linear(hidden_size, num_classes),

        )  # segment classification

    def forward(self, input_values, attention_mask):
        outputs = self.backbone(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, T, H]
        frame_logits = self.frame_classifier(hidden).squeeze(-1)  # [B, T]
        return frame_logits, hidden


def train_epoch(model,dataloader,optimizer,device,class_weights):
    model.train()
    ce_class = nn.CrossEntropyLoss(weight=class_weights.to(device))
    ce_detect = nn.BCEWithLogitsLoss()
    soft_iou = SoftIoULoss()
    total_loss = 0

    for batch in dataloader:
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        frame_labels = batch['frame_labels'].to(device)
        segments = batch['segments']

        optimizer.zero_grad()
        frame_logits, hidden = model(input_values, attention_mask)
        min_len = min(frame_logits.size(1), frame_labels.size(1))
        frame_logits = frame_logits[:, :min_len]
        frame_labels = frame_labels[:, :min_len]
        frame_logits = torch.clamp(frame_logits, -10, 10)

        # Frame-level loss
        loss_detect = ce_detect(frame_logits, frame_labels) + soft_iou(frame_logits, frame_labels)

        # Segment-level classification
        segment_loss = 0
        valid_segments = 0
        for seg in segments:
            if seg['end'] <= seg['start']:
                continue
            seg_feat = hidden[:, seg['start']:seg['end'], :].mean(dim=1)
            logits = model.classify_head(seg_feat)
            label = torch.tensor([seg['class']], dtype=torch.long).to(device)
            seg_loss = ce_class(logits, label)

            if not torch.isnan(seg_loss):
                segment_loss += seg_loss
                valid_segments += 1
        if valid_segments > 0:
            segment_loss /= valid_segments

        loss = loss_detect + segment_loss
        if torch.isnan(loss):
            print("NaN detected in loss!")
            print("frame_logits:", frame_logits)
            print("frame_labels:", frame_labels)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(dataloader)
def validate_epoch(model, dataloader, device, class_weights):
    model.eval()
    ce_class = nn.CrossEntropyLoss(weight=class_weights.to(device))
    ce_detect = nn.BCEWithLogitsLoss()
    soft_iou = SoftIoULoss()

    total_loss = 0
    total_segments = 0
    correct_segments = 0
    total_frames = 0
    correct_frames = 0

    with torch.no_grad():
        for batch in dataloader:
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            frame_labels = batch['frame_labels'].to(device)
            segments = batch['segments']

            frame_logits, hidden = model(input_values, attention_mask)

            # Align lengths
            min_len = min(frame_logits.size(1), frame_labels.size(1))
            frame_logits = frame_logits[:, :min_len]
            frame_labels = frame_labels[:, :min_len]

            # ---- Frame loss & accuracy ----
            loss_detect = ce_detect(frame_logits, frame_labels) + soft_iou(frame_logits, frame_labels)

            preds = (torch.sigmoid(frame_logits) > 0.5).long()
            correct_frames += (preds == frame_labels.long()).sum().item()
            total_frames += frame_labels.numel()

            # ---- Segment classification ----
            segment_loss = 0
            valid_segments = 0
            for seg in segments:
                if seg['end'] <= seg['start']:
                    continue  # skip broken
                seg_feat = hidden[:, seg['start']:seg['end'], :].mean(dim=1)
                logits = model.classify_head(seg_feat)
                label = torch.tensor([seg['class']], dtype=torch.long).to(device)
                seg_loss = ce_class(logits, label)
                if not torch.isnan(seg_loss):
                    segment_loss += seg_loss
                    valid_segments += 1

                    # Accuracy
                    pred_class = torch.argmax(logits, dim=-1)
                    correct_segments += (pred_class == label).sum().item()
                    total_segments += 1

            if valid_segments > 0:
                segment_loss /= valid_segments
            loss = loss_detect + segment_loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    frame_acc = correct_frames / total_frames if total_frames > 0 else 0
    seg_acc = correct_segments / total_segments if total_segments > 0 else 0

    return avg_loss, frame_acc, seg_acc

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("/wav2vec2_base/")

model = BowelModel(num_classes=3).to(device)

# ----------------------------
# Datasets and DataLoaders
# ----------------------------
dataset = BowelDataset("AS_1.wav", "AS_1.txt", processor)

# 80% train, 20% validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

class_weights = torch.tensor([0.17,0.34,0.49])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# ----------------------------
# Training loop
# ----------------------------
if __name__ == "__main__":
        num_epochs = 20
        train_losses = []
        val_losses = []
        frame_accs = []
        seg_accs = []

        for epoch in range(num_epochs):
                train_loss = train_epoch(model, train_loader, optimizer, device, class_weights)
                val_loss, frame_acc, seg_acc = validate_epoch(model, val_loader, device, class_weights)
                # Store metrics
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                frame_accs.append(frame_acc)
                seg_accs.append(seg_acc)
                print(f"Epoch {epoch + 1}/{num_epochs} "
                      f"- Train Loss: {train_loss:.4f} "
                      f"- Val Loss: {val_loss:.4f} "
                      f"- Frame Acc: {frame_acc:.4f} "
                      f"- Segment Acc: {seg_acc:.4f}")
        # ----------------------------
        # Save model
        # ----------------------------
        torch.save(model.state_dict(),"best_wav2vec2_bowel.pt")
        print("Model saved successfully.")
        # ----------------------------
        # Plot losses
        # ----------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
        plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid()
        plt.show()

        # ----------------------------
        # Plot accuracies
        # ----------------------------
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, num_epochs + 1), frame_accs, label="Frame Accuracy")
        plt.plot(range(1, num_epochs + 1), seg_accs, label="Segment Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Frame & Segment Accuracy")
        plt.legend()
        plt.grid()

        plt.show()
