#!/usr/bin/env python3

import os
import glob
import pandas as pd
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import Dataset
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from tqdm import tqdm

###############################################################################
# 1) CONFIGURATION
###############################################################################
ROOT_DIR = "/mnt/c/Users/okroe/Downloads/dev-clean/LibriSpeech/dev-clean"  # Path to LibriSpeech dev-clean
OUTPUT_CSV = "librispeech_dev_clean.csv"  # Output CSV for dataset
BATCH_SIZE = 1  # Smaller batch size for CPU
NUM_EPOCHS = 2  # Fewer epochs
LEARNING_RATE = 1e-4
ACCUMULATION_STEPS = 8  # Simulates a larger effective batch size
SAMPLE_RATE = 16000  # Target sample rate for Wav2Vec2

DEVICE = torch.device("cpu")  # Force CPU usage
print(f"Using device: {DEVICE}")

###############################################################################
# 2) PARSE LIBRISPEECH INTO A CSV
###############################################################################
def parse_librispeech(root_dir, output_csv):
    """
    Extracts (audio_path, transcript) pairs from LibriSpeech folders and writes to a CSV.
    """
    data_rows = []
    pattern = os.path.join(root_dir, "**/*.trans.txt")
    trans_files = glob.glob(pattern, recursive=True)

    for trans_file in trans_files:
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                utt_id, transcript = parts
                flac_path = os.path.join(os.path.dirname(trans_file), utt_id + ".flac")

                if os.path.isfile(flac_path):
                    data_rows.append({"path": flac_path, "sentence": transcript.lower()})

    df = pd.DataFrame(data_rows)
    print(f"Found {len(df)} utterances.")
    df.to_csv(output_csv, index=False)
    print(f"Saved dataset to {output_csv}")

if not os.path.exists(OUTPUT_CSV):
    parse_librispeech(ROOT_DIR, OUTPUT_CSV)
else:
    print(f"CSV {OUTPUT_CSV} already exists. Skipping parsing.")

###############################################################################
# 3) LOAD DATASET AND PREPROCESS AUDIO
###############################################################################
def preprocess_audio(batch):
    """
    Loads audio from the path, resamples to the target sample rate,
    and stores the waveform in numpy form.
    """
    waveform, sr = torchaudio.load(batch["path"])
    # Resample if needed
    if sr != SAMPLE_RATE:
        waveform = Resample(sr, SAMPLE_RATE)(waveform)

    batch["audio"] = waveform.squeeze().numpy()
    return batch

# Load dataset from CSV
df = pd.read_csv(OUTPUT_CSV)
dataset = Dataset.from_pandas(df)

# Preprocess (load+resample) audio
dataset = dataset.map(preprocess_audio)

# OPTIONAL: Shuffle and select a smaller subset to speed up training
# Adjust 'subset_size' as needed.
subset_size = 500
if len(dataset) > subset_size:
    dataset = dataset.shuffle(seed=42).select(range(subset_size))

###############################################################################
# 4) LOAD PRE-TRAINED MODEL & FREEZE LOWER LAYERS
###############################################################################
# Use the smaller "facebook/wav2vec2-base" checkpoint
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base").to(DEVICE)

# Freeze the feature extractor layers (lower layers)
for param in model.wav2vec2.feature_extractor.parameters():
    param.requires_grad = False

def prepare_batch(batch):
    """
    Prepares the inputs and labels using the Wav2Vec2 processor.
    """
    inputs = processor(batch["audio"], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with processor.as_target_processor():
        labels = processor(batch["sentence"]).input_ids

    # Store numpy versions so collate_fn can build torch Tensors
    batch["input_values"] = inputs.input_values[0].numpy()
    batch["labels"] = labels
    return batch

# Tokenize transcripts
dataset = dataset.map(prepare_batch, remove_columns=["audio", "path", "sentence"])

###############################################################################
# 5) DATA LOADER
###############################################################################
def collate_fn(batch):
    """
    Builds batch tensors for input_values & labels, and applies padding.
    """
    input_values = [torch.tensor(item["input_values"]) for item in batch]
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels = [torch.tensor(item["labels"]) for item in batch]
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_values": input_values, "labels": labels}

train_loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4
)

###############################################################################
# 6) TRAINING LOOP (WITH GRADIENT ACCUMULATION)
###############################################################################
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

def train_model(model, dataloader, epochs):
    """
    Trains Wav2Vec2 with gradient accumulation. 
    Feature extractor layers are frozen (not trainable).
    """
    model.train()
    optimizer.zero_grad()
    for epoch in range(epochs):
        total_loss = 0.0
        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            inputs = batch["input_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            # Forward pass
            outputs = model(input_values=inputs, labels=labels)
            loss = outputs.loss / ACCUMULATION_STEPS  # Scale loss

            # Backward pass
            loss.backward()
            total_loss += loss.item()

            # Gradient accumulation
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch + 1}: Loss = {total_loss / len(dataloader):.4f}")

# Train
train_model(model, train_loader, NUM_EPOCHS)

###############################################################################
# 7) INFERENCE AND PREDICTIONS
###############################################################################
def predict(batch):
    """
    Runs inference (greedy decoding) on a single example's audio data.
    """
    model.eval()
    with torch.no_grad():
        inputs = processor(batch["audio"], sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        logits = model(inputs.input_values.to(DEVICE)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(pred_ids)

# Print sample predictions
for example in dataset.select(range(min(5, len(dataset)))):
    print(f"Reference: {example['sentence']}")
    print(f"Prediction: {predict(example)}")
    print("-" * 40)
