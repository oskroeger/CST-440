import os
import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import MelSpectrogram, Resample
from torch.nn import Module, LSTM, Linear, functional as F
import pandas as pd
from pydub import AudioSegment
from pydub.utils import which
from tqdm import tqdm
import numpy as np

# -----------------------------------------------------------------------------
# 1. Setup ffmpeg and ffprobe paths for audio file processing
# -----------------------------------------------------------------------------
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# -----------------------------------------------------------------------------
# 2. Define a sample character vocabulary (with blank token at index 0)
#    Make sure len(VOCAB) == output_dim in the model
#    The vocabulary is used for character-to-index and index-to-character mappings.
# -----------------------------------------------------------------------------
VOCAB = [
    "<BLANK>",  # Must be at index 0 for PyTorch's default CTC, required for CTC loss
    "a", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u",
    "v", "w", "x", "y", "z",  # 26 letters
    "'",  # apostrophe
    " "   # space
]
char_to_idx = {ch: idx for idx, ch in enumerate(VOCAB)}  # Map each character to an index
idx_to_char = {idx: ch for idx, ch in enumerate(VOCAB)}  # Map each index back to a character

def text_to_int_sequence(text: str):
    """
    Convert a transcript string into a list of character indices
    based on the char_to_idx map. Ignores characters not in the vocabulary.
    """
    text = text.lower()
    indices = []
    for ch in text:
        if ch in char_to_idx:
            indices.append(char_to_idx[ch])
        #TODO: else: skip or handle unknown chars as needed
    return indices

# -----------------------------------------------------------------------------
# 3. Dataset Class for loading audio-transcription pairs
# -----------------------------------------------------------------------------
class SpeechDataset(Dataset):
    def __init__(self, metadata, clips_path, transforms=None):
        self.metadata = metadata  # Dataframe containing metadata (paths, transcripts)
        self.clips_path = clips_path  # Directory containing audio files
        self.transforms = transforms  # Audio preprocessing transforms

    def __len__(self):
        return len(self.metadata)  # Number of audio-transcription pairs

    def __getitem__(self, idx):
        # Get a row from the metadata Dataframe
        row = self.metadata.iloc[idx]
        audio_path = os.path.join(self.clips_path, row['path'].replace('.mp3', '.wav'))
        transcript = row['sentence']

        # Load the audio waveform and sample rate
        waveform, sample_rate = torchaudio.load(audio_path)
        if self.transforms:
            waveform = self.transforms(waveform)  # Apply preprocessing transforms

        return waveform, transcript  # Return the processed waveform and transcript

# -----------------------------------------------------------------------------
# 4. Convert .mp3 to .wav helper function
# -----------------------------------------------------------------------------
def convert_mp3_to_wav(mp3_path, wav_path):
    """
    Converts an .mp3 file to a .wav file using pydub.
    """
    audio = AudioSegment.from_mp3(mp3_path)  # Load .mp3 file
    audio.export(wav_path, format="wav")  # Save as .wav file

# -----------------------------------------------------------------------------
# 5. Load metadata and convert files
# -----------------------------------------------------------------------------
validated_path = "/mnt/c/users/okroe/Downloads/VoiceRecognitionDataset/cv-corpus-20.0-delta-2024-12-06/en/validated.tsv"
clips_path = "/mnt/c/users/okroe/Downloads/VoiceRecognitionDataset/cv-corpus-20.0-delta-2024-12-06/en/clips/"

# Load metadata from the validated.tsv file
metadata = pd.read_csv(validated_path, sep='\t')

# Filter metadata to include only samples with more than 1 upvote
filtered_metadata = metadata[metadata['up_votes'] > 1]

# Convert audio files to .wav format if not already done
print("Converting .mp3 files to .wav...")
for idx, row in tqdm(filtered_metadata.iterrows(), total=len(filtered_metadata), desc="Converting files"):
    mp3_file = os.path.join(clips_path, row['path'])
    wav_file = os.path.join(clips_path, row['path'].replace('.mp3', '.wav'))
    if not os.path.exists(wav_file):
        convert_mp3_to_wav(mp3_file, wav_file)

# -----------------------------------------------------------------------------
# 6. Data Preprocessing: Resample and generate MelSpectrogram
# -----------------------------------------------------------------------------
transforms = torch.nn.Sequential(
    Resample(orig_freq=48000, new_freq=16000),  # Downsample to 16 kHz
    
    # Generate MelSpectrogram with 80 mel bins
    MelSpectrogram(sample_rate=16000, n_mels=80)  # match input_dim in the model
)

# -----------------------------------------------------------------------------
# 7. Custom Collate Function for Padding/Truncation
# -----------------------------------------------------------------------------
def pad_or_truncate_waveforms(batch, max_length=None):
    """
    Pads or truncates audio waveforms so that all tensors in the batch 
    have the same time dimension.
    Returns: (waveforms, transcripts)
    """
    waveforms, transcripts = zip(*batch)
    max_length = max_length or max(waveform.size(2) for waveform in waveforms)  # Find max length in batch

    padded_waveforms = []
    for waveform in waveforms:
        length = waveform.size(2)
        if length < max_length:
            # Pad with zeros if too short
            padded_waveforms.append(F.pad(waveform, (0, max_length - length)))
        else:
            # Truncate to max_length if too long
            padded_waveforms.append(waveform[:, :, :max_length])

    return torch.stack(padded_waveforms), list(transcripts)  # Return uniform waveforms and transcripts

# -----------------------------------------------------------------------------
# 8. Create Dataset / DataLoader
# -----------------------------------------------------------------------------
dataset = SpeechDataset(filtered_metadata, clips_path, transforms=transforms)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=pad_or_truncate_waveforms)

# -----------------------------------------------------------------------------
# 9. Define the Speech-to-Text Model
# -----------------------------------------------------------------------------
class SpeechToTextModel(Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechToTextModel, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)  # BiLSTM for context
        self.fc = Linear(hidden_dim * 2, output_dim)  # Fully connected layer for classification

    def forward(self, x):
        """
        Expects x of shape (batch_size, 1, n_mels, time).
        Removes channel dim -> (batch_size, n_mels, time),
        then transposes -> (batch_size, time, n_mels).
        """
        x = x.squeeze(1)             # Remove channel dimension: (batch_size, n_mels, time)
        x = x.transpose(1, 2)        # Transpose to: (batch_size, time, n_mels)

        if x.shape[-1] != self.lstm.input_size:
            raise ValueError(f"Expected input size {self.lstm.input_size}, got {x.shape[-1]}")

        x, _ = self.lstm(x)          # Pass through LSTM: (batch_size, time, 2*hidden_dim)
        x = self.fc(x)               # Pass through fully connected layer: (batch_size, time, output_dim)
        return F.log_softmax(x, dim=-1)  # Apply log softmax activation

# -----------------------------------------------------------------------------
# 10. Instantiate the Model
# -----------------------------------------------------------------------------
input_dim = 80   # match n_mels in preprocessing step
hidden_dim = 256  # Hidden state dimension for LSTM
output_dim = len(VOCAB)  # num of symbols in VOCAB
model = SpeechToTextModel(input_dim, hidden_dim, output_dim)

# -----------------------------------------------------------------------------
# 11. Training Loop with CTC loss
# -----------------------------------------------------------------------------
def train_model(model, dataloader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam optimizer for training
    criterion = torch.nn.CTCLoss(blank=0)  # CTC loss (blank token at index 0)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()  # Zero gradients

            # 1. Get data
            waveform_batch, transcripts = batch  # transcripts is list of strings

            # 2. Forward pass (N, T, C)
            outputs = model(waveform_batch)   # shape: (batch_size, time, output_dim)

            # 3. CTC expects shape (time, batch_size, num_classes)
            outputs = outputs.transpose(0, 1) # (time, batch_size, output_dim)

            # 4. Create input_lengths (same for every item in the batch if padded to same length)
            time_dim = outputs.size(0)        # T
            batch_size = outputs.size(1)      # N
            input_lengths = [time_dim] * batch_size

            # 5. Encode transcripts -> 1D concatenated targets + target_lengths
            encoded_targets = []
            target_lengths = []
            for txt in transcripts:
                seq = text_to_int_sequence(txt)
                encoded_targets.extend(seq)
                target_lengths.append(len(seq))

            if len(encoded_targets) == 0:
                # if for some reason everything (batch) is empty, skip
                continue

            encoded_targets = torch.tensor(encoded_targets, dtype=torch.long)

            # 6. Compute the CTC loss
            loss = criterion(outputs, encoded_targets, input_lengths, target_lengths)

            # 7. Backprop & optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

# -----------------------------------------------------------------------------
# 12. WER calculation (placeholder)
# -----------------------------------------------------------------------------
def calculate_wer(predicted, actual):
    """
    #TODO: Implement WER (Word Error Rate) Function here
    """
    pass

# -----------------------------------------------------------------------------
# 13. Run Training & Save the Model
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_model(model, dataloader)  # Train the model
    torch.save(model.state_dict(), "speech_to_text_model.pth")  # Save model weights
