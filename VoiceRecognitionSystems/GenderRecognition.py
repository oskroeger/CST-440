import os
import pandas as pd
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import librosa.display

# Step 1: Convert TSV to CSV
def tsv_to_csv(input_tsv, output_csv):
    try:
        data = pd.read_csv(input_tsv, sep='\t')

        # Filter and balance dataset
        data = data.dropna(subset=['gender'])
        data = data[data['gender'].isin(['female_feminine', 'male_masculine'])]
        female_data = data[data['gender'] == 'female_feminine'].head(2000)
        male_data = data[data['gender'] == 'male_masculine'].head(2000)
        filtered_data = pd.concat([female_data, male_data])

        filtered_data = filtered_data[['path', 'gender']]
        filtered_data.to_csv(output_csv, index=False)
        print(f"CSV file created successfully: {output_csv}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Step 2: Extract Features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Step 3: Data Augmentation
def augment_audio(audio, sr):
    noise = np.random.normal(0, 0.01, audio.shape)
    return audio + noise

# Step 4: Analyze Audio File
def analyze_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.show()

    plt.figure(figsize=(10, 4))
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.title("Mel Spectrogram")
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# Step 5: Build LSTM Model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Step 6: Train Model
def train_gender_model(data_csv, audio_dir, model_save_path):
    try:
        data = pd.read_csv(data_csv)
        data['gender'] = data['gender'].map({'female_feminine': 0, 'male_masculine': 1})

        features, labels = [], []
        for _, row in data.iterrows():
            file_path = os.path.join(audio_dir, row['path'])
            if os.path.exists(file_path):
                audio, sr = librosa.load(file_path, sr=None)
                audio = augment_audio(audio, sr)  # Apply augmentation
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(row['gender'])

        features = np.array(features)
        labels = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        model = build_lstm_model(X_train.shape[1:])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)
        model.save(model_save_path)
        print(f"Model saved successfully: {model_save_path}")

        predictions = (model.predict(X_test) > 0.5).astype("int32")
        print(classification_report(y_test, predictions))
    except Exception as e:
        print(f"An error occurred during training: {e}")

# Step 7: Test Model
def test_gender_model(model_path, test_files):
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully.")

        for file_path in test_files:
            if os.path.exists(file_path):
                print(f"Processing file: {file_path}")
                analyze_audio(file_path)  # Analyze the test file
                features = extract_features(file_path)
                if features is not None:
                    features = np.expand_dims(features, axis=(0, -1))
                    prediction = model.predict(features)
                    confidence = prediction[0][0]
                    gender = "Male" if confidence > 0.5 else "Female"
                    print(f"Predicted gender for {file_path}: {gender} (Confidence: {confidence:.2f})")
                else:
                    print(f"Feature extraction failed for: {file_path}")
            else:
                print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")

# Main Workflow
if __name__ == "__main__":
    input_tsv = r"C:\Users\tatum\OneDrive\Spring-2025\CST 440\Project 1\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en\other.tsv"
    output_csv = r"C:\Users\tatum\OneDrive\Spring-2025\CST 440\Project 1\my imp\Gender.csv"
    audio_dir = r"C:\Users\tatum\OneDrive\Spring-2025\CST 440\Project 1\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en\clips"
    model_save_path = r"C:\Users\tatum\OneDrive\Spring-2025\CST 440\Project 1\my imp\gender_audio_model.h5"
    test_files = [
        r"C:\Users\tatum\OneDrive\Spring-2025\CST 440\Project 1\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en\preprocessedclips\2.mp3",
        r"C:\Users\tatum\OneDrive\Spring-2025\CST 440\Project 1\cv-corpus-20.0-delta-2024-12-06-en\cv-corpus-20.0-delta-2024-12-06\en\preprocessedclips\3.mp3",
        r"C:\Users\tatum\Downloads\Extracted_1.mp3"
                ]
    print("Step 1: Converting TSV to CSV...")
    tsv_to_csv(input_tsv, output_csv)

    print("\nStep 2: Training the model...")
    train_gender_model(output_csv, audio_dir, model_save_path)

    print("\nStep 3: Testing the model...")
    test_gender_model(model_save_path, test_files)
