import pandas as pd
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, Sequential


# Function to read a JSON file line by line
def read_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data


# Load the Kaggle dataset
data = read_json_lines('data/Entity Recognition in Resumes.json')


# Function to preprocess the data and extract entity spans
def preprocess_data_with_spans(data):
    sentences = []
    labels = []
    spans = []
    for item in data:
        sentence = item['content']
        annotations = item.get('annotation', [])
        words = sentence.split()
        entities = ['O'] * len(words)
        entity_spans = []

        for ann in annotations:
            points = ann['points'][0]
            labels_list = ann['label']
            if not labels_list:
                continue  # Skip if labels_list is empty

            start, end = points['start'], points['end']
            entity_text = points['text']
            word_start_idx = len(sentence[:start].split())
            word_end_idx = len(sentence[:end].split())

            # Ensure indices are within bounds
            word_start_idx = min(word_start_idx, len(words) - 1)
            word_end_idx = min(word_end_idx, len(words))

            for idx in range(word_start_idx, word_end_idx):
                entities[idx] = labels_list[0]
                if idx == word_start_idx:
                    entity_spans.append((entity_text, labels_list[0], start, end))

        sentences.append(words)
        labels.append(entities)
        spans.append(entity_spans)

    return sentences, labels, spans


# Preprocess the data
sentences, labels, spans = preprocess_data_with_spans(data)

# Flatten the list of labels for encoding
flat_labels = [label for sublist in labels for label in sublist]

# Initialize the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(flat_labels)

# Encode the labels
encoded_labels = [label_encoder.transform(label) for label in labels]

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

# Convert sentences to sequences
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences
max_len = max([len(seq) for seq in sequences])
sequences_padded = pad_sequences(sequences, maxlen=max_len, padding='post')
encoded_labels_padded = pad_sequences(encoded_labels, maxlen=max_len, padding='post')

# Convert labels to categorical
encoded_labels_padded = tf.keras.utils.to_categorical(encoded_labels_padded, num_classes=len(label_encoder.classes_))

# Split the data into training and test sets
train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences_padded, encoded_labels_padded,
                                                                              test_size=0.2)


# Function to display entity spans
def display_entity_spans(sentences, spans):
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i + 1}: {' '.join(sentence)}")
        for entity_text, entity_label, start, end in spans[i]:
            print(f"Entity: {entity_text}, Label: {entity_label}, Start: {start}, End: {end}")
        print()


# Display the entity spans
display_entity_spans(sentences, spans)

# bagian sini ke bawah mungkin masih harus di perbaiki or modif

# Define the NER model architecture (a simple BiLSTM-CRF model)
def build_model(max_len, num_classes):
    model = Sequential([
        layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=50, input_length=max_len),
        layers.Bidirectional(layers.LSTM(50, return_sequences=True)),
        layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

# Define other hyperparameters
max_len = max_len
num_classes = len(label_encoder.classes_)

# Build the model
model = build_model(max_len, num_classes)

# Train the model
model.fit(train_sequences, train_labels, validation_data=(test_sequences, test_labels), epochs=10, batch_size=32)

# Save the trained model
model.save('ner_model.h5')