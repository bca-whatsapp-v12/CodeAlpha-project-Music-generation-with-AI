import pickle, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Load preprocessed notes
with open('notes.pkl', 'rb') as f:
    notes = pickle.load(f)

# Prepare sequences - 100 notes input -> predict next note
pitchnames = sorted(set(notes))
n_vocab = len(pitchnames)
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

sequence_length = 50
network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in seq_in])
    network_output.append(note_to_int[seq_out])

n_patterns = len(network_input)
print(f"Patterns: {n_patterns}, Vocab: {n_vocab}")

# Reshape for LSTM [samples, time steps, features]
X = np.reshape(network_input, (n_patterns, sequence_length, 1))
X = X / float(n_vocab)
y = to_categorical(network_output, num_classes=n_vocab)

# Build LSTM Model - RNN with LSTM
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Save best weights
checkpoint = ModelCheckpoint("weights_best.hdf5", monitor='loss', save_best_only=True, mode='min')
model.fit(X, y, epochs=50, batch_size=64, callbacks=[checkpoint], verbose=1)

# Save vocab for generation
with open('pitchnames.pkl', 'wb') as f:
    pickle.dump(pitchnames, f)
print("Training Done! Saved weights_best.hdf5")
