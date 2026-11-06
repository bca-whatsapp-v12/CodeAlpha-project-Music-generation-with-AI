# Task 3: Music Generation with AI - CodeAlpha
**Domain: Artificial Intelligence**

### Objective
Train AI to generate new music sequences in Classical / Jazz style.

### Criteria Covered
1. **Collect MIDI Data:** Place .mid files in `/data` folder (Classical, Jazz). `preprocess.py` auto-reads them.
2. **Preprocess with music21:** MIDI -> Note / Chord / Rest sequences, saved as `notes.pkl`
3. **Train Deep Learning Model:** 3-Layer LSTM (RNN) - 256 units, Dropout 0.3, Softmax. Trained for 50 epochs. Saved as `weights_best.hdf5`
4. **Generate & Save:** `generate.py` predicts 200 new notes -> Converts to `generated_music.mid` -> Playable in VLC / any MIDI player. Convert to .wav via FluidSynth.

### How to Run
