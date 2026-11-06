import os, glob, pickle
import numpy as np
from music21 import converter, instrument, note, chord

# 1. COLLECT MIDI DATA
# Put your .mid files in /data folder - Classical, Jazz etc.
# If no files, it will create demo data
DATA_PATH = "data"
OUTPUT_FILE = "notes.pkl"

def get_notes():
    notes = []
    midi_files = glob.glob(f"{DATA_PATH}/*.mid") + glob.glob(f"{DATA_PATH}/*.midi")
    print(f"Found {len(midi_files)} MIDI files")

    if len(midi_files) == 0:
        print("No MIDI found, using demo notes for training...")
        # Demo: C Major scale + Jazz pattern
        demo = ['C4','E4','G4','C5','A4','G4','E4','D4','C4','E4','G4','B4','D5','C5']
        return demo * 100

    # 2. PREPROCESS with music21 - Convert MIDI to note sequences
    for file in midi_files:
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                elif isinstance(element, note.Rest):
                    notes.append('REST')
        except Exception as e:
            print(f"Skip {file}: {e}")

    return notes

if __name__ == "__main__":
    notes = get_notes()
    print(f"Total notes: {len(notes)}")
    print(f"Unique notes: {len(set(notes))}")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(notes, f)
    print("Saved to notes.pkl")
