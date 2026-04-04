from deeprhythm import DeepRhythmPredictor

model = DeepRhythmPredictor()

# to include confidence
tempo, confidence = model.predict('el_rescate.mp3', include_confidence=True)

print(f"Predicted Tempo: {tempo} BPM {confidence}")