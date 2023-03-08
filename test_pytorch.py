import librosa
from transformers import AutoProcessor, WhisperForConditionalGeneration

model_id = "openai/whisper-tiny.en"
processor = AutoProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

y, sr = librosa.load('harvard.wav')

inputs = processor(y, return_tensors="pt")

input_features = inputs.input_features

generated_ids = model.generate(inputs=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(transcription)
