import jax
import librosa
import jax.numpy as jnp
from transformers import WhisperProcessor, FlaxWhisperForConditionalGeneration

model_id = "openai/whisper-tiny.en"
processor = WhisperProcessor.from_pretrained(model_id)
model = FlaxWhisperForConditionalGeneration.from_pretrained(model_id)

y, sr = librosa.load('harvard.wav')

input_features = processor(y, return_tensors="np").input_features

jit_generate = jax.jit(model.generate, static_argnames=["max_length"])

input_features = jnp.array(input_features, dtype=jnp.float16)

pred_ids = model.generate(input_features, max_length=128)

transcription = processor.batch_decode(pred_ids.sequences, skip_special_tokens=True)[0]

print(transcription)
