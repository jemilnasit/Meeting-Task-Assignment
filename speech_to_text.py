import whisper
from joblib import dump

stt_model = whisper.load_model("base")
result = stt_model.transcribe("audio.mp3")

text = result["text"]
dump(text, "transcription.joblib")

print(text)