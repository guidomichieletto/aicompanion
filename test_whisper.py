import whisper

wmodel = whisper.load_model("./models/small.pt", device="cuda")

def audio_to_text(file_path):
    result = wmodel.transcribe(audio = file_path, language = "it", fp16 = False)
    return result["text"]

print(audio_to_text("./generated_audio/final_output.wav"))