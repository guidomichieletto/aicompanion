from kokoro import KPipeline, KModel  # genera audio
import soundfile as soundfile         # gestisce file wave
from pydub import AudioSegment        # unisce i file
import base64                         # codifica in base64

kmodel = KModel(model='./models/kokoro-v1_0.pth', config='./models/config.json')
kpipeline = KPipeline(lang_code='i', model=kmodel)

def text_to_speech(text):
    generator = kpipeline(text, voice='./models/voices/im_nicola.pt', speed=0.9)

    audio = AudioSegment.empty()

    for i, (gs, ps, audio_chunk) in enumerate(generator):
        soundfile.write(f'./generated_audio/output_{i}.wav', audio_chunk, 24000, "PCM_16")
        audio += AudioSegment.from_wav(f'./generated_audio/output_{i}.wav')

    audio.export('./generated_audio/final_output.wav', format='wav')

    with open('./generated_audio/final_output.wav', 'rb') as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    return audio_b64

print(text_to_speech("Ciao, come stai? Questo è un test di sintesi vocale."))