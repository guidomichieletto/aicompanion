import os
import yaml
import logging

from flask import Flask, request, jsonify, render_template

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

from kokoro import KPipeline, KModel  # genera audio
import soundfile as soundfile         # gestisce file wave
from pydub import AudioSegment        # unisce i file
import base64                         # codifica in base64

import whisper

# Config
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Web Server
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# load model
lcmodel = ChatOllama(model=config['models']['ai'], temperature=0, reasoning=False)

# RAG
embeddings = OllamaEmbeddings(model=config['models']['embedding'])
vectorstore = InMemoryVectorStore.load(config['rag_db_path'], embeddings)
retriever = vectorstore.as_retriever()

# TTS
kmodel = KModel(model=config['kokoro']['model'], config=config['kokoro']['config'])
kpipeline = KPipeline(lang_code='i', model=kmodel)

# STT
wmodel = whisper.load_model(config['models']['whisper'], device="cuda")

# Protection against prompt injection
def is_safe(message):
    prompt = config['prompts']['secure'].format(message=message)
    response = lcmodel.invoke([('human', prompt)])
    return not ("unsafe" in response.content.lower())

# creating context
context = [
    ('system', config['prompts']['first']),
    ('system', '')
]

def text_to_speech(text):
    # ensure output folder exists
    os.makedirs('./generated_audio', exist_ok=True)
    generator = kpipeline(text, voice=config['kokoro']['voice'], speed=0.9)

    audio = AudioSegment.empty()

    for i, (gs, ps, audio_chunk) in enumerate(generator):
        soundfile.write(f'./generated_audio/output_{i}.wav', audio_chunk, 24000, "PCM_16")
        audio += AudioSegment.from_wav(f'./generated_audio/output_{i}.wav')

    audio.export('./generated_audio/final_output.wav', format='wav')

    with open('./generated_audio/final_output.wav', 'rb') as f:
        audio_bytes = f.read()
    
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    return audio_b64

def speech_to_text(file_path):
    try:
        result = wmodel.transcribe(audio = file_path, language = "it", fp16 = False)
        return result.get("text", "")
    except Exception as e:
        print(f"Errore durante la trascrizione: {e}")
        return ""
    
def ai_req(message):
    if not is_safe(message):
        logging.warning("Unsafe request: " + message)
        unsafe_resp = "Mi dispiace, non posso rispondere a questa richiesta."
        return {"response": unsafe_resp, "audio": text_to_speech(unsafe_resp), "emotion": "ARRABBIATO"}

    logging.info("User: " + message)
    context.append(('human', message))

    # rag
    doc = retriever.invoke(message)
    doc_text = "\n".join([d.page_content for d in doc])
    doc_text = "Usa questo testo per rispondere alla domanda:\n" + doc_text + "\nSe non conosci la risposta, dì che non lo sai.\n"
    context[1] = ('system', doc_text)

    response = lcmodel.invoke(context)
    context.append(('ai', response.content))

    # generate audio
    audio_b64 = text_to_speech(response.content)

    # context garbage collector
    # remove oldest user-assistant pair but keep system prompt and rag info
    if len(context) > config['max_context_length']:
        context.pop(2)
        context.pop(2)

    logging.info("AI: " + response.content + "\n")
    return {"response": response.content, "audio": audio_b64, "emotion": "NEUTRO"}

@app.route("/")
def index():
    return render_template("index.html", history = context)

# Crystal compatibility endpoint
@app.route("/crystal", methods=['POST'])
def crystal():
    audio = request.get_data()
    with open('./audio/user_input.wav', 'wb') as f:
        f.write(audio)

    message = speech_to_text('./audio/user_input.wav')
    req = ai_req(message)

    return jsonify({
        "text": req["response"],
        "audio": req["audio"],
        "emotion": req["emotion"]
    })

# Web app endpoint
@app.route("/aicompanion", methods=['POST'])
def aicompanion():
    message = request.get_json().get('message')
    audio_b64 = request.get_json().get('audio')

    if audio_b64 not in ["", None]:
        # ensure output folder exists before writing
        os.makedirs('./audio', exist_ok=True)
        
        audio_bytes = base64.b64decode(audio_b64)

        with open('./audio/user_input.wav', 'wb') as f:
            f.write(audio_bytes)
        
        message = speech_to_text('./audio/user_input.wav')

    req = ai_req(message)

    return jsonify(req)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config['server_port'], debug=False)