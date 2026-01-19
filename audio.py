import whisper

model = whisper.load_model("base")


def audio_to_text(file_path):
    if not file_path:
        raise FileNotFoundError(f"{file_path} not found")
    print(f"Transcribing audio file {file_path}")
    result = model.transcribe(str(file_path))
    text = result["text"]

    return text


# audio_to_text("/Users/lakshand/Desktop/python/RAG2/Capitol Towers.m4a")
