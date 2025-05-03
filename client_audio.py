import openai
import time, os

HOST = 'localhost'
PORT = 8000
PROTOCOL = 'http'
BASE_URL = f"{PROTOCOL}://{HOST}:{PORT}/v1/"
client = openai.OpenAI(
    api_key="EMPTY",
    base_url=BASE_URL,
)

MODEL = 'openai/whisper-tiny'

audio_file_path = '/home/me/Desktop/bot_clicker/1.mp3'

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model=MODEL
        )
    return transcription.text


if __name__ == '__main__':
    timer = time.time()

    # Transcribe the audio
    transcription = transcribe_audio(audio_file_path)

    # Print the transcription
    print("Transcription:")
    print(transcription)

    print(f"Time taken: {time.time() - timer:.2f} seconds")
