from run import process_video
from src.utils import transcribe

def process_and_transcribe(video_path, temp_dir, model):
    process_video(video_path, temp_dir)
    transcribed_data = transcribe(model, temp_dir)
    return transcribed_data
