import gc
import time

import streamlit as st
from moviepy.editor import AudioFileClip


class VideoProcessor:
    def __init__(self, video_path: str, directory: str) -> None:
        """
        Initialize the Processor with the specified video path and directory path.

        Args:
            video_path (str): The path to the video file.
            directory (str): The directory where output files will be saved.

        """
        self.video_path: str = video_path
        self.directory: str = directory

    def get_chunks(self, args: tuple[str, float, float, str]) -> None:
        """
        Process a chunk of audio from the video file.

        Args:
            args (tuple): A tuple containing the input file path, start time, end time,
                          and output directory.

        """
        input_file, start_time, end_time, output_dir = args
        audio = AudioFileClip(input_file).subclip(start_time, end_time)
        output_path = f"{output_dir}/{start_time}:{end_time}_lec.wav"
        audio.write_audiofile(output_path, ffmpeg_params=["-ac", "1"], codec="pcm_s16le")
        audio.close()
        gc.collect()

    def process_video(self) -> None:
        """
        Process the video file to extract audio chunks and display progress.

        This method reads the video file, divides it into chunks, and processes each chunk
        to save the audio segments in the specified directory.
        """
        audioclip = AudioFileClip(self.video_path)
        duration = audioclip.duration
        audioclip.close()

        chunk_duration = 50
        chunks = int(duration / chunk_duration) + 1

        progress_text = "Первый этап в процессе. Пожалуйста, подождите."
        progress_bar = st.progress(0, text=progress_text)

        for i in range(chunks):
            args = (self.video_path, max(0, i * chunk_duration - 5), min((i + 1) * chunk_duration + 5, duration), self.directory)
            self.get_chunks(args=args)

            progress_ratio = min((i + 1) / chunks, 1.0)
            progress_bar.progress(progress_ratio, text=f"{int(progress_ratio * 100)}% обработано")

            time.sleep(0.1)

        progress_bar.empty()
