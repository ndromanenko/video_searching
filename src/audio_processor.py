import gc
from pathlib import Path

import streamlit as st
import torch

from src.stt_model import STTModel


class AudioProcessor:
    def __init__(self, model: STTModel, directory: str) -> None:
        """
        Initialize the AudioProcessor with the specified model and directory.

        Args:
            model (STTModel): The speech-to-text model to use for transcription.
            directory (str): The directory where audio files are located.

        """
        self.model: STTModel = model
        self.directory: str = directory

    def transcribe_files(self) -> list[str]:
        """
        Transcribe audio files in the specified directory using the provided model.

        This method processes audio files in batches, transcribes them, and returns
        a list of dictionaries containing the time and transcribed text.

        Returns:
            list: A list of dictionaries with 'time' and 'text' keys for each transcribed audio.
            
        """
        directory = Path(self.directory)

        dev_type = self.model.device if isinstance(self.model.device, str) else self.model.device.type
        batch_size = 1 if dev_type == "cpu" else 16

        chunks_time_text = {}

        audio_files = [f for f in directory.iterdir() if f.is_file() and not f.name.startswith(".")]

        progress_text = "Второй этап в процессе. Пожалуйста, подождите."
        progress_bar = st.progress(0, text=progress_text)

        for i in range(0, len(audio_files), batch_size):
            batch_files = audio_files[i:i + batch_size]
            file_paths = [str(audio) for audio in batch_files]

            results = self.model.transcribe(file_paths)

            for audio, result in zip(batch_files, results, strict=True):
                times = audio.stem.split("_")[0]
                chunks_time_text[times] = result

            progress_ratio = min((i + batch_size) / len(audio_files), 1.0)
            progress_bar.progress(progress_ratio, text=f"{int(progress_ratio * 100)}% обработано")

            if dev_type == "cuda":
                torch.cuda.empty_cache()
            elif dev_type == "mps":
                torch.mps.empty_cache()
            gc.collect() 

        progress_bar.empty()  

        chunks = []
        for time, text in chunks_time_text.items():
            entry = {
                "time": time,
                "text": text.text
            }
            chunks.append(entry)
        
        return chunks
