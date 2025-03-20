import gc
import os
import time
import torch
from moviepy.editor import AudioFileClip
import streamlit as st


def process_chunk(args):
    input_file, start_time, end_time, output_dir = args
    audio = AudioFileClip(input_file).subclip(start_time, end_time)
    output_path = f"{output_dir}/{start_time}:{end_time}_lec.wav"
    audio.write_audiofile(output_path, ffmpeg_params=["-ac", "1"], codec="pcm_s16le")
    audio.close()
    gc.collect()

def process_video(file, output_dir):
    audioclip = AudioFileClip(file)
    duration = audioclip.duration
    audioclip.close()

    chunk_duration = 50 
    chunks = int(duration / chunk_duration) + 1

    progress_text = "First stage in progress. Please wait."
    progress_bar = st.progress(0, text=progress_text)
    
    for i in range(chunks):
        args = (file, max(0, i * chunk_duration - 5), min((i + 1) * chunk_duration + 5, duration), output_dir)
        process_chunk(args=args)

        progress_bar.progress((i + 1) / chunks, text=progress_text)
        time.sleep(0.1)
    
    progress_bar.empty()


def transcribe(model, directory):

    audio_files = os.listdir(directory)

    batch_size = 16
    chunks_time_text = {}

    audio_files = [f for f in audio_files if not f.startswith(".")]

    total_batches = len(audio_files) // batch_size + (1 if len(audio_files) % batch_size > 0 else 0)
    progress_text = "Second stage in progress. Please wait."
    progress_bar = st.progress(0, text=progress_text)

    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]
        file_paths = [f"{directory}/{audio}" for audio in batch_files]
        
        results = model.transcribe(file_paths)
        
        for audio, result in zip(batch_files, results):
            times = audio.split('_')[0]
            chunks_time_text[times] = result
        
        torch.mps.empty_cache()
        gc.collect()

        progress_percentage = (i // batch_size + 1) / total_batches
        progress_bar.progress(progress_percentage)

    progress_bar.empty()

    chunks = []
    for time in chunks_time_text.keys():
        entry = {
            'time': time,
            'text': chunks_time_text[time]
        }
        chunks.append(entry)
    
    return chunks


def add_data_with_metadata(data, retriever):
    texts = [item['text'] for item in data]
    metadatas = [{'time': item['time']} for item in data]

    if not texts:
        print("Warning: No texts to embed!")
        return 
    
    retriever.add_texts(texts=texts, metadatas=metadatas)


def search(query, retriever, top_k=1):
    similarity_content = retriever.similarity_search_with_score(query, k=top_k)
    
    for_ranking = []
    mapping = dict()
    context = ''
    number = 1

    for doc, score in similarity_content:

        time = int(doc.metadata['time'].split(':')[0])
        minute = time // 60
        second = time % 60
        context = context + f'{number}. ' + doc.page_content + '\n'
        number += 1 

        if second < 10:
            for_ranking.append(doc.page_content)
            mapping[doc.page_content] = f'Время: {minute}:0{second}'
        else:
            for_ranking.append(doc.page_content)
            mapping[doc.page_content] = f'Время: {minute}:{second}'

    return for_ranking, mapping, context