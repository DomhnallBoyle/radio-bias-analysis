"""
    Filename: 
    Description:
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: domhnallboyle@gmail.com
    Python Version: 3.6
"""
import argparse
import os
import pickle
import tempfile
import sys
from pathlib import Path

from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm

sys.path.append(os.environ.get('SPK_VERIFICATION_REPO'))
from encoder import inference as encoder

encoder.load_model(Path('speaker_encoder.pt'))


def get_embedding(audio_path):
    preprocessed_wav = encoder.preprocess_wav(audio_path)
    embedding = encoder.embed_utterance(preprocessed_wav)

    return embedding


def main(args):
    seconds_per_chunk = 5
    chunk_length_ms = seconds_per_chunk * 1000
    audio = AudioSegment.from_file(args.audio_path, 'm4a')
    embedding_data = []

    for i, chunk in enumerate(tqdm(make_chunks(audio, chunk_length_ms))):
        start_time = i * chunk_length_ms
        end_time = start_time + chunk_length_ms

        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            chunk.export(f.name, format='wav')
            embedding = get_embedding(audio_path=f.name)
            embedding_data.append([start_time / 1000, end_time / 1000, embedding])

    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(embedding_data, f)


if __name__ == '__main__':
    # USAGE: python -W ignore processor.py <audio_path>
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_path')

    main(parser.parse_args())

