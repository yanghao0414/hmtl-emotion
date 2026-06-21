import librosa
from transformers import Wav2Vec2Processor
import torch
import numpy as np
import warnings
import os

#  librosa 
warnings.filterwarnings("ignore", category=UserWarning)

#  Wav2Vec 2.0 
PROCESSOR_NAME = "facebook/wav2vec2-base" 

class AudioPreprocessor:
    def __init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_NAME)
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def preprocess_audio(self, audio_file_path):
        if not os.path.exists(audio_file_path):
            return None
            
        try:
            speech, rate = librosa.load(audio_file_path, sr=self.sampling_rate)
            
            processed = self.processor(
                speech, 
                sampling_rate=self.sampling_rate, 
                return_tensors="pt", 
                padding=True
            )
            
            input_values = processed.input_values.squeeze(0)
            
            # Wav2Vec2Processor  attention_mask
            # 1attention_mask
            attention_mask = torch.ones(input_values.shape, dtype=torch.long)
            
            return {
                'input_values': input_values,
                'attention_mask': attention_mask
            }

        except Exception as e:
            return None
