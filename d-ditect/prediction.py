import numpy as np
import librosa
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder 
from keras.models import load_model  # type: ignore
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences # type: ignore
from vosk import Model, KaldiRecognizer
import wave
import json  
from pydub import AudioSegment

# Constants
EMOTIONS = ['happy', 'sad', 'neutral', 'angry', 'disgust', 'fear', 'surprise']

NON_DEPRESSED_EMOTIONS = {
    "happy": 1.0,       #1.0 means fully non-depressed
    "neutral": 1.0,     
    "surprise": 1.0     
}
AUDIO_MODEL_PATH = "models/tess_trained_model.h5"
SENTIMENT_MODEL_PATH = "models/sentiment_analysis_cnn_model.h5"
MAX_SEQUENCE_LENGTH = 100

#---------------------------------------------------------------------   
# Extract MFCC features from an audio file.
#---------------------------------------------------------------------   
def extract_mfcc(filename):
    try:
        y, sr = librosa.load(str(filename), duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except Exception as e:
        raise ValueError(f"Failed to process audio file: {e}")
   
def audio_to_text(audio_file):
    try:
        # Set the path to the model directory
        model_path = "vosk-model-small-en-us-0.15" 
        
        # Convert audio to mono if it's stereo
        audio = AudioSegment.from_wav(audio_file)
        if audio.channels != 1:
            audio = audio.set_channels(1)
            audio.export(audio_file, format="wav")  
        
        # Load the Vosk model
        model = Model(model_path)
        
        # Open the audio file
        wf = wave.open(audio_file, "rb")
        # Check if the audio file is valid
        if wf.getnchannels() != 1:
            raise ValueError("Audio file must be mono (single channel)")
        
        # Create a recognizer instance
        recognizer = KaldiRecognizer(model, wf.getframerate())
        
        # Start recognizing
        text = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text += json.loads(result)["text"] + " "  
        
        # Return the transcribed text
        text =  text.strip()
    
        if not text:
            return "No speech detected"
            
        return text
        
    except Exception as e:
        raise ValueError(f"Failed to convert audio to text: {e}")
    

#---------------------------------------------------------------------   
# Predict depression status from audio file.
#---------------------------------------------------------------------   
def predict_depression(audio_file, enc):
    try:
        # Load the model
        model = load_model(AUDIO_MODEL_PATH)
        
        # Extract and prepare features
        mfcc_features = extract_mfcc(audio_file)
        X_input = mfcc_features.reshape(1, 40, 1)
        
        # Make prediction
        y_pred_probs = model.predict(X_input, verbose=0)
        predicted_label = enc.inverse_transform(y_pred_probs)[0][0]
        
        # Get the probability from NON_DEPRESSED_EMOTIONS, default to 0 (fully depressed)
        non_depressed_prob = NON_DEPRESSED_EMOTIONS.get(predicted_label, 0.0)
        depressed_prob = 1.0 - non_depressed_prob 

        return (
        f'  - Emotion: {predicted_label}'
        f' - Depression: {depressed_prob * 100:.2f}%'
        )

    except Exception as e:
        raise ValueError(f"Failed to analyze audio: {e}")

#--------------------------------------------------------------------- 
# Analyze audio file for depression detection.
#--------------------------------------------------------------------- 
def analyze_depression_audio(audio_file):
    try:
        # Validate file existence
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
            
        # Initialize and fit encoder
        enc = OneHotEncoder(sparse_output=False)
        enc.fit(np.array(EMOTIONS).reshape(-1, 1))
        
        # Make prediction
        return predict_depression(audio_file, enc)
        
    except Exception as e:
        return f"PREDICTION ERROR: {str(e)}"

#--------------------------------------------------------------------- 
# Predict sentiment from text input.
#--------------------------------------------------------------------- 
def predict_sentiment(audio_file):
    try:

        transcribed_text = audio_to_text(audio_file)
        print(f"Transcribed Text: {transcribed_text}")

        if transcribed_text in ["No speech detected", "Error processing audio"]:
            return "Unable to analyze - No clear speech detected", 0.5, transcribed_text

        # Validate input
        if not transcribed_text.strip():
            raise ValueError("Input text cannot be empty")
            
        # Load model and tokenizer
        model = load_model(SENTIMENT_MODEL_PATH)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Preprocess text
        input_ids = tokenizer.encode(
            transcribed_text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        )
        
        # Pad sequence
        input_ids = pad_sequences(
            [input_ids],
            maxlen=MAX_SEQUENCE_LENGTH,
            dtype='long',
            truncating='post',
            padding='post'
        )
        
        # Add channel dimension if needed
        input_ids = np.expand_dims(input_ids, axis=-1)
        
        # Make prediction
        pred_prob = model.predict(input_ids, verbose=0)[0][0]
        prediction = ('No depression detected'
                     if pred_prob > 0.5
                     else 'Depression detected')

        return prediction, pred_prob, transcribed_text
        
    except Exception as e:
        raise ValueError(f"Failed to analyze text: {e}")

