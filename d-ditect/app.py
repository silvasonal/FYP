import os
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, Response, jsonify, request, session, redirect, url_for, g
import threading
import sounddevice as sd  
import soundfile as sf
  
from prediction import predict_sentiment, analyze_depression_audio

# Global variables for video and audio control
video = None
is_streaming = False
is_recording = False
audio_thread = None

RECORDINGS_DIR = 'recordings'
AUDIO_FILEPATH = os.path.join(RECORDINGS_DIR, f'audio_recoding.wav')
REPORT_PATH = os.path.join(RECORDINGS_DIR, f'detection_report.txt')

def write_user_to_report():
    username = session.get('username')
    if username:
        # Read the existing content of the report
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, 'r') as file:
                content = file.read()
        else:
            content = ""  
        
        # Write the username at the top followed by the existing content
        with open(REPORT_PATH, 'w') as file:
            file.write(f"User: {username}\n\n") 
            file.write(content) 

#----------------------------------------------------------------------------------------------------
# AUDIO ANALYSIS 
#----------------------------------------------------------------------------------------------------
# Ensure recordings directory exists
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

def record_audio():
    global is_recording
    
    # Audio recording parameters
    sample_rate = 44100
    channels = 2
    
    # Initialize the audio recording
    audio_data = []
    
    def callback(indata, frames, time, status):
        if is_recording:
            audio_data.append(indata.copy())
    
    # Start recording
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        while is_recording:
            sd.sleep(100)  
    
    # Save the recording if we have data
    if audio_data:
        audio_array = np.concatenate(audio_data, axis=0)
        sf.write(AUDIO_FILEPATH, audio_array, sample_rate)  
        return AUDIO_FILEPATH
    return None

def start_audio_recording():
    global audio_thread, is_recording
    is_recording = True
    audio_thread = threading.Thread(target=record_audio)
    audio_thread.start()

def stop_audio_recording():
    global is_recording, audio_thread
    is_recording = False
    if audio_thread:
        audio_thread.join()
        audio_thread = None

def check_wav_in_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            return os.path.join(folder_path, file)  # Return the first .wav file found
    return None

def analyze_audio():    
    
    if check_wav_in_folder(RECORDINGS_DIR):
        result = analyze_depression_audio(AUDIO_FILEPATH)
    
        with open(REPORT_PATH, 'a') as report_file:
            report_file.write('Audio Analysis\n')
            report_file.write('---------------\n')
            report_file.write(result)
            report_file.write('\n\n')

#----------------------------------------------------------------------------------------------------
# SENTIMENT ANALYSIS (TEXTUAL)
#----------------------------------------------------------------------------------------------------
def analyze_sentiment():
    prediction, pred_prob, transcribed_text = predict_sentiment(AUDIO_FILEPATH)
          
    with open(REPORT_PATH, 'a') as report_file:
        report_file.write('Sentiment Analysis\n')
        report_file.write('------------------\n')
            
        if transcribed_text in ["No speech detected", "Error processing audio"]:
            report_file.write("Status: No clear speech detected in the audio\n")
            report_file.write("Please ensure you speak clearly during the recording\n")
        else:
            report_file.write(f'Transcribed Text from Audio: {transcribed_text}\n')
            report_file.write(f'Depression Probability: {100-pred_prob*100:.2f}%\n')
            report_file.write(f'PREDICTION: {prediction}')
            report_file.write('\n\n')

#----------------------------------------------------------------------------------------------------
# VISUAL ANALYSIS (VIDEO)
#----------------------------------------------------------------------------------------------------
# Load the cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Load the trained model 
model = load_model('models/visual_trained_model.h5')

# define the Labels_dict with emotions 
labels_dict = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

def initialize_video():
    global video
    if video is None:
        video = cv2.VideoCapture(0)
    return video

def release_video():
    global video, is_streaming
    if video is not None:
        video.release()
        video = None
    is_streaming = False

def delete_all_files_in_folder():
    try:
        for file in os.listdir(RECORDINGS_DIR):
            file_path = os.path.join(RECORDINGS_DIR, file)
            if os.path.isfile(file_path):  # Ensure it's a file
                os.remove(file_path)                
    except Exception as e:
        print(f"An error occurred: {e}")

# render image while capturing the emotions
def visual_detection_and_analysis():
    global video, is_streaming
    
    emotion_count = {'Happy': 0, 'Sad': 0, 'Angry': 0, 'Fear': 0, 'Surprise': 0, 'Neutral': 0, 'Disgust': 0}
    frame_count = 0
    face_count = 0
    
    initialize_video()
    is_streaming = True
    
    while is_streaming and video is not None:
        success, frame = video.read()
        if not success:
            break
            
        #preprocess the frame
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            face_count += 1
            
            for (x, y, w, h) in faces:
                sub_face_image = gray[y:y+h, x:x+w]
                resized = cv2.resize(sub_face_image, (48, 48))
                normalize = resized/255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                predicted_emotion = labels_dict[label]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                emotion_count[predicted_emotion] += 1
        
        if face_count > 0:
            total_emotions = sum(emotion_count.values())
            percentages = {emotion: (count/total_emotions)*100 
                         for emotion, count in emotion_count.items()}
            
            y_pos = 15
            cv2.putText(frame, f'Frames Processed: {frame_count}', (5, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(frame, f'Faces Detected: {face_count}', (5, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            
            for emotion, percent in percentages.items():
                cv2.putText(frame, f'{emotion}: {percent:.1f}%', (5, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20
                
            save_analysis_results(percentages)

        if is_streaming:
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

#Save visual analysis to the text file
def save_analysis_results(percentages):
    try:
        depressed_percent = (percentages['Sad'] + percentages['Angry'] + 
                            percentages['Fear'] + percentages['Disgust']/2 + 
                            percentages['Surprise']/2 + percentages['Neutral']/2)
        non_depressed_percent = (percentages['Happy'] + percentages['Neutral']/2 + 
                                percentages['Disgust']/2 + percentages['Surprise']/2)
        
        with open(REPORT_PATH, 'w') as report_file:
            report_file.write('Visual Analysis\n')
            report_file.write('----------------\n')
            for emotion, percent in percentages.items():
                report_file.write(f'{emotion}: {percent:.2f}%\n')
        
            report_file.write(f'\nDepression probability: {depressed_percent:.2f}%\n')
            report_file.write(f'Non-Depression probability: {non_depressed_percent:.2f}%\n')
            report_file.write(f'\nPREDICTION: ')
            report_file.write('Depression detected' if depressed_percent > non_depressed_percent 
                             else 'No depression detected')
            report_file.write('\n\n')
    except Exception as e:
        print(f"Error saving analysis results: {e}")



#----------------------------------------------------------------------------------------------------
# ROUTES
#----------------------------------------------------------------------------------------------------
# Initialize the Flask app
app = Flask(__name__)
app.secret_key = '123#'  
        
@app.route('/')
def home():
    if 'username' not in session:
        return redirect('/login')  
    return render_template('index.html', username=session.get('username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if username:
            session['username'] = username
            return redirect('/')  
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear() 
    delete_all_files_in_folder()
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(visual_detection_and_analysis(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_video():
    global is_streaming
    is_streaming = True
    start_audio_recording() 
    delete_all_files_in_folder() 
    return jsonify({"status": "started"})

@app.route('/stop')
def stop_video():
    global is_streaming
    is_streaming = False
    stop_audio_recording()  
    release_video()
    return jsonify({"status": "stopped"})

@app.route('/analyze', methods=['POST'])
def analyze_audio_text():
    try:
        if not os.path.exists(AUDIO_FILEPATH):
            return jsonify({
                "status": "error", 
                "message": "No audio recording found. Please ensure recording is complete."
            }), 400
            
        analyze_audio()  # Analyze the audio
        analyze_sentiment()  # Analyze transcribed audio for sentiment
        write_user_to_report() 
        
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": "Analysis failed. Please ensure audio was recorded properly."
        }), 500

@app.route('/refresh')
def refresh_app():
    delete_all_files_in_folder()
    return jsonify({"status": "refreshed"})

@app.route('/get_report')
def get_report():
    try:
        username = session.get('username')

        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, 'r') as file:
                content = file.read()
            return jsonify({"status": "success", "content": content, "username": username,})
        else:
            return jsonify({"status": "error", "message": "Report file not found"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, 'w') as f:
            f.write('No analysis data available yet.\n')
    
    delete_all_files_in_folder()
    app.run(debug=False)