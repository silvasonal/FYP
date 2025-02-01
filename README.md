# Multimodal Depression Detection Setup

This project focuses on detecting depression using multimodal data, which includes audio, visual, and textual inputs. The system utilizes machine learning models for processing and analyzing these data types to assess the mental health status of individuals.

## Requirements

- Python 3.x

## Setup Instructions

### 1. Clone the Repository
First, clone this repository to your local machine using the following command:
git clone "https://github.com/silvasonal/FYP.git"


### 2. Download the Vosk Model 
This project requires the Vosk speech recognition model for analyzing audio data. Download the vosk-model-small-en-us-0.15 from the official Vosk website: https://alphacephei.com/vosk/models

"After downloading and extracting the Vosk model, place the extracted folder in the root directory of the d-ditect folder in this repository."

### 3. Install Required Libraries
pip install keras tensorflow opencv-python flask sounddevice soundfile scikit-learn librosa matplotlib

### 4. Execute the Following Commands to Run the Application:
cd d-ditect
python app.py

###------------------------------------------------------------------------------------------------------------

### 5. To run the test Cases
Install Required Libraries in the Testing Folder:
    npm init playwright@latest

Execute the Test Cases using the following command in Testing Folder:
    npx playwright test automationTesting.spec.ts --headed
