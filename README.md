# Emotion Detection Web App - Final

This project is prepared for offline use. The repository contains all app code and UI. 
Due to environment limitations, the pre-trained model file (emotion_model.h5) was not bundled here.
Please run the included `download_model.py` locally on your machine to fetch a pre-trained `.h5` model,
or manually download a model and place it at `models/emotion_model.h5`.

Example (download from HuggingFace or GitHub raw):
python download_model.py --source "https://huggingface.co/geeknix/emotion-reg/resolve/main/emotion_model.h5"

After placing `models/emotion_model.h5`, install requirements and run the app:
pip install -r requirements.txt
python app.py

Notes:
- If TensorFlow is not installed, the app will run in fallback mode and always return 'neutral'.
- The app includes a theme toggle (light/dark), webcam capture, image preview, spinner, and face crop preview.
- Hosting files for Render are included (Procfile, render.yaml).
