# AI Storytelling App for Kids

This project is a Streamlit storytelling application for ISOM5240 Assignment 1.

## Features

- Upload an image
- Generate an image caption with a Hugging Face image model
- Generate a 50-100 word story for children
- Convert the story into speech

## Models and Tools

- Image captioning model: `Salesforce/blip-image-captioning-base`
- Story generation model: `google/flan-t5-small`
- Text-to-speech: `gTTS`
- Web framework: `Streamlit`

## Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Streamlit Cloud Deployment

1. Upload this project to GitHub.
2. Sign in to Streamlit Cloud.
3. Create a new app from this repository.
4. Set the main file path to `streamlit_app.py` if needed.
5. Deploy and copy the public URL for submission.
