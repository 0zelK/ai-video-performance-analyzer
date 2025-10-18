# AI-Powered Video Performance Analyzer

This is a web app that predicts the potential reach of a video before it’s published, using only its metadata (title, description, keywords, duration, etc.).

This was built as part of an AI internship in the **R&D department** to explore how pre-upload metadata can estimate video performance.  
The project includes data cleaning, feature engineering, model training, and full-stack deployment.

## Overview

The tool estimates video performance based on metadata through two specialized models trained on different datasets:
- **Long-form model (YouTube videos ≥180s): RMSE ≈ 1.24
- **Short-form model (TikTok and Shorts <180s): RMSE ≈ 1.63
It provides users with estimated performance scores and practical feedback on metadata quality.

## Technologies

**Languages:** Python, JavaScript, HTML/CSS/CSS  
**Frameworks and Libraries:** FastAPI, LightGBM, SentenceTransformers, Pandas, NumPy, Scikit-learn  
**Tools:** ffprobe, Joblib, Uvicorn

## Data and Model

### Datasets
Three public datasets were combined and cleaned (≈155K total entries):
- YouTube 131K Dataset: https://huggingface.co/datasets/vargr/youtube  
- YouTube Shorts/Longs Dataset: https://www.kaggle.com/datasets/taimoor888/youtube-long-vs-shorts-video-analysis?
- TikTok Metadata Dataset: https://www.kaggle.com/datasets/raminhuseyn/dataset-from-tiktok

### Feature Engineering
- Text features: title/description lengths, word counts, stopword ratios, capitalization and punctuation ratios  
- Numeric features: duration, duration category, missing-data flags  
- **Text embeddings:** 384-dimensional SentenceTransformer vectors for title, description, and keywords  

### Model Training
- Algorithm: LightGBM  
- Optimized parameters: `num_leaves`, `max_depth`, `feature_fraction`, `learning_rate`, `n_estimators`  
- Train/validation split: 85 / 15  
- Metric: RMSE

Note: The use of text embeddings improved validation error by about **35%**, allowing the model to interpret semantic relationships instead of memorizing text patterns.

## Backend Pipeline

1. Receives metadata and optional video file from the user  
2. Extracts duration using `ffprobe`  
3. Reproduces feature-engineering steps used in training  
4. Generates text embeddings for title, description, and keywords  
5. Combines all features into a single vector  
6. Loads the corresponding LightGBM model (long or short)  
7. Returns prediction and feedback in JSON format

## Model Interpretation and Feedback

- I used SHAP to identify which features most influenced the prediction outputs.
- For long-form videos, the most impactful variables included text embeddings (title, description, keywords), duration, and the title-to-description length ratio.
- These insights were used to design the app’s feedback system, which compares the user’s metadata against top 10% performers in the dataset and generates improvement suggestions based on averaged high-performing values.

## Frontend

A minimal HTML/CSS/JavaScript interface:
- Dark-theme layout  
- Model selection (long or short)  
- Input form for metadata and video upload  
- Dynamic result display with estimated views and feedback

## Results Summary

| Model Type | Dataset Size | Validation RMSE | Notes                          |
|-------------|--------------|----------------|--------------------------------|
| Long-form   | 100K+ rows   | 1.24           | Consistent and accurate        |
| Short-form  | 55K+ rows    | 1.63           | Higher noise, less predictable |

## Limitations

- Based only on metadata (no thumbnail, trend, or recommendation signals)  
- Limited to YouTube and TikTok datasets  
- Short-form results affected by noise and missing text information

## Installation

### Requirements
- Python 3.10+  
- ffmpeg (with `ffprobe` available in PATH)  
- Modern web browser
- Download the '\models' folder and add it to the app directory. You can find it in this link: https://drive.google.com/drive/folders/1LBNjx4ZfpIdIFivf948Ax3Ogo5yCpbXD?usp=sharing

### Dependencies (bash)
pip install fastapi uvicorn joblib sentence-transformers torch numpy pandas scikit-learn lightgbm python-multipart ffmpeg-python

## Running the project

You will need 2 terminals for this. 
- In the first terminal, navigate to the project's backend folder, then run:
uvicorn main:app --reload

- In the second terminal, navigate to the project's frontend folder, then run:
python3 -m http.server 3000

- Then, open "http://localhost:3000" in your browser.
- You're done!


## Note

Make sure to check out this google drive link: https://drive.google.com/drive/folders/1LBNjx4ZfpIdIFivf948Ax3Ogo5yCpbXD?usp=sharing.
I have included in it both .pkl models (which you need to run the app), the scripts and helper functions I used along the way, and the data including both the raw and cleaned datasets.
