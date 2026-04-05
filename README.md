# Live Vision American Sign Language Translator
<img width="2382" height="1760" alt="Gemini_Generated_Image_eilj2meilj2meilj" src="https://github.com/user-attachments/assets/551bc2b8-f75a-4776-b401-01ce59683074" />

## Set up
### Create and activate environment
conda create -n asl python=3.10

conda activate asl

### Install dependencies

pip install -r requirements.txt

## Training PipeLine
![Hand Landmarks](https://github.com/user-attachments/assets/b6c4abce-04c5-482c-8c49-382df1f4c6bc)

## Tech Stack

- Python 3.10
- MediaPipe — hand landmark detection (21 joints per hand)
- OpenCV — camera capture and frame drawing
- scikit-learn — Random Forest classifier

## Structure
```
backend/
├── classifier/
│   ├── classify_letter_model.p      # trained letter model (A–Z, del, space)
│   └── classify_digit_model.p       # trained digit model (0–9)
│
├── dataset/
│   └── archive/
│       ├── asl_alphabet_train/      # letter training images (3000/class)
│       ├── asl_digit_train/         # digit training images (500/class)
        ├── asl_alphabet_test/      
│       ├── asl_digit_test/        
│       ├── letter_dataset.pickle    # extracted letter landmarks
│       └── digit_dataset.pickle     # extracted digit landmarks
│
├── main.py          # live camera app entry point
├── utils.py         # landmark extraction, normalization, drawing helpers
├── handedness.py    # routes left/right hand to correct model
├── postprocess.py   # debounce, autocorrect, word building
├── hud.py           # HUD drawing on camera frame
├── symbols.py       # symbol mode overlay
├── train.py         # training pipeline
├── test_model.py    # evaluate models on unseen images
└── requirements.txt
```

## Features

- [x] Classify 26 Alphabets (A–Z)
- [x] Classify USA Numbers (0–9)
- [x] Auto-route Left hand → numbers, Right hand → letters
- [x] Handle both 2 hands simultaneously
- [ ] Switch between alphabet and number automatically
- [x] Handle del and space gestures
- [ ] Handle lightening 
