
# AgriTech Assistant - README

AgriTech Assistant is an innovative agricultural platform designed to empower Indian farmers by providing intelligent solutions for crop yield prediction, plant disease detection, and a chatbot for resolving queries in Indian languages. This README provides an overview of the project and guides you on how to set up and run the application.

---

## Project Overview

### Key Features
1. **Crop Yield Prediction**: Predicts crop yields based on environmental factors, soil type, and agricultural practices.
2. **Plant Disease Detection**: Detects plant diseases by analyzing uploaded images of plant leaves.
3. **Farmer Query Chatbot**: Solves farmers' queries in Indian languages using an AI-powered chatbot.

---

## Directory Structure
```
manavpatel571-AgriTech/
├── main.py
├── app.py                # Main application file to run
├── main2.py
├── models/               # Pre-trained models and encoders
│   ├── plant_disease_model.pt
│   ├── xgboost_model.pkl
│   ├── Soil_Type_encoder.pkl
│   ├── catboost_model.pkl
│   ├── Crop_Type_encoder.pkl
│   └── State_encoder.pkl
├── final_app.py
├── requirements.txt      # Dependencies for the project
├── LICENSE
├── templates/            # HTML templates for the web app
│   ├── index.html
│   ├── home.html
│   ├── index2.html
│   ├── base.html
│   ├── chat_widget.html
│   ├── result2.html
│   └── result.html
├── Procfile              # Configuration for deployment
└── static/               # Static files for the web app
    ├── images/
    └── uploads/
```

---

## How to Run

### Step 1: Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/manavpatel571/AgriTech.git
cd AgriTech
```

### Step 2: Set Up the Environment
Create a virtual environment and install the required dependencies:
```bash
python -m venv env
source env/bin/activate   # For Linux/Mac
env\Scripts\activate     # For Windows
pip install -r requirements.txt
```

### Step 3: Run the Application
Run the `app.py` file to start the application:
```bash
python app.py
```
The application will be accessible at `http://127.0.0.1:5000`.

---

## Services Provided
1. **Crop Yield Prediction**: Navigate to the respective feature and input environmental data for predictions.
2. **Plant Disease Detection**: Upload images of plant leaves to detect diseases.
3. **Farmer Query Chatbot**: Interact with the chatbot to get answers to farming-related questions in Indian languages.

---

## License
This project is licensed under the terms specified in the `LICENSE` file.

---

## Contribution
Contributions are welcome! Feel free to fork the repository and create pull requests.

---

For any issues or queries, please contact us through the support section of the application.
