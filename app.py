from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Models and Encoders
try:
    catboost_model = joblib.load('models/catboost_model.pkl')
    xgboost_model = joblib.load('models/xgboost_model.pkl')
    state_encoder = joblib.load('models/State_encoder.pkl')
    crop_type_encoder = joblib.load('models/Crop_Type_encoder.pkl')
    soil_type_encoder = joblib.load('models/Soil_Type_encoder.pkl')
    
    states = state_encoder.classes_
    crop_types = crop_type_encoder.classes_
    soil_types = soil_type_encoder.classes_
except Exception as e:
    print(f"Error loading prediction models: {e}")

# Load Chatbot Model
def load_chatbot_model():
    model_name = "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading chatbot model: {e}")
        return None, None

chatbot_model, chatbot_tokenizer = load_chatbot_model()

# Disease Detection Model
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.res1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU()
            )
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.res2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU()
            )
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.res1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x + self.res2(x)
        x = self.classifier(x)
        return x

# Predefined Chatbot Responses
FARMING_RESPONSES = {
    'फसल': """मुख्य फसलें:
1. खरीफ फसलें: धान, मक्का, ज्वार, बाजरा
2. रबी फसलें: गेहूं, चना, सरसों
3. जायद फसलें: तरबूज, खरबूजा, ककड़ी""",

    'खाद': """खाद के प्रकार:
1. जैविक खाद
2. रासायनिक खाद
3. कम्पोस्ट खाद
सही मात्रा में खाद का प्रयोग करें।""",

    'कीट': """कीट नियंत्रण के उपाय:
1. जैविक कीटनाशक का प्रयोग
2. फसल चक्र अपनाएं
3. समय पर निराई-गुड़ाई
4. रोग प्रतिरोधी किस्में चुनें""",

    'सिंचाई': """सिंचाई के तरीके:
1. फव्वारा सिंचाई
2. बूंद-बूंद सिंचाई
3. नहर सिंचाई
पानी की बचत करें।"""
}

# Utility Functions
def feature_engineering(df):
    processed = df.copy()
    
    numerical_cols = ['Year', 'Rainfall', 'Irrigation_Area']
    
    processed['Year_Rainfall_interaction'] = processed['Year'] * processed['Rainfall']
    processed['Year_Irrigation_Area_interaction'] = processed['Year'] * processed['Irrigation_Area']
    processed['Rainfall_Irrigation_Area_interaction'] = processed['Rainfall'] * processed['Irrigation_Area']
    processed['Year_ratio_to_rainfall'] = processed['Year'] / (processed['Rainfall'] + 1e-5)
    processed['Rainfall_ratio_to_rainfall'] = processed['Rainfall'] / (processed['Rainfall'] + 1e-5)
    processed['Year Rainfall'] = processed['Year'] * processed['Rainfall']
    processed['Year Irrigation_Area'] = processed['Year'] * processed['Irrigation_Area']
    processed['Rainfall Irrigation_Area'] = processed['Rainfall'] * processed['Irrigation_Area']
    processed['Year Rainfall Irrigation_Area'] = processed['Year'] * processed['Rainfall'] * processed['Irrigation_Area']
    processed['Rainfall_log'] = np.log1p(processed['Rainfall'])
    processed['Irrigation_Area_log'] = np.log1p(processed['Irrigation_Area'])
    
    for col in numerical_cols:
        processed[f'State_{col}_mean'] = processed['State']
        processed[f'State_{col}_std'] = processed['State']
        processed[f'State_{col}_max'] = processed['State']
        processed[f'State_{col}_min'] = processed['State']
        processed[f'Crop_Type_{col}_mean'] = processed['Crop_Type']
        processed[f'Crop_Type_{col}_std'] = processed['Crop_Type']
        processed[f'Crop_Type_{col}_max'] = processed['Crop_Type']
        processed[f'Crop_Type_{col}_min'] = processed['Crop_Type']
    
    return processed

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    return image

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/chat_interface')
def chat_interface():
    return render_template('chat_widget.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json['message'].strip()
        
        # Check predefined responses
        for keyword, response in FARMING_RESPONSES.items():
            if keyword in message.lower():
                return jsonify({'response': response})
        
        # Use model for other queries
        if chatbot_model is None or chatbot_tokenizer is None:
            return jsonify({'response': 'माफ़ कीजिये, मॉडल लोड नहीं हो पाया। कृपया बाद में प्रयास करें।'})

        prompt = f"User: {message}\nAssistant:"
        inputs = chatbot_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = chatbot_model.generate(
                inputs.input_ids,
                max_length=200,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=chatbot_tokenizer.pad_token_id,
                eos_token_id=chatbot_tokenizer.eos_token_id,
            )
        
        response = chatbot_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Assistant:")[-1].strip()
        
        if not response:
            response = "माफ़ कीजिये, मैं इस सवाल का जवाब नहीं दे सकता। कृपया कृषि से संबंधित दूसरा प्रश्न पूछें।"
        
        return jsonify({'response': response})
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'response': 'माफ़ कीजिये, कुछ तकनीकी समस्या आ गई। कृपया दोबारा प्रयास करें।'
        }), 500

@app.route('/yield_prediction')
def yield_prediction():
    return render_template('index.html',
                         states=states,
                         crop_types=crop_types,
                         soil_types=soil_types)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'State': request.form['state'],
            'Crop_Type': request.form['crop_type'],
            'Soil_Type': request.form['soil_type'],
            'Year': float(request.form['year']),
            'Rainfall': float(request.form['rainfall']),
            'Irrigation_Area': float(request.form['irrigation_area'])
        }

        input_df = pd.DataFrame([input_data])
        input_df['State'] = state_encoder.transform([input_data['State']])[0]
        input_df['Crop_Type'] = crop_type_encoder.transform([input_data['Crop_Type']])[0]
        input_df['Soil_Type'] = soil_type_encoder.transform([input_data['Soil_Type']])[0]

        input_processed = feature_engineering(input_df)
        
        catboost_pred = catboost_model.predict(input_processed)[0]
        xgboost_pred = xgboost_model.predict(input_processed)[0]
        final_prediction = 0.6 * catboost_pred + 0.4 * xgboost_pred

        return render_template('result.html',
                             prediction=round(final_prediction, 2),
                             input_data=input_data)

    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/disease_detection')
def disease_detection():
    return render_template('index2.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index2.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index2.html', error='No selected file')
    
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        try:
            image = process_image(filename)
            
            with torch.no_grad():
                outputs = disease_model(image)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                probability = probabilities[predicted].item() * 100
                disease = class_names[predicted.item()]
                
                return render_template('result2.html',
                                     filename=file.filename,
                                     disease=disease,
                                     probability=f"{probability:.2f}%")
        except Exception as e:
            return render_template('index2.html', error=str(e))
    
    return render_template('index2.html')

if __name__ == '__main__':
    # Initialize disease detection model
    disease_model = CustomCNN(num_classes=38)
    disease_model.load_state_dict(torch.load('models/plant_disease_model.pt', map_location=torch.device('cpu')))
    disease_model.eval()
    
    app.run(debug=True)