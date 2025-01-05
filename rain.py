from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load crop yield prediction models and encoders
dt_model = joblib.load('models/dt_model.pkl')
lgbm_model = joblib.load('models/lgbm_model.pkl')
state_encoder = joblib.load('models/State_encoder.pkl')
crop_type_encoder = joblib.load('models/Crop_Type_encoder.pkl')
soil_type_encoder = joblib.load('models/Soil_Type_encoder.pkl')
 
# Get available options for categorical variables
states = state_encoder.classes_
crop_types = crop_type_encoder.classes_
soil_types = soil_type_encoder.classes_

# Chatbot Response Dictionaries
PLANT_RESPONSES = { 
    'चावल': """चावल के प्रमुख फायदे:
1. ऊर्जा का मुख्य स्रोत
2. आसानी से पचने वाला भोजन
3. ग्लूटेन फ्री होता है
4. विटामिन बी से भरपूर
5. हृदय के लिए फायदेमंद
6. रक्तचाप नियंत्रित करने में मदद करता है""",

    'गेहूं': """गेहूं के प्रमुख फायदे:
1. फाइबर से भरपूर
2. प्रोटीन का अच्छा स्रोत
3. एंटीऑक्सीडेंट से भरपूर
4. पाचन में सहायक
5. वजन घटाने में मददगार
6. ऊर्जा प्रदान करता है""",

    'टमाटर': """टमाटर के फायदे:
1. विटामिन सी से भरपूर
2. एंटीऑक्सीडेंट गुणों से युक्त 
3. त्वचा के लिए फायदेमंद
4. आंखों के लिए लाभदायक
5. हृदय रोग से बचाव
6. कैंसर से बचाव में सहायक""",
}

# Disease detection model class and categories
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy',
    'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

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

def feature_engineering(df):
    """
    Simplified feature engineering that matches the training data
    """
    processed = df.copy()
    
    # Keep only the basic features that were used in training
    basic_features = ['Year', 'State', 'Crop_Type', 'Rainfall', 'Soil_Type', 'Irrigation_Area']
    processed = processed[basic_features]
    
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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/yield_prediction')
def yield_prediction():
    return render_template('index.html',
                         states=states,
                         crop_types=crop_types,
                         soil_types=soil_types)

@app.route('/disease_detection')
def disease_detection():
    return render_template('index2.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create input data dictionary
        input_data = {
            'State': request.form['state'],
            'Crop_Type': request.form['crop_type'],
            'Soil_Type': request.form['soil_type'],
            'Year': float(request.form['year']),
            'Rainfall': float(request.form['rainfall']),
            'Irrigation_Area': float(request.form['irrigation_area'])
        }

        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        input_df['State'] = state_encoder.transform([input_data['State']])[0]
        input_df['Crop_Type'] = crop_type_encoder.transform([input_data['Crop_Type']])[0]
        input_df['Soil_Type'] = soil_type_encoder.transform([input_data['Soil_Type']])[0]

        # Apply feature engineering
        input_processed = feature_engineering(input_df)
        
        # Make predictions
        dt_pred = dt_model.predict(input_processed)[0]
        lgbm_pred = lgbm_model.predict(input_processed)[0]
        
        # Combine predictions using the same weights as in training
        final_prediction = 0.9 * dt_pred + 0.108 * lgbm_pred

        return render_template('result.html',
                             prediction=round(final_prediction, 2),
                             input_data=input_data)

    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
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
                    outputs = model(image)
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

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json['message'].strip().lower()
        
        # Check for keywords in the message
        for keyword, response in PLANT_RESPONSES.items():
            if keyword in message:
                return jsonify({'response': response})
        
        # Default response if no matching keywords
        return jsonify({'response': """माफ़ कीजिये, मैं इस सवाल का जवाब नहीं दे सकता। 
कृपया निम्नलिखित फसलों के बारे में पूछें:
1. चावल
2. गेहूं   
3. टमाटर
या अन्य कृषि संबंधित प्रश्न पूछें।"""})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize disease detection model
    model = CustomCNN(num_classes=38)
    model.load_state_dict(torch.load('models/plant_disease_model.pt', map_location=torch.device('cpu')))
    model.eval()
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)