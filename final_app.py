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
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load crop yield prediction models and encoders
try:
    catboost_model = joblib.load('catboost_model.pkl')
    xgboost_model = joblib.load('xgboost_model.pkl')
    state_encoder = joblib.load('State_encoder.pkl')
    crop_type_encoder = joblib.load('Crop_Type_encoder.pkl')
    soil_type_encoder = joblib.load('Soil_Type_encoder.pkl')

    # Get available options for categorical variables
    states = state_encoder.classes_
    crop_types = crop_type_encoder.classes_
    soil_types = soil_type_encoder.classes_
except:
    print("Warning: Some model files couldn't be loaded")

# Agricultural context for SLM
AGRICULTURAL_CONTEXT = """
This is an agricultural chatbot that helps farmers with:
- Crop information and benefits
- Farming techniques and best practices
- Plant diseases and treatments
- Weather-related farming advice
- Soil management techniques
- Irrigation practices
The responses should be helpful and in Hindi or English.
"""

class SimpleSLM:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
            self.context = AGRICULTURAL_CONTEXT
        except Exception as e:
            print(f"Error initializing SLM: {str(e)}")
            self.tokenizer = None
            self.model = None
        
    def generate_response(self, input_text, max_length=150):
        try:
            if not self.tokenizer or not self.model:
                return "Model initialization error. Please try again later."

            full_input = f"{self.context}\nUser: {input_text}\nAssistant:"
            inputs = self.tokenizer.encode(full_input, return_tensors="pt")
            
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.replace(full_input, "").strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Initialize SLM
slm = SimpleSLM()

# Predefined responses dictionary
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

# Disease detection model class
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
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

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

@app.route('/chat_widget')
def chat_widget():
    return render_template('chat_widget.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json['message'].strip()
        
        # Check for predefined responses first
        for keyword, response in PLANT_RESPONSES.items():
            if keyword in message.lower():
                return jsonify({'response': response})
        
        # Use SLM for other queries
        response = slm.generate_response(message)
        
        # If response is empty or too short, use default response
        if len(response) < 10:
            response = """माफ़ कीजिये, मैं इस सवाल का जवाब नहीं दे सकता। 
कृपया निम्नलिखित फसलों के बारे में पूछें:
1. चावल
2. गेहूं
3. टमाटर
या अन्य कृषि संबंधित प्रश्न पूछें।"""
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
        input_data = {
            'State': request.form['state'],
            'Crop_Type': request.form['crop_type'],
            'Soil_Type': request.form['soil_type'],
            'Year': float(request.form['year']),
            'Rainfall': float(request.form['rainfall']),
            'Irrigation_Area': float(request.form['irrigation_area'])
        }

        # Create DataFrame and encode categorical variables
        input_df = pd.DataFrame([input_data])
        input_df['State'] = state_encoder.transform([input_data['State']])[0]
        input_df['Crop_Type'] = crop_type_encoder.transform([input_data['Crop_Type']])[0]
        input_df['Soil_Type'] = soil_type_encoder.transform([input_data['Soil_Type']])[0]

        # Make predictions
        catboost_pred = catboost_model.predict(input_df)[0]
        xgboost_pred = xgboost_model.predict(input_df)[0]
        final_prediction = 0.6 * catboost_pred + 0.4 * xgboost_pred

        return render_template('result.html',
                             prediction=round(final_prediction, 2),
                             input_data=input_data)

    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index2.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index2.html', error='No selected file')
    
    if file and allowed_file(file.filename):
        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            # Process image and make prediction
            image = process_image(filename)
            
            # Load disease detection model
            model = CustomCNN(num_classes=38)
            model.load_state_dict(torch.load('plant_disease_model.pt', map_location=torch.device('cpu')))
            model.eval()
            
            with torch.no_grad():
                outputs = model(image)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                probability = probabilities[predicted].item() * 100
                
                return render_template('result2.html',
                                     filename=file.filename,
                                     disease=predicted.item(),
                                     probability=f"{probability:.2f}%")
                
        except Exception as e:
            return render_template('index2.html', error=str(e))
    
    return render_template('index2.html', error='Invalid file type')

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the Flask application
    app.run(debug=True)