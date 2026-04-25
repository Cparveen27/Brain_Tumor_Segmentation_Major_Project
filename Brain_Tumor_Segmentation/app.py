from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt 
from flask import send_from_directory
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os 
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np 
import mysql.connector 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

# Configure upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

# Ensure the upload and output folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations for classification and relevancy
image_transform_classification = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the MobileNet model class (same as the one used during training)
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# Load the trained classification model for tumor/no-tumor detection
classification_model = MobileNetModel(num_classes=4)
classification_model.load_state_dict(torch.load("mobilenet_pred.pt", map_location=device))
classification_model = classification_model.to(device)
classification_model.eval()

# Load the trained relevancy detection model
relevancy_model = MobileNetModel(num_classes=2)
relevancy_model.load_state_dict(torch.load("mobilenet.pt", map_location=device))
relevancy_model = relevancy_model.to(device)
relevancy_model.eval()

# U-Net++ Model Architecture for Segmentation
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, num_classes=2, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output

# Load U-Net++ segmentation model
unet_plus_plus_model = NestedUNet(num_classes=2, input_channels=3)
unet_plus_plus_model.load_state_dict(torch.load('unet_plus_plus_best_model.pt', map_location=device))
unet_plus_plus_model.to(device)
unet_plus_plus_model.eval()

# Transform for U-Net++ segmentation
unet_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Class names for classification
CLASS_NAMES = ['No tumor', 'Pituitary', 'Meningioma', 'Glioma']

def predict_relevance(image):
    image = image_transform_classification(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = relevancy_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

def predict_image(image):
    image = image_transform_classification(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = classification_model(image)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), CLASS_NAMES[predicted.item()]

# U-Net++ Prediction Function
def predict_unet_plus_plus(image_path, confidence_threshold=0.5):
    """Make prediction using U-Net++ model"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Transform image
    image_tensor = unet_transform(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        output = unet_plus_plus_model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]
    
    # Convert to numpy
    prediction = prediction.squeeze().cpu().numpy()
    confidence = confidence.squeeze().cpu().numpy()
    
    # Apply confidence threshold - only keep predictions with high confidence
    prediction_filtered = np.zeros_like(prediction)
    prediction_filtered[confidence >= confidence_threshold] = prediction[confidence >= confidence_threshold]
    
    # Resize back to original size
    prediction_resized = cv2.resize(prediction_filtered.astype(np.uint8), 
                                  original_size, 
                                  interpolation=cv2.INTER_NEAREST)
    
    return prediction_resized, confidence.mean()

def create_overlay(original_image, prediction, alpha=0.6):
    """Create overlay of prediction on original image"""
    # Convert PIL image to numpy if needed
    if isinstance(original_image, Image.Image):
        original_np = np.array(original_image)
    else:
        original_np = original_image
    
    # Ensure original image is in RGB format
    if len(original_np.shape) == 2:  # Grayscale
        original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)
    elif original_np.shape[2] == 4:  # RGBA
        original_np = cv2.cvtColor(original_np, cv2.COLOR_RGBA2RGB)
    
    # Resize prediction to match original image size
    if prediction.shape != original_np.shape[:2]:
        prediction = cv2.resize(prediction, (original_np.shape[1], original_np.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
    
    # Create colored mask (red for tumor with transparency)
    colored_mask = np.zeros_like(original_np)
    colored_mask[prediction == 1] = [255, 0, 0]  # Red color for tumor
    
    # Create overlay by blending original image with colored mask
    overlay = original_np.copy()
    tumor_areas = prediction == 1
    
    # Blend only the tumor areas
    overlay[tumor_areas] = cv2.addWeighted(original_np[tumor_areas], 1 - alpha, 
                                          colored_mask[tumor_areas], alpha, 0)
    
    return overlay

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)
    gender = db.Column(db.Enum('M', 'F', 'O'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in session 
            session['user_email'] = user.email 
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')   # Updated from 'username' to 'name'
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')

        # Validate mobile number
        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be exactly 10 digits.', 'danger')
            return render_template('login.html')

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('login.html')

        # Check if name (username) already exists
        if User.query.filter_by(name=name).first():
            flash('Name is already taken. Please choose a different one.', 'danger')
            return render_template('login.html')

        # Validate password
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('login.html')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('login.html')

        # Hash the password
        hashed_password = bcrypt.generate_password_hash(password)

        # Create a new user instance
        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            age=age,
            gender=gender,
            mobile=mobile
        )

        # Add and commit the new user to the database
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = file.filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            # Open image with PIL for relevancy and classification prediction
            image_pil = Image.open(image_path).convert('RGB')

            # Step 1: Check if the image is relevant
            relevance_prediction = predict_relevance(image_pil)
            result = {}
            result['image_filename'] = 'uploads/' + filename

            if relevance_prediction == 0:  # Irrelevant image
                result['message'] = "The uploaded image is irrelevant. Please upload Brain MRI image."
                return render_template('prediction.html', result=result)
            else:  # Relevant image, proceed with tumor prediction
                # Step 2: Classify the image for tumor presence
                tumor_prediction, confidence_score, class_name = predict_image(image_pil)
                
                # Add classification results to result
                result['classification_confidence'] = f"{confidence_score:.3f}"
                result['class_name'] = class_name
                
                # Check if "No tumor" is detected - STOP here if no tumor
                if class_name == "No tumor":
                    result['message'] = f"No tumor detected in the image. (Confidence: {confidence_score:.3f})"
                    return render_template('prediction.html', result=result)
                
                # If tumor is detected (Glioma, Meningioma, or Pituitary tumor), proceed with segmentation
                else:
                    result['message'] = f"Tumor detected: {class_name} (Confidence: {confidence_score:.3f})"

                    # Perform segmentation using U-Net++
                    predicted_mask, segmentation_confidence = predict_unet_plus_plus(image_path)
                    
                    # Create overlay
                    overlay = create_overlay(image_pil, predicted_mask, alpha=0.6)
                    
                    # Save the predicted mask
                    mask_filename = 'mask_' + filename
                    mask_path = os.path.join(app.config['OUTPUT_FOLDER'], mask_filename)
                    plt.imsave(mask_path, predicted_mask, cmap='gray')
                    
                    # Save the overlay as proper image (not plot)
                    overlay_filename = 'overlay_' + filename
                    overlay_path = os.path.join(app.config['OUTPUT_FOLDER'], overlay_filename)
                    
                    # Convert overlay to PIL and save
                    overlay_pil = Image.fromarray(overlay.astype(np.uint8))
                    overlay_pil.save(overlay_path)

                    result['mask_filename'] = 'outputs/' + mask_filename
                    result['overlay_filename'] = 'outputs/' + overlay_filename
                    result['segmentation_confidence'] = f"{segmentation_confidence:.3f}"
                    
                    # Calculate tumor statistics
                    tumor_pixels = np.sum(predicted_mask == 1)
                    total_pixels = predicted_mask.size
                    tumor_percentage = (tumor_pixels / total_pixels) * 100
                    result['tumor_percentage'] = f"{tumor_percentage:.2f}%"
                    result['tumor_pixels'] = f"{tumor_pixels:,}"
                    result['total_pixels'] = f"{total_pixels:,}" 

                    # Generate PDF with the results
                    pdf_filename = generate_pdf(result)

                    # Step 3: Store the prediction result in the database
                    user_email = session.get('user_email')  # Get user email from session
                    if user_email:  # Ensure the email is present in the session
                        store_prediction_in_db(user_email=user_email,
                                               input_image_filename=result['image_filename'],
                                               mask_filename=result.get('mask_filename'),
                                               overlay_filename=result.get('overlay_filename'),
                                               class_name=result['class_name'])

                    return render_template('prediction.html', result=result, pdf_filename=pdf_filename)
        else:
            flash('Allowed file types are png, jpg, jpeg, gif', 'danger')
            return redirect(request.url)
    else:
        return render_template('prediction.html')  
import time   
def generate_pdf(result):
    # Define the path for saving the PDF
    pdf_filename = 'prediction_result_' + str(int(time.time())) + '.pdf'
    pdf_path = os.path.join(app.config['OUTPUT_FOLDER'], pdf_filename)

    # Create the PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Add title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 750, "NeuroCare AI | MRI Scan Prediction Results")

    # Add Image (Optional)
    try:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], result['image_filename'])  # Join the correct path
        c.setFont("Helvetica", 12)

        # Ensure the image exists at the path
        if os.path.exists(image_path):
            c.drawImage(image_path, 100, 500, width=200, height=200)
        else:
            c.drawString(100, 500, "Image not found at path: " + image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        c.drawString(100, 500, "Error loading image.")

    # Add prediction information
    c.drawString(100, 460, f"Prediction: {result['class_name']}")
    c.drawString(100, 440, f"Confidence: {result['classification_confidence']}%")

    # Add recommendation based on the result
    recommendations = {
        "Glioma": [
            "Consult a neurologist for further evaluation.",
            "MRI follow-up scans may be required to monitor tumor growth.",
            "Consider surgical options for tumor removal."
        ],
        "Meningioma": [
            "Regular MRI scans to monitor the tumor size.",
            "Discuss surgery options with your doctor.",
            "Consider radiation therapy if surgery isn't possible."
        ],
        "Pituitary": [
            "Hormonal testing to assess pituitary function.",
            "Discuss possible surgery for tumor removal.",
            "Follow-up with an endocrinologist is important."
        ],
        "No tumor": [
            "No tumor detected, continue with regular checkups."
        ]
    }

    c.drawString(100, 420, "Recommendations:")
    y_position = 400
    for line in recommendations.get(result['class_name'], []):
        c.drawString(100, y_position, line)
        y_position -= 20

    # Save the PDF
    c.save()

    return pdf_filename


@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    # Send the generated PDF to the user for download
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

    
def store_prediction_in_db(user_email, input_image_filename, mask_filename, overlay_filename, class_name):
    # Establish a connection to the database
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',  
        database='LifeCareDB'
    )

    cursor = connection.cursor()

    # Insert data into the prediction_results table
    insert_query = """
    INSERT INTO prediction_results (user_email, input_image_filename, mask_filename, overlay_filename, class_name)
    VALUES (%s, %s, %s, %s, %s)
    """

    values = (user_email, input_image_filename, mask_filename, overlay_filename, class_name)

    cursor.execute(insert_query, values)

    # Commit the transaction
    connection.commit()

    # Close the cursor and connection
    cursor.close()
    connection.close() 

@app.route('/history_prediction')
def history_prediction():
    # Get user email from the session
    user_email = session.get('user_email')
    
    if not user_email:
        flash('Please log in to view your prediction history.', 'danger')
        return redirect(url_for('login'))  # Redirect to login if the user is not logged in

    # Fetch prediction history from the database
    predictions = get_prediction_history(user_email)
    
    return render_template('history_prediction.html', predictions=predictions)

def get_prediction_history(user_email):
    # Establish a connection to the database
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='', 
        database='LifeCareDB'
    )

    cursor = connection.cursor()

    # Query to fetch the user's prediction history
    query = """
    SELECT id, user_email, input_image_filename, mask_filename, overlay_filename, class_name, created_at
    FROM prediction_results
    WHERE user_email = %s
    ORDER BY created_at DESC
    """

    cursor.execute(query, (user_email,))
    
    # Fetch all the results
    predictions = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    connection.close()

    # Return the results as a list of dictionaries
    return [{'id': prediction[0],
             'user_email': prediction[1],
             'input_image_filename': prediction[2], 
             'mask_filename': prediction[3], 
             'overlay_filename': prediction[4], 
             'class_name': prediction[5], 
             'created_at': prediction[6]} for prediction in predictions] 

@app.route('/accuracy_page')
def accuracy_page():
    # CNN Performance Metrics
    cnn_accuracy = 90.0
    cnn_precision = 91.0  # Precision (Macro Avg)
    cnn_recall = 89.0     # Recall (Macro Avg)
    cnn_f1_score = 90.0   # F1-Score (Macro Avg)

    # MobileNet Performance Metrics
    mobilenet_accuracy = 99.0
    mobilenet_precision = 99.0  # Precision (Macro Avg)
    mobilenet_recall = 99.0     # Recall (Macro Avg)
    mobilenet_f1_score = 99.0   # F1-Score (Macro Avg)

    # U-Net++ Performance Metrics
    unet_accuracy = 99.33
    unet_mean_iou = 80.39
    unet_jaccard_score = 61.46

    return render_template('accuracy_page.html', 
                           cnn_accuracy=cnn_accuracy, cnn_precision=cnn_precision, cnn_recall=cnn_recall, cnn_f1_score=cnn_f1_score,
                           mobilenet_accuracy=mobilenet_accuracy, mobilenet_precision=mobilenet_precision, mobilenet_recall=mobilenet_recall, mobilenet_f1_score=mobilenet_f1_score,
                           unet_accuracy=unet_accuracy, unet_mean_iou=unet_mean_iou, unet_jaccard_score=unet_jaccard_score)

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) 
# # List available models (convert generator to list)
# available_models = list(genai.list_models())

# # Print the available models
# print("Available models:", available_models)
chatbot_model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    system_instruction=(
        "You are a medical AI assistant specialized in brain tumor classification. "
        "You can help analyze MRI scan images and predict the presence of different types of brain tumors, "
        "such as Glioma, Meningioma, and Pituitary tumors. "
        "Ask for relevant details, such as symptoms,  or MRI scans, and provide recommendations for follow-up actions."
    )
)


# Initialize chat session
chat_session = chatbot_model.start_chat(history=[])


# Format chatbot response with indentation and line breaks
def format_museum_response(raw_response):
    lines = raw_response.split('. ')
    formatted_lines = []
    indent = "    "

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('**For'):
            formatted_lines.append(f"\n{line}\n")
        elif line.startswith('* **'):
            formatted_lines.append(f"{indent}{line}\n")
        elif line.startswith('*'):
            formatted_lines.append(f"{indent}{indent}{line}\n")
        else:
            formatted_lines.append(f"{line}.\n")

    return ''.join(formatted_lines).strip()


@app.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.json.get('message', '')
    if user_message:
        # Add user message to chat history
        chat_session.send_message(user_message)
        # Get the assistant's response
        response = chat_session.history[-1].parts[0].text
        formatted_response = format_museum_response(response)
        return jsonify({'response': formatted_response})
    return jsonify({'response': 'I didn\'t understand that. Could you please try again?'})

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)  # Remove user ID from session
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables
    app.run(debug=True)