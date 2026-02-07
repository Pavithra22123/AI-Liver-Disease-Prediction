# Add these new imports at the top
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
import numpy as np
import pickle
import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from collections import Counter
from flask_mail import Mail, Message
from threading import Thread


app = Flask(__name__)
app.secret_key = "liver_secret_key_2024"

# ========================================
# CONFIGURATION
# ========================================
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)



app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'pavithra22123@gmail.com'
app.config['MAIL_PASSWORD'] = 'hoib naas ohya rbob'
app.config['MAIL_DEFAULT_SENDER'] = 'pavithra22123@gmail.com'


mail = Mail(app)


IMAGE_SIZE = (128, 128)
DB = "liver.db"

import os
import pickle
from tensorflow.keras.models import load_model

# Base directory for relative paths
BASE_DIR = os.path.dirname(__file__)

# ========================================
# 1. LOAD ANN BLOOD TEST MODEL
# ========================================
try:
    blood_model_path = os.path.join(BASE_DIR, "ann_model.h5")
    blood_model = load_model(blood_model_path)
    print("✓ ANN Blood test model loaded")
except Exception as e:
    print(f"⚠ ANN Blood test model error: {e}")
    blood_model = None

# ========================================
# 2. LOAD SCALER FOR ANN PREPROCESSING
# ========================================
try:
    scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded")
except Exception as e:
    print(f"⚠ Scaler error: {e}")
    scaler = None

# ========================================
# 3. LOAD PREPROCESSING LIMITS
# ========================================
try:
    limits_path = os.path.join(BASE_DIR, "limits.pkl")
    with open(limits_path, "rb") as f:
        limits = pickle.load(f)
    print("✓ Preprocessing limits loaded")
except Exception as e:
    print(f"⚠ Limits error: {e}")
    limits = None

# ========================================
# 4. LOAD IMAGE CLASSIFICATION MODEL
# ========================================
try:
    image_model_path = os.path.join(BASE_DIR, "mobilenet_liver_model.h5")
    image_model = load_model(image_model_path)
    print("✓ Image classification model loaded")
except Exception as e:
    print(f"⚠ Image model error: {e}")
    image_model = None

# ========================================
# MEDICAL REFERENCE DATA
# ========================================
# ========================================
# FIXED MEDICAL REFERENCE DATA
# Replace your existing medical data sections with these
# ========================================

# IMAGE SCAN DATA - Fixed risk levels and severity
# ========================================
# COMPLETE MEDICAL REFERENCE DATA - UPDATED
# Replace lines 50-200 in your app.py with this section
# ========================================

# IMAGE SCAN DATA - Fixed risk levels and severity
CLASS_NAMES = ['f0', 'f1', 'f2', 'f3', 'f4']

STAGE_MEANING = {
    'f0': 'No Fibrosis',
    'f1': 'Mild Fibrosis',
    'f2': 'Moderate Fibrosis',
    'f3': 'Severe Fibrosis',
    'f4': 'Cirrhosis'
}

SEVERITY_LEVEL = {
    'f0': '🟢 No Risk',        # Changed from 'Low'
    'f1': '🟡 Low Risk',       # Changed from 'Low'
    'f2': '🟠 Medium Risk',    # Keep same
    'f3': '🔴 High Risk',      # Keep same
    'f4': '🔴 Critical Risk'   # Changed from 'Critical'
}

MEDICAL_INFO = {
    'f0': {
        'symptoms': ['No noticeable symptoms', 'Normal liver function', 'Healthy metabolism'],
        'diet': ['Balanced diet with fruits and vegetables', 'Avoid alcohol', 'Maintain healthy weight'],
        'doctor': 'Routine health check once per year.',
        'score': 95  # Increased from 90
    },
    'f1': {
        'symptoms': ['Mild fatigue', 'Occasional discomfort', 'Reduced appetite'],
        'diet': ['Low-fat meals', 'High fiber intake', 'Avoid fried foods', 'Avoid alcohol'],
        'doctor': 'Consult hepatologist every 6 months.',
        'score': 75
    },
    'f2': {
        'symptoms': ['Persistent tiredness', 'Upper abdominal pain', 'Mild nausea'],
        'diet': ['Protein-rich diet', 'Reduce sugar', 'Complete alcohol avoidance', 'Avoid processed foods'],
        'doctor': 'Regular monitoring with blood tests every 3 months.',
        'score': 60
    },
    'f3': {
        'symptoms': ['Abdominal swelling', 'Loss of appetite', 'Weight loss', 'Weakness'],
        'diet': ['Low sodium diet', 'Avoid processed foods', 'Medical nutrition therapy', 'Protein monitoring'],
        'doctor': 'Immediate specialist care required. Monthly checkups.',
        'score': 40
    },
    'f4': {
        'symptoms': ['Jaundice', 'Severe abdominal swelling', 'Mental confusion', 'Bleeding risk'],
        'diet': ['Strict medical diet', 'Fluid restriction', 'Hospital-supervised nutrition', 'Low sodium'],
        'doctor': 'Urgent hospitalization required. Consider liver transplant evaluation.',
        'score': 20
    }
}

# ========================================
# BLOOD TEST DATA - Fixed risk levels
# ========================================
BLOOD_STAGE_INFO = {
    "No Liver Disease": {
        "risk": "🟢 No Risk",  # Changed from "Low Risk"
        "health_score_range": "90-100",
        "stage": "Normal",
        "symptoms": [
            "No noticeable symptoms",
            "Normal energy levels",
            "Healthy metabolism",
            "Good appetite",
            "No abdominal discomfort"
        ],
        "diet": [
            "Balanced diet with fruits and vegetables",
            "Lean proteins (chicken, fish, legumes)",
            "Whole grains and fiber",
            "Stay hydrated (8-10 glasses water/day)",
            "Avoid excessive alcohol",
            "Limit processed foods"
        ],
        "doctor": "Continue healthy lifestyle. Annual checkup recommended."
    },
    "Fatty Liver Disease": {
        "risk": "🟡 Low Risk",  # Changed from "Moderate Risk"
        "health_score_range": "70-85",
        "stage": "Mild",
        "symptoms": [
            "Mild fatigue or tiredness",
            "Occasional upper right abdominal discomfort",
            "Slight weight gain",
            "Reduced appetite occasionally",
            "Mild bloating"
        ],
        "diet": [
            "Low-fat, high-fiber diet",
            "Avoid fried and processed foods",
            "Increase vegetables and fruits",
            "Limit sugar and refined carbohydrates",
            "Choose lean proteins",
            "Avoid alcohol completely",
            "Green tea and coffee (moderate amounts)"
        ],
        "doctor": "Consult hepatologist every 6 months. Weight management recommended."
    },
    "Hepatitis": {
        "risk": "🟠 Moderate Risk",  # Changed from "High Risk"
        "health_score_range": "50-70",
        "stage": "Moderate",
        "symptoms": [
            "Persistent fatigue and weakness",
            "Loss of appetite",
            "Nausea and vomiting",
            "Mild jaundice (yellowing of eyes/skin)",
            "Dark urine",
            "Upper abdominal pain",
            "Low-grade fever"
        ],
        "diet": [
            "High-calorie, high-protein diet",
            "Small, frequent meals (5-6 times/day)",
            "Avoid alcohol completely",
            "Limit salt intake",
            "Fresh fruits and vegetables",
            "Avoid raw or undercooked foods",
            "Stay well hydrated"
        ],
        "doctor": "Immediate medical attention required. Regular monitoring with blood tests and ultrasound."
    },
    "Cirrhosis": {
        "risk": "🔴 High Risk",  # Changed from "Critical Risk"
        "health_score_range": "30-50",  # Adjusted range
        "stage": "Severe",
        "symptoms": [
            "Severe fatigue and weakness",
            "Significant weight loss",
            "Jaundice (yellow skin and eyes)",
            "Abdominal swelling (ascites)",
            "Swelling in legs and ankles",
            "Easy bruising and bleeding",
            "Confusion or difficulty concentrating",
            "Itchy skin"
        ],
        "diet": [
            "Strict low-sodium diet (less than 2g/day)",
            "Protein-controlled diet (consult nutritionist)",
            "Small, frequent meals",
            "Fluid restriction if advised",
            "Avoid alcohol completely",
            "Soft, easy-to-digest foods",
            "Vitamin and mineral supplements as prescribed"
        ],
        "doctor": "Urgent hospitalization may be required. Regular monitoring by hepatologist. Consider liver transplant evaluation."
    },
    "Severe Liver Inflammation": {
        "risk": "🔴 Critical Risk",
        "health_score_range": "20-40",  # Adjusted range
        "stage": "Severe",
        "symptoms": [
            "Extreme fatigue",
            "Severe abdominal pain",
            "High fever",
            "Severe nausea and vomiting",
            "Mental confusion",
            "Bleeding tendency",
            "Rapid heartbeat",
            "Breathing difficulty"
        ],
        "diet": [
            "Hospital-supervised nutrition",
            "IV fluids if unable to eat",
            "Strict medical diet as prescribed",
            "High-calorie liquid supplements",
            "Complete alcohol avoidance",
            "Low-fat, easily digestible foods"
        ],
        "doctor": "EMERGENCY: Immediate hospitalization required. ICU monitoring may be needed."
    },
    "Advanced Liver Disease": {
        "risk": "🔴 Critical Risk",
        "health_score_range": "10-30",
        "stage": "Advanced/End-Stage",
        "symptoms": [
            "Severe jaundice",
            "Massive abdominal swelling",
            "Mental confusion (hepatic encephalopathy)",
            "Bleeding from esophagus or stomach",
            "Kidney failure symptoms",
            "Extreme weakness",
            "Loss of consciousness"
        ],
        "diet": [
            "Strict hospital-supervised diet",
            "Protein restriction",
            "Severe sodium restriction",
            "Fluid restriction",
            "Soft, pureed foods only",
            "Nutritional supplements as prescribed"
        ],
        "doctor": "CRITICAL: Immediate ICU admission. Liver transplant evaluation urgently needed."
    }
}

# ====================================================================================
# FIX 2: ENHANCED LANGUAGE SUPPORT
# Update your existing TRANSLATIONS with more comprehensive coverage
# ====================================================================================

TRANSLATIONS = {
    'en': {
        # Navigation
        'app_name': 'LiverAI',
        'home': 'Home',
        'login': 'Login',
        'register': 'Register',
        'logout': 'Logout',
        'dashboard': 'Dashboard',
        'predict': 'New Prediction',
        'history': 'My History',
        'settings': 'Settings',
        'contact': 'Contact Us',
        'about': 'About',
        'faq': 'FAQ',
        'privacy': 'Privacy Policy',
        'terms': 'Terms & Conditions',
        
        # Main content
        'welcome': 'Welcome',
        'hero_title': 'AI-Powered Liver Disease Prediction System',
        'hero_description': 'Advanced machine learning technology for early detection and risk assessment of liver diseases. Get instant predictions using blood test analysis or medical imaging.',
        
        # Features
        'blood_test': 'Blood Test Analysis',
        'blood_test_desc': 'ML-powered analysis of clinical parameters',
        'image_scan': 'Image Scanning',
        'image_scan_desc': 'Deep learning for fibrosis detection',
        'risk_assessment': 'Risk Assessment',
        'risk_assessment_desc': 'Comprehensive health score reports',
        
        # Stats
        'blood_accuracy': 'Blood Test Accuracy',
        'image_accuracy': 'Image Scan Accuracy',
        'prediction_methods': 'Prediction Methods',
        
        # Actions
        'start_prediction': 'Start Prediction',
        'view_history': 'View History',
        'login_to_start': 'Login to Start',
        'create_account': 'Create Account',
        'submit': 'Submit',
        'cancel': 'Cancel',
        'view': 'View',
        'delete': 'Delete',
        'download': 'Download Report',
        'book_appointment': 'Book Doctor Appointment',
        
        # Results
        'results': 'Results',
        'disease': 'Disease',
        'risk': 'Risk Level',
        'health_score': 'Health Score',
        'stage': 'Stage',
        'symptoms': 'Symptoms',
        'diet': 'Diet Recommendations',
        'doctor_advice': 'Doctor Advice',
        
        # Risk levels
        'no_risk': 'No Risk',
        'low_risk': 'Low Risk',
        'medium_risk': 'Medium Risk',
        'moderate_risk': 'Moderate Risk',
        'high_risk': 'High Risk',
        'critical_risk': 'Critical Risk',
        
        # Footer
        'all_rights_reserved': 'All Rights Reserved',
        
        # Chatbot
        'chat_assistant': 'Chat Assistant',
        'type_message': 'Type your message...',
        
        # Appointments
        'my_appointments': 'My Appointments',
        'appointment_date': 'Appointment Date',
        'appointment_time': 'Appointment Time',
        'doctor_specialty': 'Doctor Specialty',
        'appointment_status': 'Status',
    },
    
    'hi': {  # Hindi
        'app_name': 'लिवरएआई',
        'home': 'होम',
        'login': 'लॉगिन',
        'register': 'रजिस्टर',
        'logout': 'लॉगआउट',
        'dashboard': 'डैशबोर्ड',
        'predict': 'नई भविष्यवाणी',
        'history': 'मेरा इतिहास',
        'settings': 'सेटिंग्स',
        'contact': 'संपर्क करें',
        'about': 'हमारे बारे में',
        'faq': 'सामान्य प्रश्न',
        'privacy': 'गोपनीयता नीति',
        'terms': 'नियम और शर्तें',
        
        'welcome': 'स्वागत है',
        'hero_title': 'AI-संचालित यकृत रोग भविष्यवाणी प्रणाली',
        'hero_description': 'यकृत रोगों का शीघ्र पता लगाने और जोखिम मूल्यांकन के लिए उन्नत मशीन लर्निंग तकनीक। रक्त परीक्षण विश्लेषण या मेडिकल इमेजिंग का उपयोग करके त्वरित भविष्यवाणी प्राप्त करें।',
        
        'blood_test': 'रक्त परीक्षण विश्लेषण',
        'blood_test_desc': 'नैदानिक मापदंडों का ML-संचालित विश्लेषण',
        'image_scan': 'छवि स्कैनिंग',
        'image_scan_desc': 'फाइब्रोसिस का पता लगाने के लिए डीप लर्निंग',
        'risk_assessment': 'जोखिम मूल्यांकन',
        'risk_assessment_desc': 'व्यापक स्वास्थ्य स्कोर रिपोर्ट',
        
        'blood_accuracy': 'रक्त परीक्षण सटीकता',
        'image_accuracy': 'छवि स्कैन सटीकता',
        'prediction_methods': 'भविष्यवाणी विधियां',
        
        'start_prediction': 'भविष्यवाणी शुरू करें',
        'view_history': 'इतिहास देखें',
        'login_to_start': 'शुरू करने के लिए लॉगिन करें',
        'create_account': 'खाता बनाएं',
        'submit': 'जमा करें',
        'cancel': 'रद्द करें',
        'view': 'देखें',
        'delete': 'हटाएं',
        'download': 'रिपोर्ट डाउनलोड करें',
        'book_appointment': 'डॉक्टर अपॉइंटमेंट बुक करें',
        
        'results': 'परिणाम',
        'disease': 'रोग',
        'risk': 'जोखिम स्तर',
        'health_score': 'स्वास्थ्य स्कोर',
        'stage': 'चरण',
        'symptoms': 'लक्षण',
        'diet': 'आहार सिफारिशें',
        'doctor_advice': 'डॉक्टर की सलाह',
        
        'no_risk': 'कोई जोखिम नहीं',
        'low_risk': 'कम जोखिम',
        'medium_risk': 'मध्यम जोखिम',
        'moderate_risk': 'मध्यम जोखिम',
        'high_risk': 'उच्च जोखिम',
        'critical_risk': 'गंभीर जोखिम',
        
        'all_rights_reserved': 'सर्वाधिकार सुरक्षित',
        'chat_assistant': 'चैट सहायक',
        'type_message': 'अपना संदेश टाइप करें...',
        
        'my_appointments': 'मेरी अपॉइंटमेंट्स',
        'appointment_date': 'अपॉइंटमेंट तिथि',
        'appointment_time': 'अपॉइंटमेंट समय',
        'doctor_specialty': 'डॉक्टर विशेषता',
        'appointment_status': 'स्थिति',
    },
    
    'ta': {  # Tamil
        'app_name': 'லிவர்AI',
        'home': 'முகப்பு',
        'login': 'உள்நுழைய',
        'register': 'பதிவு செய்ய',
        'logout': 'வெளியேறு',
        'dashboard': 'டாஷ்போர்டு',
        'predict': 'புதிய கணிப்பு',
        'history': 'எனது வரலாறு',
        'settings': 'அமைப்புகள்',
        'contact': 'தொடர்பு கொள்ள',
        'about': 'எங்களை பற்றி',
        'faq': 'அடிக்கடி கேட்கப்படும் கேள்விகள்',
        'privacy': 'தனியுரிமை கொள்கை',
        'terms': 'விதிமுறைகள் மற்றும் நிபந்தனைகள்',
        
        'welcome': 'வரவேற்கிறோம்',
        'hero_title': 'AI-இயங்கும் கல்லீரல் நோய் கணிப்பு அமைப்பு',
        'hero_description': 'கல்லீரல் நோய்களின் ஆரம்ப கண்டறிதல் மற்றும் ஆபத்து மதிப்பீட்டுக்கான மேம்பட்ட இயந்திர கற்றல் தொழில்நுட்பம்.',
        
        'blood_test': 'இரத்த பரிசோதனை பகுப்பாய்வு',
        'blood_test_desc': 'மருத்துவ அளவுருக்களின் ML-இயங்கும் பகுப்பாய்வு',
        'image_scan': 'படம் ஸ்கேனிங்',
        'image_scan_desc': 'ஃபைப்ரோசிஸ் கண்டறிதலுக்கான ஆழமான கற்றல்',
        'risk_assessment': 'ஆபத்து மதிப்பீடு',
        'risk_assessment_desc': 'விரிவான உடல்நலம் மதிப்பெண் அறிக்கைகள்',
        
        'blood_accuracy': 'இரத்த பரிசோதனை துல்லியம்',
        'image_accuracy': 'படம் ஸ்கேன் துல்லியம்',
        'prediction_methods': 'கணிப்பு முறைகள்',
        
        'start_prediction': 'கணிப்பு தொடங்கு',
        'view_history': 'வரலாறு பார்க்க',
        'login_to_start': 'தொடங்க உள்நுழைய',
        'create_account': 'கணக்கு உருவாக்கு',
        'submit': 'சமர்ப்பிக்கவும்',
        'cancel': 'ரத்து செய்',
        'view': 'பார்க்க',
        'delete': 'நீக்கு',
        'download': 'அறிக்கை பதிவிறக்க',
        'book_appointment': 'மருத்துவர் சந்திப்பு பதிவு செய்க',
        
        'results': 'முடிவுகள்',
        'disease': 'நோய்',
        'risk': 'ஆபத்து நிலை',
        'health_score': 'உடல்நலம் மதிப்பெண்',
        'stage': 'நிலை',
        'symptoms': 'அறிகுறிகள்',
        'diet': 'உணவு பரிந்துரைகள்',
        'doctor_advice': 'மருத்துவர் ஆலோசனை',
        
        'no_risk': 'ஆபத்து இல்லை',
        'low_risk': 'குறைந்த ஆபத்து',
        'medium_risk': 'நடுத்தர ஆபத்து',
        'moderate_risk': 'மிதமான ஆபத்து',
        'high_risk': 'அதிக ஆபத்து',
        'critical_risk': 'முக்கியமான ஆபத்து',
        
        'all_rights_reserved': 'அனைத்து உரிமைகளும் பாதுகாக்கப்பட்டவை',
        'chat_assistant': 'அரட்டை உதவியாளர்',
        'type_message': 'உங்கள் செய்தியை தட்டச்சு செய்யவும்...',
        
        'my_appointments': 'எனது சந்திப்புகள்',
        'appointment_date': 'சந்திப்பு தேதி',
        'appointment_time': 'சந்திப்பு நேரம்',
        'doctor_specialty': 'மருத்துவர் சிறப்பு',
        'appointment_status': 'நிலை',
    },
    
    'te': {  # Telugu
        'app_name': 'లివర్AI',
        'home': 'హోమ్',
        'login': 'లాగిన్',
        'register': 'రిజిస్టర్',
        'logout': 'లాగౌట్',
        'dashboard': 'డాష్‌బోర్డ్',
        'predict': 'కొత్త అంచనా',
        'history': 'నా చరిత్ర',
        'settings': 'సెట్టింగ్‌లు',
        'contact': 'సంప్రదించండి',
        'about': 'మా గురించి',
        'faq': 'తరచుగా అడిగే ప్రశ్నలు',
        'privacy': 'గోప్యతా విధానం',
        'terms': 'నిబంధనలు మరియు షరతులు',
        
        'welcome': 'స్వాగతం',
        'hero_title': 'AI-శక్తితో కాలేయ వ్యాధి అంచనా వ్యవస్థ',
        'hero_description': 'కాలేయ వ్యాధుల ప్రారంభ గుర్తింపు మరియు ప్రమాద అంచనా కోసం అధునాతన మెషిన్ లెర్నింగ్ టెక్నాలజీ.',
        
        'blood_test': 'రక్త పరీక్ష విశ్లేషణ',
        'blood_test_desc': 'క్లినికల్ పారామితుల ML-శక్తితో విశ్లేషణ',
        'image_scan': 'చిత్ర స్కానింగ్',
        'image_scan_desc': 'ఫైబ్రోసిస్ గుర్తింపు కోసం డీప్ లెర్నింగ్',
        'risk_assessment': 'ప్రమాద అంచనా',
        'risk_assessment_desc': 'సమగ్ర ఆరోగ్య స్కోరు నివేదికలు',
        
        'blood_accuracy': 'రక్త పరీక్ష ఖచ్చితత్వం',
        'image_accuracy': 'చిత్ర స్కాన్ ఖచ్చితత్వం',
        'prediction_methods': 'అంచనా పద్ధతులు',
        
        'start_prediction': 'అంచనా ప్రారంభించండి',
        'view_history': 'చరిత్ర చూడండి',
        'login_to_start': 'ప్రారంభించడానికి లాగిన్ చేయండి',
        'create_account': 'ఖాతా సృష్టించండి',
        'submit': 'సమర్పించు',
        'cancel': 'రద్దు చేయి',
        'view': 'చూడండి',
        'delete': 'తొలగించు',
        'download': 'నివేదిక డౌన్‌లోడ్',
        'book_appointment': 'డాక్టర్ అపాయింట్‌మెంట్ బుక్ చేయండి',
        
        'results': 'ఫలితాలు',
        'disease': 'వ్యాధి',
        'risk': 'ప్రమాద స్థాయి',
        'health_score': 'ఆరోగ్య స్కోరు',
        'stage': 'దశ',
        'symptoms': 'లక్షణాలు',
        'diet': 'ఆహార సిఫార్సులు',
        'doctor_advice': 'వైద్యుల సలహా',
        
        'no_risk': 'ప్రమాదం లేదు',
        'low_risk': 'తక్కువ ప్రమాదం',
        'medium_risk': 'మధ్యస్థ ప్రమాదం',
        'moderate_risk': 'మధ్యస్థ ప్రమాదం',
        'high_risk': 'అధిక ప్రమాదం',
        'critical_risk': 'క్లిష్టమైన ప్రమాదం',
        
        'all_rights_reserved': 'అన్ని హక్కులు రిజర్వ్ చేయబడ్డాయి',
        'chat_assistant': 'చాట్ అసిస్టెంట్',
        'type_message': 'మీ సందేశాన్ని టైప్ చేయండి...',
        
        'my_appointments': 'నా అపాయింట్‌మెంట్‌లు',
        'appointment_date': 'అపాయింట్‌మెంట్ తేదీ',
        'appointment_time': 'అపాయింట్‌మెంట్ సమయం',
        'doctor_specialty': 'డాక్టర్ స్పెషాలిటీ',
        'appointment_status': 'స్థితి',
    }
}

# ====================================================================================
# UPDATE CONTEXT PROCESSOR - Make translations available in all templates
# ====================================================================================

@app.context_processor
def inject_language():
    """Inject language and translation function into all templates"""
    lang = session.get('language', 'en')
    return {
        'current_lang': lang,
        'get_text': lambda key: get_text(key, lang),
        'available_languages': {
            'en': 'English',
            'hi': 'हिन्दी (Hindi)',
            'ta': 'தமிழ் (Tamil)',
            'te': 'తెలుగు (Telugu)'
        }
    }

# Helper function for translations
def get_text(key, lang='en'):
    """Get translated text for given key"""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

# ====================================================================================
# LANGUAGE SELECTION ROUTE
# ====================================================================================

@app.route("/set_language/<lang>")
def set_language(lang):
    """Set user's preferred language"""
    if lang in TRANSLATIONS:
        session['language'] = lang
    return redirect(request.referrer or url_for('index'))

# ========================================
# 5. USAGE IN TEMPLATES
# ========================================
# In any HTML template, add language switcher in header:
"""
<div class="language-switcher">
    <select onchange="window.location.href='/set_language/' + this.value">
        {% for code, name in available_languages.items() %}
            <option value="{{ code }}" {% if current_lang == code %}selected{% endif %}>
                {{ name }}
            </option>
        {% endfor %}
    </select>
</div>

<!-- Use translations like this: -->
<h1>{{ get_text('welcome') }}</h1>
<a href="#">{{ get_text('home') }}</a>
<button>{{ get_text('submit') }}</button>
"""

# ========================================
# DOCTOR APPOINTMENT BOOKING SYSTEM
# Add to app.py
# ========================================

# 1. ADD APPOINTMENTS TABLE TO DATABASE
def init_db():
    # ... existing tables ...
    
    # Add this new table
    with get_db() as con:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                record_id INTEGER,
                patient_name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                disease TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                preferred_date TEXT,
                preferred_time TEXT,
                doctor_specialty TEXT,
                message TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (record_id) REFERENCES records(id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                type TEXT NOT NULL,
                appointment_id INTEGER,
                read INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (appointment_id) REFERENCES appointments(id)
            )
        """)
        con.commit()

# 2. HELPER FUNCTION TO CHECK IF APPOINTMENT NEEDED
def needs_appointment(risk_level, disease):
    """Check if user needs doctor appointment based on risk"""
    risk_lower = risk_level.lower() if risk_level else ""
    disease_lower = disease.lower() if disease else ""
    
    # No appointment for No Risk and Low Risk
    if 'no risk' in risk_lower or 'low risk' in risk_lower:
        return False
    
    # No appointment for No Fibrosis and No Liver Disease
    if disease in ['No Fibrosis', 'No Liver Disease', 'Mild Fibrosis', 'Fatty Liver Disease']:
        return False
    
    # Appointment needed for Moderate, High, Critical
    if any(word in risk_lower for word in ['moderate', 'medium', 'high', 'critical']):
        return True
    
    return False


# ========================================
# ADD NOTIFICATION SYSTEM
# ========================================

def get_user_notifications(user_id):
    """Get unread notifications for user"""
    with get_db() as con:
        notifications = con.execute("""
            SELECT n.*, a.status as appointment_status, a.preferred_date, a.doctor_specialty
            FROM notifications n
            LEFT JOIN appointments a ON n.appointment_id = a.id
            WHERE n.user_id = ? AND n.read = 0
            ORDER BY n.created_at DESC
        """, (user_id,)).fetchall()
    return notifications

def create_notification(user_id, message, notification_type, appointment_id=None):
    """Create a new notification"""
    with get_db() as con:
        con.execute("""
            INSERT INTO notifications (user_id, message, type, appointment_id, read)
            VALUES (?, ?, ?, ?, 0)
        """, (user_id, message, notification_type, appointment_id))
        con.commit()

# Add these imports at the top of your existing imports
from datetime import date

# ========================================
# ENHANCED APPOINTMENT BOOKING WITH NOTIFICATIONS
# ========================================

@app.route("/book_appointment/<int:record_id>", methods=["GET", "POST"])
def book_appointment(record_id):
    """Book doctor appointment with email and notification"""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    # Get record details
    with get_db() as con:
        record = con.execute("""
            SELECT r.*, u.name, u.email, u.phone 
            FROM records r 
            JOIN users u ON r.user_id = u.id 
            WHERE r.id = ? AND r.user_id = ?
        """, (record_id, session["user_id"])).fetchone()
        
        if not record:
            return "Record not found", 404
        
        # Check if appointment is needed
        if not needs_appointment(record['risk'], record['disease']):
            return render_template("error.html", 
                message="Appointment booking is only available for moderate to critical risk cases.")
    
    if request.method == "POST":
        try:
            preferred_date = request.form.get("preferred_date")
            preferred_time = request.form.get("preferred_time")
            doctor_specialty = request.form.get("doctor_specialty")
            message = request.form.get("message", "")
            phone = request.form.get("phone")
            
            # Validate date is not in past
            appointment_date = datetime.strptime(preferred_date, "%Y-%m-%d")
            if appointment_date.date() < datetime.now().date():
                return render_template("book_appointment.html", 
                    record=record,
                    error="Please select a future date for your appointment.")
            
            with get_db() as con:
                cur = con.cursor()
                cur.execute("""
                    INSERT INTO appointments (
                        user_id, record_id, patient_name, email, phone,
                        disease, risk_level, preferred_date, preferred_time,
                        doctor_specialty, message, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """, (
                    session["user_id"], record_id, record['name'], record['email'],
                    phone, record['disease'], record['risk'],
                    preferred_date, preferred_time, doctor_specialty, message
                ))
                appointment_id = cur.lastrowid
                con.commit()
            
            # Create notification for user
            notification_msg = f"Your appointment request for {preferred_date} at {preferred_time} has been submitted and is pending confirmation."
            create_notification(session["user_id"], notification_msg, "appointment_created", appointment_id)
            
            # Send confirmation email
            email_subject = f"🏥 Appointment Request Submitted - {preferred_date}"
            email_html = f"""
            <div style="font-family: Arial; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center;">
                    <h1>📅 Appointment Request Received</h1>
                </div>
                
                <div style="padding: 30px; background: #f9fafb;">
                    <h2>Dear {record['name']},</h2>
                    <p>Your appointment request has been successfully submitted.</p>
                    
                    <div style="background: white; padding: 20px; border-left: 4px solid #667eea; margin: 20px 0;">
                        <h3>Appointment Details</h3>
                        <p><strong>Date:</strong> {preferred_date}</p>
                        <p><strong>Time:</strong> {preferred_time}</p>
                        <p><strong>Doctor:</strong> {doctor_specialty}</p>
                        <p><strong>Disease:</strong> {record['disease']}</p>
                        <p><strong>Risk Level:</strong> {record['risk']}</p>
                        <p><strong>Status:</strong> <span style="color: #f59e0b;">⏳ Pending Confirmation</span></p>
                    </div>
                    
                    <div style="background: #fef3c7; padding: 15px; border-left: 4px solid #f59e0b; margin: 20px 0;">
                        <strong>⏳ What's Next?</strong><br>
                        Our medical staff will review your request and confirm your appointment within 24-48 hours. 
                        You will receive an email notification once confirmed.
                    </div>
                </div>
                
                <div style="text-align: center; padding: 20px; color: #6b7280; font-size: 14px;">
                    <p>© 2024 LiverAI - Automated Appointment System</p>
                </div>
            </div>
            """
            send_email(email_subject, record['email'], email_html)
            
            return render_template("appointment_success.html", 
                appointment_date=preferred_date,
                appointment_time=preferred_time,
                doctor_specialty=doctor_specialty,
                status="pending"
            )
        except Exception as e:
            return render_template("book_appointment.html", 
                record=record, 
                error=f"Error: {str(e)}")
    
    # GET request - show form
    if record['risk'] and 'critical' in record['risk'].lower():
        suggested_specialty = "Hepatologist (Liver Specialist) - URGENT"
    elif record['disease'] in ['Cirrhosis', 'Advanced Liver Disease']:
        suggested_specialty = "Hepatologist (Liver Specialist)"
    elif record['disease'] in ['Hepatitis', 'Severe Liver Inflammation']:
        suggested_specialty = "Gastroenterologist / Hepatologist"
    else:
        suggested_specialty = "Gastroenterologist"
    
    return render_template("book_appointment.html", 
        record=record,
        suggested_specialty=suggested_specialty
    )

# ========================================
# ADMIN UPDATE APPOINTMENT WITH NOTIFICATIONS
# ========================================

@app.route("/update_appointment/<int:appointment_id>/<status>")
def update_appointment(appointment_id, status):
    """Admin updates appointment status with email notification"""
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    
    valid_statuses = ['pending', 'confirmed', 'completed', 'cancelled']
    if status not in valid_statuses:
        return "Invalid status", 400
    
    with get_db() as con:
        # Get appointment details
        appointment = con.execute("""
            SELECT a.*, u.name, u.email
            FROM appointments a
            JOIN users u ON a.user_id = u.id
            WHERE a.id = ?
        """, (appointment_id,)).fetchone()
        
        if not appointment:
            return "Appointment not found", 404
        
        # Update status
        con.execute("""
            UPDATE appointments 
            SET status = ? 
            WHERE id = ?
        """, (status, appointment_id))
        con.commit()
        
        # Create notification
        status_messages = {
            'confirmed': f"✅ Your appointment for {appointment['preferred_date']} at {appointment['preferred_time']} has been CONFIRMED!",
            'cancelled': f"❌ Your appointment for {appointment['preferred_date']} at {appointment['preferred_time']} has been CANCELLED.",
            'completed': f"✅ Your appointment for {appointment['preferred_date']} has been marked as COMPLETED.",
        }
        
        if status in status_messages:
            create_notification(
                appointment['user_id'], 
                status_messages[status], 
                f"appointment_{status}", 
                appointment_id
            )
            
            # Send email notification
            status_colors = {
                'confirmed': '#22c55e',
                'cancelled': '#ef4444',
                'completed': '#3b82f6'
            }
            
            status_emojis = {
                'confirmed': '✅',
                'cancelled': '❌',
                'completed': '✅'
            }
            
            email_subject = f"{status_emojis.get(status, '')} Appointment {status.title()} - {appointment['preferred_date']}"
            email_html = f"""
            <div style="font-family: Arial; max-width: 600px; margin: 0 auto;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center;">
                    <h1>{status_emojis.get(status, '')} Appointment {status.title()}</h1>
                </div>
                
                <div style="padding: 30px; background: #f9fafb;">
                    <h2>Dear {appointment['name']},</h2>
                    <p>Your appointment status has been updated.</p>
                    
                    <div style="background: white; padding: 20px; border-left: 4px solid {status_colors.get(status, '#667eea')}; margin: 20px 0;">
                        <h3>Appointment Details</h3>
                        <p><strong>Date:</strong> {appointment['preferred_date']}</p>
                        <p><strong>Time:</strong> {appointment['preferred_time']}</p>
                        <p><strong>Doctor:</strong> {appointment['doctor_specialty']}</p>
                        <p><strong>Status:</strong> <span style="color: {status_colors.get(status, '#667eea')};">{status.upper()}</span></p>
                    </div>
                    
                    {'<div style="background: #dcfce7; padding: 15px; border-left: 4px solid #22c55e; margin: 20px 0;"><strong>✅ Confirmed!</strong><br>Please arrive 15 minutes before your scheduled time. Bring your medical records and test results.</div>' if status == 'confirmed' else ''}
                    
                    {'<div style="background: #fee2e2; padding: 15px; border-left: 4px solid #ef4444; margin: 20px 0;"><strong>❌ Cancelled</strong><br>If you need to reschedule, please book a new appointment through your account.</div>' if status == 'cancelled' else ''}
                </div>
                
                <div style="text-align: center; padding: 20px; color: #6b7280; font-size: 14px;">
                    <p>© 2024 LiverAI - Automated Appointment System</p>
                </div>
            </div>
            """
            send_email(email_subject, appointment['email'], email_html)
    
    return redirect(url_for("admin_appointments"))

# ========================================
# USER NOTIFICATIONS ENDPOINT
# ========================================

@app.route("/notifications")
def notifications():
    """View all notifications"""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    with get_db() as con:
        notifications = con.execute("""
            SELECT n.*, a.preferred_date, a.preferred_time, a.status as appointment_status
            FROM notifications n
            LEFT JOIN appointments a ON n.appointment_id = a.id
            WHERE n.user_id = ?
            ORDER BY n.created_at DESC
        """, (session["user_id"],)).fetchall()
    
    return render_template("notifications.html", notifications=notifications)

@app.route("/mark_notification_read/<int:notification_id>")
def mark_notification_read(notification_id):
    """Mark notification as read"""
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    with get_db() as con:
        con.execute("""
            UPDATE notifications 
            SET read = 1 
            WHERE id = ? AND user_id = ?
        """, (notification_id, session["user_id"]))
        con.commit()
    
    return jsonify({"success": True})

@app.route("/get_unread_notifications")
def get_unread_notifications():
    """API endpoint for unread notifications count"""
    if "user_id" not in session:
        return jsonify({"count": 0})
    
    with get_db() as con:
        count = con.execute("""
            SELECT COUNT(*) as count 
            FROM notifications 
            WHERE user_id = ? AND read = 0
        """, (session["user_id"],)).fetchone()["count"]
    
    return jsonify({"count": count})

# ========================================
# CONTEXT PROCESSOR FOR NOTIFICATIONS
# ========================================

@app.context_processor
def inject_notifications():
    """Inject notifications into all templates"""
    if "user_id" in session:
        notifications = get_user_notifications(session["user_id"])
        unread_count = len(notifications)
        return {
            'notifications': notifications,
            'unread_notifications_count': unread_count
        }
    return {
        'notifications': [],
        'unread_notifications_count': 0
    }

# ========================================
# 🔧 FIX #2: CHATBOT API WITH PROPER RESPONSES
# ========================================


# ========================================
# 🔧 FIXED COMPREHENSIVE CHATBOT
# ========================================

@app.route('/chatbot')
def chatbot():
    """Separate chatbot page"""
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chatbot messages"""
    from flask import request, jsonify
    
    data = request.get_json()
    user_message = data.get('message', '').lower()
    
    # Get bot response
    response = get_bot_response(user_message)
    
    return jsonify({'response': response})

def get_bot_response(message):
    """
    Comprehensive bot response system
    """
    msg = message.lower().strip()
    
    # ==========================================
    # GREETINGS
    # ==========================================
    if any(word in msg for word in ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good evening', 'hola', 'namaste']):
        return '👋 Hello! I\'m your AI Health Assistant. I can help you with liver health, blood tests, appointments, and information about this website. How can I assist you today?'
    
    # ==========================================
    # WEBSITE INFORMATION
    # ==========================================
    elif 'information about' in msg and 'website' in msg:
        return """🌐 About LiverAI Platform:
        
This is a comprehensive AI-powered liver health monitoring system that offers:

✅ Blood Test Analysis - Upload your liver function test reports for instant AI analysis
✅ Medical Image Analysis - Scan liver CT/MRI images for disease detection
✅ Risk Assessment - Get detailed risk levels and stage predictions
✅ Appointment Booking - Schedule appointments with specialists based on your results
✅ Patient Dashboard - Track all your test results and medical history
✅ Admin Panel - Healthcare providers can manage patient records
✅ AI Chatbot - Get instant answers about liver health (that's me! 👋)

Our goal is to provide early detection and monitoring of liver diseases using advanced AI technology."""
    
    elif ('what is this' in msg or 'about this' in msg or 'about website' in msg or 'website features' in msg or 
          'what does this website' in msg or 'tell me about website' in msg):
        return """🌐 About LiverAI Platform:
        
This is a comprehensive AI-powered liver health monitoring system that offers:

✅ Blood Test Analysis - Upload your liver function test reports for instant AI analysis
✅ Medical Image Analysis - Scan liver CT/MRI images for disease detection
✅ Risk Assessment - Get detailed risk levels and stage predictions
✅ Appointment Booking - Schedule appointments with specialists based on your results
✅ Patient Dashboard - Track all your test results and medical history
✅ Admin Panel - Healthcare providers can manage patient records
✅ AI Chatbot - Get instant answers about liver health (that's me! 👋)

Our goal is to provide early detection and monitoring of liver diseases using advanced AI technology."""
    
    elif 'how to use' in msg or 'how does it work' in msg or 'getting started' in msg or 'how use' in msg:
        return """📖 How to Use LiverAI:

Step 1: Create Account
• Register with your email, name, and password
• Choose between Patient or Admin role

Step 2: Login
• Access your personalized dashboard

Step 3: Upload Test Results
• Click "Blood Test Prediction" for lab reports
• Click "Image Scan Prediction" for CT/MRI scans
• Fill in required parameters or upload images

Step 4: Get AI Analysis
• Receive instant predictions on disease stage
• View risk levels and recommendations
• Download detailed PDF reports

Step 5: Book Appointments (if needed)
• If risk is Moderate/High/Critical, book with specialists
• Choose date, time, and doctor type
• Receive email confirmation

Step 6: Track History
• View all past results in your dashboard
• Monitor disease progression over time"""
    
    elif 'parameter' in msg or 'what parameter' in msg:
        return """🧪 Blood Test Parameters We Analyze:

1. Total Bilirubin - Measures bile pigment (liver/blood health)
2. Direct Bilirubin - Conjugated bilirubin (bile duct function)
3. Indirect Bilirubin - Unconjugated bilirubin (calculated value)
4. SGPT/ALT - Enzyme indicating liver cell damage
5. SGOT/AST - Enzyme for liver/heart damage
6. Alkaline Phosphatase - Enzyme for bile duct/bone issues
7. Total Proteins - Overall protein levels in blood
8. Albumin - Main protein made by liver
9. Globulin - Immune system proteins (calculated)
10. A/G Ratio - Albumin to Globulin ratio (liver function)
11. Gender - Male/Female (affects normal ranges)
12. Age - Patient age (affects interpretation)

All values are analyzed by our AI to predict liver disease stage and risk level."""
    
    elif 'who can use' in msg or 'user types' in msg or 'user role' in msg or 'roles' in msg:
        return """👥 User Types on LiverAI:

1. Patient Users
• Upload and analyze blood tests
• Upload liver scans (CT/MRI)
• View prediction results
• Book appointments with doctors
• Track medical history
• Download reports
• Chat with AI assistant

2. Admin/Doctor Users
• View all patient records
• Search and filter patient data
• Monitor high-risk patients
• Manage appointment requests
• Generate analytics
• Export patient reports

Anyone can register as a patient. Admin accounts are for healthcare providers."""
    
    # ==========================================
    # NORMAL RANGES & BLOOD TESTS
    # ==========================================
    elif ('normal range' in msg or 'normal value' in msg or 'blood test range' in msg or 
          'what are normal' in msg or 'test normal' in msg):
        return """📊 Normal Blood Test Ranges for Liver Function:

🔹 Total Bilirubin: 0.3-1.2 mg/dL
🔹 Direct Bilirubin: 0-0.3 mg/dL
🔹 Indirect Bilirubin: 0.1-1.0 mg/dL
🔹 SGPT/ALT: 7-55 IU/L (Men: up to 50, Women: up to 35)
🔹 SGOT/AST: 8-40 IU/L (Men: up to 40, Women: up to 32)
🔹 Alkaline Phosphatase: 40-140 IU/L
🔹 Total Proteins: 6.0-8.3 g/dL
🔹 Albumin: 3.5-5.5 g/dL
🔹 Globulin: 2.0-3.5 g/dL
🔹 A/G Ratio: 1.0-2.5

⚠️ Important: Values outside these ranges may indicate liver problems. Always consult a doctor for interpretation."""
    
    elif 'sgpt' in msg or 'alt' in msg and 'what' in msg:
        return """🔬 SGPT/ALT (Alanine Aminotransferase):

What is it?
An enzyme found mainly in the liver. When liver cells are damaged, ALT is released into the bloodstream.

Normal Range: 7-55 IU/L

High Levels Mean:
• Liver cell damage or inflammation
• Hepatitis (viral, alcoholic, or autoimmune)
• Fatty liver disease
• Cirrhosis
• Liver tumors
• Medication side effects

Very High (>10x normal):
• Acute viral hepatitis
• Drug-induced liver injury
• Shock or severe trauma

Causes: Alcohol, obesity, medications, viral infections, autoimmune diseases"""
    
    elif 'sgot' in msg or ('ast' in msg and 'what' in msg):
        return """🔬 SGOT/AST (Aspartate Aminotransferase):

What is it?
An enzyme found in liver, heart, muscles, and kidneys. Less specific to liver than ALT.

Normal Range: 8-40 IU/L

High Levels Mean:
• Liver disease
• Heart attack
• Muscle injury
• Hemolysis (red blood cell destruction)

AST/ALT Ratio:
• Ratio < 1: Usually fatty liver or chronic hepatitis
• Ratio > 2: Suggests alcoholic liver disease or cirrhosis

Note: Always check AST with ALT for better diagnosis"""
    
    elif 'bilirubin' in msg:
        return """🟡 Bilirubin (Total, Direct, Indirect):

What is Bilirubin?
A yellow pigment produced when red blood cells break down. Processed by the liver and excreted in bile.

Types:
1️⃣ Total Bilirubin (0.3-1.2 mg/dL) - Total amount in blood
2️⃣ Direct/Conjugated (0-0.3 mg/dL) - Processed by liver
3️⃣ Indirect/Unconjugated (0.1-1.0 mg/dL) - Not yet processed

High Bilirubin Causes:
• High Indirect: Hemolysis, Gilbert's syndrome
• High Direct: Bile duct obstruction, hepatitis, cirrhosis
• High Total: Liver disease, gallstones, pancreatic cancer

Symptoms of High Bilirubin:
• Jaundice (yellow skin/eyes)
• Dark urine
• Pale stools
• Itching"""
    
    elif 'alkaline phosphatase' in msg or 'alp' in msg:
        return """🔬 Alkaline Phosphatase (ALP):

What is it?
An enzyme found in liver, bile ducts, and bones.

Normal Range: 40-140 IU/L

High Levels Mean:
• Bile duct obstruction (gallstones, tumors)
• Liver disease (cirrhosis, hepatitis)
• Bone disorders (Paget's disease, fractures)
• Pregnancy (normal increase)

Low Levels Mean:
• Malnutrition
• Zinc deficiency
• Hypothyroidism

When High with Other Tests:
• High ALP + High Bilirubin = Bile duct problem
• High ALP + Normal Bilirubin = Possible bone disease"""
    
    elif 'albumin' in msg and 'what' in msg:
        return """🔬 Albumin:

What is it?
The main protein made by the liver. Helps maintain blood volume and transports hormones, vitamins, and drugs.

Normal Range: 3.5-5.5 g/dL

Low Albumin (Hypoalbuminemia) Causes:
• Chronic liver disease (cirrhosis)
• Malnutrition
• Kidney disease (nephrotic syndrome)
• Inflammatory bowel disease
• Severe infections

Symptoms:
• Swelling in legs/ankles (edema)
• Fluid in abdomen (ascites)
• Weakness and fatigue

High Albumin:
Rare, usually due to dehydration"""
    
    elif 'a/g ratio' in msg or 'ag ratio' in msg or 'albumin globulin ratio' in msg:
        return """🔬 A/G Ratio (Albumin/Globulin Ratio):

What is it?
The ratio of albumin to globulin proteins in blood.

Normal Range:1.0-2.5

Low A/G Ratio (<1.0) Means:
• Liver disease (cirrhosis, hepatitis)
• Kidney disease
• Autoimmune disorders
• Multiple myeloma
• Low albumin or high globulin

**High A/G Ratio (>2.5) Means:**
- Genetic deficiency of globulin
- Leukemia
- Immune deficiency

**Calculation:**
A/G Ratio = Albumin ÷ Globulin
If Total Protein = 7.0 g/dL and Albumin = 4.0 g/dL
Globulin = 7.0 - 4.0 = 3.0 g/dL
A/G Ratio = 4.0 ÷ 3.0 = 1.33 ✅"""
    
    # ==========================================
    # LIVER DISEASES
    # ==========================================
    elif 'liver disease' in msg or 'types of liver' in msg or 'what liver disease' in msg:
        return """🏥 Liver Diseases Detected by Our AI:

1. No Liver Disease ✅
- Stage: F0 (Healthy)
- Risk: None/Low
- Description: Normal liver function

2. Fatty Liver Disease (NAFLD) 🟡
- Stage: F1-F2
- Risk: Low to Moderate
- Fat accumulation in liver cells
- Often related to obesity, diabetes

3. Hepatitis 🟠
- Stage: F2-F3
- Risk: Moderate to High
- Liver inflammation (viral, alcoholic, autoimmune)
- Types: Hepatitis A, B, C, D, E

4. Liver Fibrosis 🔴
- Stage: F2-F3
- Risk: Moderate to High
- Scarring of liver tissue
- Progresses to cirrhosis if untreated

5. Cirrhosis 🔴
- Stage: F4
- Risk: High to Critical
- Severe scarring and liver damage
- Can lead to liver failure

6. Liver Cancer (HCC) ⚫
- Stage: Advanced
- Risk: Critical
- Hepatocellular carcinoma
- Often develops from cirrhosis

Would you like details on a specific disease?"""
    
    elif 'fatty liver' in msg or 'nafld' in msg:
        return """🟡 Fatty Liver Disease (NAFLD):

What is it?
Fat builds up in liver cells when you don't drink much alcohol.

Types:
1️⃣ Simple Fatty Liver (Steatosis) - Just fat, no inflammation
2️⃣ NASH (Non-Alcoholic Steatohepatitis) - Fat + inflammation + damage

Causes:
- Obesity & overweight
- Type 2 diabetes
- High cholesterol
- Metabolic syndrome
- Poor diet (high sugar, refined carbs)

Symptoms:
Often NO symptoms! May have:
- Fatigue
- Upper right abdomen discomfort
- Enlarged liver

Diagnosis:
- Blood tests (elevated ALT/AST)
- Ultrasound/CT/MRI
- FibroScan (measures liver stiffness)

Treatment:
✅ Lose 7-10% of body weight
✅ Exercise 150+ minutes/week
✅ Mediterranean diet
✅ Control diabetes & cholesterol
✅ Avoid alcohol
❌ No specific medication (yet)

Prognosis:
- Can be REVERSED with lifestyle changes!
- 20% may progress to NASH
- 20% of NASH may progress to cirrhosis"""
    
    elif 'hepatitis' in msg:
        return """🦠 Hepatitis (Liver Inflammation):

Types:

Hepatitis A (HAV)
- Spread: Contaminated food/water
- Duration: Acute (2-6 months)
- Vaccine: Yes ✅
- Treatment: Rest, fluids (self-limiting)

Hepatitis B (HBV)
- Spread: Blood, sexual contact, mother-to-baby
- Duration: Can become chronic (90% of infants, 5% adults)
- Vaccine: Yes ✅
- Treatment: Antiviral drugs (tenofovir, entecavir)

Hepatitis C (HCV)
- Spread: Blood-to-blood contact
- Duration: 75-85% become chronic
- Vaccine: No ❌
- Treatment: Antiviral drugs (cure rate >95%!)

Hepatitis D (HDV)
- Spread: Only with Hepatitis B
- Rare but severe

Hepatitis E (HEV)
- Spread: Contaminated water
- Usually self-limiting

Alcoholic Hepatitis
- Cause: Heavy alcohol use
- Treatment: Stop drinking immediately

Autoimmune Hepatitis
- Cause: Immune system attacks liver
- Treatment: Immunosuppressants

Symptoms:
- Jaundice (yellow skin/eyes)
- Fatigue
- Nausea
- Abdominal pain
- Dark urine
- Fever"""
    
    elif 'cirrhosis' in msg:
        return """🔴 Cirrhosis (Liver Scarring):

What is it?
Late-stage liver disease where healthy tissue is replaced by scar tissue, permanently damaging the liver.

Stages:
- Compensated - Liver still functions, few/no symptoms
- Decompensated - Liver failing, severe symptoms

Causes:
1. Chronic alcoholism (most common)
2. Chronic hepatitis B or C
3. Fatty liver disease (NASH)
4. Autoimmune diseases
5. Bile duct diseases
6. Genetic disorders (hemochromatosis, Wilson's)

Symptoms:
Early: None or mild fatigue
Advanced:
- Jaundice
- Ascites (fluid in abdomen)
- Edema (swelling in legs)
- Easy bruising/bleeding
- Spider veins on skin
- Confusion (hepatic encephalopathy)
- Itching

Complications:
⚠️ Portal hypertension
⚠️ Varices (enlarged veins that can bleed)
⚠️ Hepatic encephalopathy (brain fog)
⚠️ Liver cancer
⚠️ Liver failure

Treatment:
- Treat underlying cause
- Medications for complications
- Liver transplant (severe cases)
- Avoid alcohol completely
- Low-sodium diet

Prognosis:
Cannot be reversed, but progression can be slowed."""
    
    elif 'fibrosis' in msg or 'liver fibrosis' in msg or 'scarring' in msg:
        return """🟠 Liver Fibrosis:

What is it?
Scarring of the liver due to repeated or continuous injury/inflammation.

Fibrosis Stages (METAVIR Score):
- F0 - No fibrosis (normal)
- F1 - Mild fibrosis (portal fibrosis)
- F2 - Moderate fibrosis (few septa)
- F3 - Severe fibrosis (many septa)
- F4 - Cirrhosis (severe scarring)

Causes:
- Chronic hepatitis B or C
- Alcohol abuse
- NASH (fatty liver with inflammation)
- Autoimmune hepatitis
- Bile duct diseases

Diagnosis:
- Blood tests (FibroTest, APRI score)
- FibroScan (elastography)
- Liver biopsy (gold standard)
- Imaging (ultrasound, MRI)

Treatment:
- Treat underlying disease
- Antifibrotic medications (in development)
- Lifestyle changes
- Regular monitoring

Key Point:
Early fibrosis (F1-F2) can sometimes be REVERSED if the cause is eliminated!"""
    
    elif 'liver cancer' in msg or 'hepatocellular carcinoma' in msg or 'hcc' in msg:
        return """⚫ Liver Cancer (Hepatocellular Carcinoma - HCC):

What is it?
Cancer that starts in liver cells (hepatocytes).

Risk Factors:
- Chronic hepatitis B or C
- Cirrhosis
- Fatty liver disease
- Alcohol abuse
- Aflatoxin exposure (moldy food)
- Diabetes & obesity
- Family history

Symptoms:
Early: Usually none
Advanced:
- Weight loss
- Loss of appetite
- Upper abdominal pain
- Swelling/lump in abdomen
- Jaundice
- White, chalky stools

Diagnosis:
- AFP blood test (tumor marker)
- CT scan or MRI
- Liver biopsy

Stages:
- Stage A (Early) - Single tumor, liver function good
- Stage B (Intermediate) - Multiple tumors
- Stage C (Advanced) - Spread to blood vessels
- Stage D (Terminal) - End-stage

Treatment:
- Surgery: Tumor removal or liver transplant
- Ablation: Kill tumor with heat/cold
- Chemoembolization: Block blood supply to tumor
- Targeted therapy: Sorafenib, lenvatinib
- Immunotherapy: Nivolumab, pembrolizumab

Prevention:
✅ Hepatitis B vaccine
✅ Treat hepatitis C
✅ Limit alcohol
✅ Maintain healthy weight
✅ Regular screening if high-risk"""
    
    # ==========================================
    # SYMPTOMS
    # ==========================================
    elif 'symptom' in msg or 'sign' in msg or 'how do i know' in msg or 'warning sign' in msg:
        return """⚠️ Liver Disease Symptoms:

Early Stage (Often Silent):
- Fatigue & weakness
- Mild nausea
- Loss of appetite
- Mild abdominal discomfort

Moderate Stage:
- Jaundice (yellow skin/eyes) 🟡
- Dark urine (tea-colored) 🟤
- Pale/clay-colored stools
- Itchy skin
- Easy bruising
- Swollen abdomen

Advanced Stage:
- Severe jaundice
- Ascites (fluid in belly)
- Leg swelling (edema)
- Confusion/altered mental state
- Vomiting blood
- Black, tarry stools
- Spider-like blood vessels on skin
- Enlarged spleen

Emergency Signs - Seek Immediate Help:
🚨 Severe abdominal pain
🚨 Vomiting blood
🚨 Severe confusion
🚨 Difficulty breathing
🚨 Sudden swelling

Note: Many liver diseases have NO symptoms until advanced. Regular screening is important!"""
    
    # ==========================================
    # DIET & PREVENTION
    # ==========================================
    elif 'diet' in msg or 'food' in msg or 'eat' in msg or 'nutrition' in msg:
        return """🍎 Liver-Healthy Diet Recommendations:

FOODS TO EAT (Liver-Friendly):
✅ Vegetables: Leafy greens, broccoli, cauliflower, carrots, beets
✅ Fruits: Berries, apples, citrus fruits, grapes
✅ Whole Grains: Oats, brown rice, quinoa
✅ Lean Protein: Fish (salmon, sardines), chicken, tofu, legumes
✅ Healthy Fats: Olive oil, nuts, avocados
✅ Coffee: 2-3 cups/day (shown to protect liver!)
✅ Green Tea: Antioxidant-rich
✅ Garlic: Helps liver enzymes
✅ Turmeric: Anti-inflammatory

FOODS TO AVOID:
❌ Alcohol - #1 enemy of liver
❌ Sugar & refined carbs - White bread, pastries, soda
❌ Fried foods - High in trans fats
❌ Processed meats - Hot dogs, bacon, sausage
❌ High-sodium foods - Canned soups, chips
❌ Raw shellfish - Risk of hepatitis A

LIVER DETOX FOODS:
🌿 Cruciferous vegetables (broccoli, kale)
🌿 Beetroot
🌿 Walnuts
🌿 Grapefruit
🌿 Green tea

HYDRATION:
💧 Drink 8-10 glasses of water daily

DIET PATTERNS:
- Mediterranean Diet ⭐⭐⭐⭐⭐
- DASH Diet
- Plant-based diets"""
    
    elif 'prevent' in msg or 'prevention' in msg or 'avoid liver disease' in msg or 'how to avoid' in msg:
        return """🛡️ How to Prevent Liver Disease:

1. Maintain Healthy Weight
- BMI 18.5-24.9
- Lose weight gradually (1-2 lbs/week)
- Avoid crash diets

2. Limit Alcohol
- Men: Max 2 drinks/day
- Women: Max 1 drink/day
- Better: Avoid completely

3. Get Vaccinated
✅ Hepatitis A vaccine
✅ Hepatitis B vaccine
(No vaccine for Hepatitis C yet)

4. Practice Safe Hygiene
- Wash hands before eating
- Avoid contaminated food/water
- Don't share needles, razors, toothbrushes

5. Exercise Regularly
- 150 minutes moderate activity/week
- Or 75 minutes vigorous activity/week
- Builds muscle, reduces fat

6. Avoid Toxins
- Use medications as prescribed
- Avoid acetaminophen overdose
- Be careful with herbal supplements
- Avoid aflatoxins (moldy nuts/grains)

7. Eat Healthy Diet
- Mediterranean diet
- Lots of vegetables & fruits
- Limit sugar & processed foods

8. Regular Checkups
- Annual blood tests if high-risk
- Liver ultrasound if needed
- Monitor liver enzymes

9. Manage Chronic Conditions
- Control diabetes
- Control cholesterol
- Manage high blood pressure

10. Avoid Risky Behaviors
- Safe sex (prevent hepatitis B/C)
- Sterile tattoo/piercing equipment
- Screen blood before transfusions"""
    
    # ==========================================
    # APPOINTMENTS
    # ==========================================
    elif 'appointment' in msg or 'book' in msg or 'schedule' in msg or 'how to book' in msg:
        return """📅 How to Book an Appointment on LiverAI:

Eligibility:
You can book appointments if your test results show:
- Moderate Risk
- High Risk  
- Critical Risk

Steps to Book:
1️⃣ Complete a blood test or image scan prediction
2️⃣ If eligible, you'll see "Book Appointment" button on results page
3️⃣ Click the button
4️⃣ Fill in the appointment form:
   • Select Date (future dates only)
   • Select Time Slot
   • Choose Specialist Type
   • Add any additional notes/symptoms
5️⃣ Submit the form
6️⃣ Receive confirmation email

Specialist Types Available:
🩺 Gastroenterologist - Digestive system specialist
🩺 Hepatologist - Liver disease specialist (recommended for severe cases)
🩺 General Physician - For initial consultation

After Booking:
✅ Appointment saved in your dashboard
✅ Email confirmation sent
✅ Doctor can view your test results
✅ You can view appointment status

Note: Low-risk patients don't need immediate specialist appointments but should monitor regularly."""
    
    # ==========================================
    # TREATMENT
    # ==========================================
    elif 'treatment' in msg or 'cure' in msg or 'medicine' in msg or 'medication' in msg:
        return """💊 Liver Disease Treatment Options:

Fatty Liver Disease:
- Weight loss (7-10% of body weight)
- Exercise & healthy diet
- Control diabetes & cholesterol
- Vitamin E (in some cases)
- No specific medication yet

Hepatitis B:
- Antiviral drugs: Tenofovir, Entecavir
- Interferon injections (some cases)
- Lifelong monitoring
- Liver transplant (severe)

Hepatitis C:
- Direct-Acting Antivirals (DAAs)
- Sofosbuvir, Ledipasvir, Velpatasvir
- 8-12 week treatment
- Cure rate >95%!

Alcoholic Liver Disease:
- STOP DRINKING (most important!)
- Nutritional support
- Corticosteroids (severe cases)
- Liver transplant (end-stage)

Cirrhosis:
- Treat underlying cause
- Manage complications
- Medications for ascites, encephalopathy
- Beta-blockers for varices
- Liver transplant (only cure)

General Medications:
- Diuretics (for fluid retention)
- Lactulose (for encephalopathy)
- Antibiotics (for infections)
- Vitamin supplements

Important: Always consult a hepatologist for proper treatment plan!"""
    
    # ==========================================
    # DIAGNOSTIC TESTS
    # ==========================================
    elif ('diagnostic test' in msg or 'what test' in msg or 'how to diagnose' in msg or 
          'liver test' in msg or 'tests for liver' in msg):
        return """🔬 Liver Disease Diagnostic Tests:

Blood Tests:
- Liver Function Tests (LFT)
  - ALT/SGPT, AST/SGOT
  - Bilirubin (total, direct, indirect)
  - Alkaline Phosphatase
  - Albumin, Total Protein
  - A/G Ratio

- Additional Tests:
  - Prothrombin Time (PT/INR)
  - Complete Blood Count (CBC)
  - Hepatitis viral markers
  - Alpha-fetoprotein (AFP) for cancer

Imaging Tests:
- Ultrasound - First-line, shows fatty liver, tumors
- CT Scan - Detailed images of liver structure
- MRI - Best for detecting tumors, vascular issues
- FibroScan - Measures liver stiffness (fibrosis)

Advanced Tests:
- Liver Biopsy - Gold standard, examines tissue
- Endoscopy - Checks for varices (enlarged veins)
- ERCP - Examines bile ducts
- FibroTest - Blood test alternative to biopsy

Genetic Tests:
- Hemochromatosis gene (iron overload)
- Wilson's disease gene (copper buildup)
- Alpha-1 antitrypsin deficiency

When to Get Tested:
✅ Annual checkup if high-risk
✅ Symptoms of liver disease
✅ Family history of liver disease
✅ Chronic hepatitis B or C
✅ Heavy alcohol use
✅ Obesity or diabetes"""
    
    # ==========================================
    # RISK FACTORS
    # ==========================================
    elif 'risk factor' in msg or 'what cause' in msg or 'causes of liver' in msg:
        return """⚠️ Liver Disease Risk Factors:

Lifestyle Factors:
🔴 Heavy alcohol consumption (>2 drinks/day men, >1 women)
🔴 Obesity (BMI >30)
🔴 Poor diet (high sugar, saturated fat)
🔴 Lack of exercise
🔴 Unprotected sex (hepatitis B, C risk)
🔴 IV drug use (sharing needles)

Medical Conditions:
🔴 Type 2 diabetes
🔴 High cholesterol
🔴 High triglycerides
🔴 Metabolic syndrome
🔴 Autoimmune diseases

Infections
🔴 Hepatitis B virus
🔴 Hepatitis C virus
🔴 Hepatitis D virus

Toxins & Medications:
🔴 Acetaminophen overdose
🔴 Long-term use of certain medications
🔴 Herbal supplements (some can harm liver)
🔴 Aflatoxins (moldy food)
🔴 Industrial chemicals

Genetic Factors:
🔴 Hemochromatosis (iron buildup)
🔴 Wilson's disease (copper buildup)
🔴 Alpha-1 antitrypsin deficiency
🔴 Family history of liver disease

Other Factors:
🔴 Age (risk increases with age)
🔴 Tattoos/piercings with unsterile equipment
🔴 Blood transfusions before 1992
🔴 Work exposure to toxins

High-Risk Groups:
- People with chronic hepatitis
- Heavy drinkers (>10 years)
- People with diabetes + obesity
- Healthcare workers (needle stick risk)"""
    
    # ==========================================
    # MISCELLANEOUS
    # ==========================================
    elif 'thank' in msg or 'thanks' in msg:
        return '😊 You\'re welcome! Feel free to ask if you have more questions about liver health or our website. Stay healthy!'
    
    elif 'help' in msg or 'what can you do' in msg or 'options' in msg:
        return """🤖 I can help you with:

Liver Health Topics:
- Liver diseases (fatty liver, hepatitis, cirrhosis, etc.)
- Blood test parameters & normal ranges
- Symptoms & warning signs
- Treatment options
- Diet & nutrition recommendations
- Prevention strategies
- Risk factors

Website Features:
- How to use LiverAI
- Blood test analysis
- Image scan predictions
- Booking appointments
- Understanding your results
- User dashboard features
- Admin panel functions

Medical Information:
- Diagnostic tests
- Medications & treatments
- When to see a doctor
- Emergency warning signs

Just type your question and I'll do my best to help! 😊"""
    
    elif 'emergency' in msg or 'urgent' in msg or 'hospital' in msg or 'when to go' in msg:
        return """🚨 WHEN TO SEEK IMMEDIATE MEDICAL HELP:

Call Emergency Services (911) if you have:
🚨 Severe abdominal pain
🚨 Vomiting blood or blood in vomit
🚨 Black, tarry stools (internal bleeding)
🚨 Severe confusion or disorientation
🚨 Difficulty staying awake
🚨 Difficulty breathing
🚨 Sudden severe swelling in abdomen/legs
🚨 Seizures

Go to ER Soon if you have:
⚠️ Yellowing of skin/eyes (jaundice)
⚠️ Severe itching all over body
⚠️ Very dark urine (like cola)
⚠️ Very pale/white stools
⚠️ Unexplained bruising/bleeding
⚠️ Swollen abdomen with pain

Note: This chatbot is for information only. In emergencies, always seek immediate medical attention!"""
    
    elif 'doctor' in msg or 'specialist' in msg or 'hepatologist' in msg or 'when to see' in msg:
        return """👨‍⚕️ When to See a Liver Specialist:

See a Hepatologist (Liver Specialist) if:
- Diagnosed with chronic hepatitis B or C
- Cirrhosis or advanced fibrosis
- Elevated liver enzymes for >6 months
- Suspected liver cancer
- Considering liver transplant
- Autoimmune liver disease
- Genetic liver disorders

See a Gastroenterologist if:
- Digestive issues + liver concerns
- Fatty liver disease
- Bile duct problems
- Need endoscopy or colonoscopy

See a General Physician first if:
- Mild symptoms
- First-time abnormal blood tests
- Prevention & screening
- They can refer you to specialists

What to Bring to Appointment:
📋 All test results (blood, imaging)
📋 List of medications & supplements
📋 Family medical history
📋 List of symptoms & when they started
📋 Questions written down"""
    
    # ==========================================
    # DEFAULT RESPONSE
    # ==========================================
    else:
        return """I'm here to help! You can ask me about:

📊 Blood Tests - Normal ranges, what they mean
🏥 Liver Diseases - Symptoms, causes, treatments
💊 Medications - Treatment options
🍎 Diet & Nutrition- What to eat for healthy liver
⚠️ Symptoms - Warning signs
📅 Appointments - How to book with specialists
🌐 Website- How to use LiverAI platform
🔬 Diagnostic Tests - What tests you may need

Or type 'help' to see all available topics!"""

# 4. VIEW USER'S APPOINTMENTS
@app.route("/my_appointments")
def my_appointments():
    """View all appointments for logged-in user"""
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    with get_db() as con:
        appointments = con.execute("""
            SELECT * FROM appointments 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        """, (session["user_id"],)).fetchall()
    
    return render_template("my_appointments.html", appointments=appointments)

# 5. ADMIN - VIEW ALL APPOINTMENTS
@app.route("/admin_appointments")
def admin_appointments():
    """Admin view of all appointments"""
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    
    with get_db() as con:
        appointments = con.execute("""
            SELECT * FROM appointments 
            ORDER BY 
                CASE status 
                    WHEN 'pending' THEN 1 
                    WHEN 'confirmed' THEN 2 
                    ELSE 3 
                END,
                created_at DESC
        """).fetchall()
    
    return render_template("admin_appointments.html", appointments=appointments)



# ========================================
# DATABASE SETUP
# ========================================
def get_db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con
# ========================================
# FIX: sqlite3.OperationalError: no such table: appointments
# ========================================

# METHOD 1: Update your init_db() function
# Find init_db() in app.py and ADD this table creation:

def init_db():
    with get_db() as con:
        cur = con.cursor()
        
        # Users table (existing)
        cur.execute(""" 
            CREATE TABLE IF NOT EXISTS users(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                phone TEXT,
                guardian_name TEXT,
                guardian_phone TEXT,
                role TEXT DEFAULT 'user',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Records table (existing)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS records(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_type TEXT NOT NULL,
                disease TEXT,
                risk TEXT,
                health_score INTEGER,
                stage TEXT,
                confidence REAL,
                image_path TEXT,
                blood_params TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Messages table (existing)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                subject TEXT,
                message TEXT NOT NULL,
                sender TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # ⭐⭐⭐ ADD THIS - Appointments table (NEW) ⭐⭐⭐
        cur.execute("""
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                record_id INTEGER,
                patient_name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                disease TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                preferred_date TEXT,
                preferred_time TEXT,
                doctor_specialty TEXT,
                message TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (record_id) REFERENCES records(id)
            )
        """)
         # ================= NOTIFICATIONS TABLE =================
        cur.execute("""
            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                message TEXT NOT NULL,
                type TEXT NOT NULL,
                appointment_id INTEGER,
                read INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (appointment_id) REFERENCES appointments(id)
            )
        """)
        
        con.commit()


init_db()


# ADD THIS ROUTE TO YOUR app.py
# This will update all existing records in database
# ========================================

@app.route("/fix_risk_levels")
def fix_risk_levels():
    """
    One-time fix: Update all risk levels in database to match new system
    Visit http://localhost:5000/fix_risk_levels once, then remove this route
    """
    if session.get("role") != "admin":
        return "Admin access required", 403
    
    try:
        with get_db() as con:
            cur = con.cursor()
            
            # Get all records
            records = cur.execute("SELECT id, disease, prediction_type FROM records").fetchall()
            
            updated_count = 0
            
            for record in records:
                record_id = record['id']
                disease = record['disease']
                pred_type = record['prediction_type']
                
                new_risk = None
                
                # Update based on prediction type
                if pred_type == 'blood_test':
                    # Blood test diseases
                    if disease == "No Liver Disease":
                        new_risk = "🟢 No Risk"
                    elif disease == "Fatty Liver Disease":
                        new_risk = "🟡 Low Risk"
                    elif disease == "Hepatitis":
                        new_risk = "🟠 Moderate Risk"
                    elif disease == "Cirrhosis":
                        new_risk = "🔴 High Risk"
                    elif disease == "Severe Liver Inflammation":
                        new_risk = "🔴 Critical Risk"
                    elif disease == "Advanced Liver Disease":
                        new_risk = "🔴 Critical Risk"
                        
                elif pred_type == 'image_scan':
                    # Image scan diseases
                    if disease == "No Fibrosis":
                        new_risk = "🟢 No Risk"
                    elif disease == "Mild Fibrosis":
                        new_risk = "🟡 Low Risk"
                    elif disease == "Moderate Fibrosis":
                        new_risk = "🟠 Medium Risk"
                    elif disease == "Severe Fibrosis":
                        new_risk = "🔴 High Risk"
                    elif disease == "Cirrhosis":
                        new_risk = "🔴 Critical Risk"
                
                # Update if we found a matching risk level
                if new_risk:
                    cur.execute("UPDATE records SET risk = ? WHERE id = ?", (new_risk, record_id))
                    updated_count += 1
                    print(f"✓ Updated record {record_id}: {disease} → {new_risk}")
            
            con.commit()
            
            return f"""
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        max-width: 800px;
                        margin: 50px auto;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }}
                    .container {{
                        background: rgba(255,255,255,0.1);
                        backdrop-filter: blur(10px);
                        padding: 30px;
                        border-radius: 15px;
                    }}
                    h1 {{ color: #ffc107; }}
                    .success {{ color: #4ade80; font-size: 24px; }}
                    a {{
                        display: inline-block;
                        margin-top: 20px;
                        padding: 10px 20px;
                        background: #ffc107;
                        color: #000;
                        text-decoration: none;
                        border-radius: 8px;
                        font-weight: bold;
                    }}
                    a:hover {{ background: #ffb300; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>✅ Risk Levels Updated Successfully!</h1>
                    <p class="success">Updated {updated_count} records in the database.</p>
                    <p>All risk levels have been corrected to match the new system:</p>
                    <ul>
                        <li>🟢 No Risk - for healthy reports</li>
                        <li>🟡 Low Risk - for mild conditions</li>
                        <li>🟠 Moderate Risk - for moderate conditions</li>
                        <li>🔴 High Risk - for serious conditions</li>
                        <li>🔴 Critical Risk - for critical conditions</li>
                    </ul>
                    <a href="{url_for('admin_dashboard')}">← Back to Admin Dashboard</a>
                    <p style="margin-top: 20px; font-size: 12px; opacity: 0.7;">
                        Note: You can now remove the /fix_risk_levels route from your app.py
                    </p>
                </div>
            </body>
            </html>
            """
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h1>Error updating risk levels:</h1><pre>{str(e)}</pre>", 500
# ========================================
# STEP 3: Add email helper functions (copy-paste these)
# ========================================

def send_async_email(app, msg):
    with app.app_context():
        try:
            mail.send(msg)
            print("✓ Email sent successfully")
        except Exception as e:
            print(f"✗ Email error: {e}")

def send_email(subject, recipient, html_body, text_body=None):
    msg = Message(subject, recipients=[recipient])
    msg.html = html_body
    msg.body = text_body or "Please view this email in HTML format."
    Thread(target=send_async_email, args=(app, msg)).start()

def generate_email_html(patient_name, disease, risk, health_score, stage, confidence, prediction_type):
    """Simple email template"""
    
    if 'low' in risk.lower() or 'no' in disease.lower():
        color = '#10b981'
        emoji = '✅'
    elif 'medium' in risk.lower() or 'moderate' in risk.lower():
        color = '#f59e0b'
        emoji = '⚠️'
    else:
        color = '#ef4444'
        emoji = '🚨'
    
    return f"""
    <div style="font-family: Arial; max-width: 600px; margin: 0 auto;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center;">
            <h1>🏥 Liver Disease Test Results</h1>
        </div>
        
        <div style="padding: 30px; background: #f9fafb;">
            <h2>Dear {patient_name},</h2>
            <p>Your liver disease prediction analysis has been completed.</p>
            
            <div style="background: white; padding: 20px; border-left: 4px solid {color}; margin: 20px 0;">
                <h3>{emoji} Results</h3>
                <p><strong>Test Type:</strong> {prediction_type}</p>
                <p><strong>Diagnosis:</strong> {disease}</p>
                <p><strong>Risk Level:</strong> <span style="color: {color};">{risk}</span></p>
                <p><strong>Health Score:</strong> {health_score}/100</p>
                <p><strong>Stage:</strong> {stage}</p>
                <p><strong>Confidence:</strong> {confidence}%</p>
            </div>
            
            <div style="background: #fef3c7; padding: 15px; border-left: 4px solid #f59e0b; margin: 20px 0;">
                <strong>⚠️ Disclaimer:</strong> This is AI-based prediction, NOT a medical diagnosis. 
                Please consult a healthcare professional.
            </div>
            
            <center>
                <a href="http://localhost:5000/login" style="display: inline-block; padding: 12px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 5px;">
                    View Full Report
                </a>
            </center>
        </div>
        
        <div style="text-align: center; padding: 20px; color: #6b7280; font-size: 14px;">
            <p>© 2024 LiverAI - Automated Report System</p>
        </div>
    </div>
    """

# ========================================
# NEW FEATURE 1: ANALYTICS DASHBOARD
# ========================================
# ========================================
# ENHANCED ANALYTICS ROUTE WITH PERCENTAGES
# Replace your existing analytics route
# ========================================

@app.route("/analytics")
def analytics():
    """Enhanced analytics dashboard with percentages"""
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    
    with get_db() as con:
        # Total users
        total_users = con.execute(
            "SELECT COUNT(*) AS count FROM users WHERE role='user'"
        ).fetchone()["count"]
        
        # Total predictions
        total_predictions = con.execute(
            "SELECT COUNT(*) AS count FROM records"
        ).fetchone()["count"]
        
        # Disease distribution with percentages
        disease_counts = con.execute("""
            SELECT disease, COUNT(*) AS count 
            FROM records 
            GROUP BY disease 
            ORDER BY count DESC
        """).fetchall()
        
        # Calculate percentages for diseases
        disease_data = {
            "labels": [],
            "values": [],
            "percentages": []
        }
        for row in disease_counts:
            disease_data["labels"].append(row["disease"])
            disease_data["values"].append(row["count"])
            percentage = (row["count"] / total_predictions * 100) if total_predictions > 0 else 0
            disease_data["percentages"].append(round(percentage, 1))
        
        # Monthly trends (last 12 months)
        monthly_data = con.execute("""
            SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS count 
            FROM records 
            GROUP BY month 
            ORDER BY month DESC 
            LIMIT 12
        """).fetchall()
        
        # Risk level distribution with percentages
        risk_data = con.execute("""
            SELECT risk, COUNT(*) AS count 
            FROM records 
            GROUP BY risk 
            ORDER BY count DESC
        """).fetchall()
        
        # Calculate risk percentages
        risk_stats = {
            "labels": [],
            "values": [],
            "percentages": []
        }
        for row in risk_data:
            risk_stats["labels"].append(row["risk"])
            risk_stats["values"].append(row["count"])
            percentage = (row["count"] / total_predictions * 100) if total_predictions > 0 else 0
            risk_stats["percentages"].append(round(percentage, 1))
        
        # Last 7 days activity
        week_data = con.execute("""
            SELECT DATE(created_at) AS date, COUNT(*) AS count 
            FROM records 
            WHERE created_at >= date('now', '-7 days') 
            GROUP BY date 
            ORDER BY date
        """).fetchall()
        
        # Prediction type distribution with percentages
        blood_count = con.execute(
            "SELECT COUNT(*) AS count FROM records WHERE prediction_type='blood_test'"
        ).fetchone()["count"]
        
        image_count = con.execute(
            "SELECT COUNT(*) AS count FROM records WHERE prediction_type='image_scan'"
        ).fetchone()["count"]
        
        blood_percentage = (blood_count / total_predictions * 100) if total_predictions > 0 else 0
        image_percentage = (image_count / total_predictions * 100) if total_predictions > 0 else 0
        
        # Health score distribution
        score_ranges = con.execute("""
            SELECT 
                CASE 
                    WHEN health_score >= 90 THEN '90-100 (Excellent)'
                    WHEN health_score >= 70 THEN '70-89 (Good)'
                    WHEN health_score >= 50 THEN '50-69 (Fair)'
                    WHEN health_score >= 30 THEN '30-49 (Poor)'
                    ELSE '0-29 (Critical)'
                END AS score_range,
                COUNT(*) AS count
            FROM records
            GROUP BY score_range
            ORDER BY 
                CASE 
                    WHEN health_score >= 90 THEN 1
                    WHEN health_score >= 70 THEN 2
                    WHEN health_score >= 50 THEN 3
                    WHEN health_score >= 30 THEN 4
                    ELSE 5
                END
        """).fetchall()
        
        score_stats = {
            "labels": [row["score_range"] for row in score_ranges],
            "values": [row["count"] for row in score_ranges],
            "percentages": [
                round((row["count"] / total_predictions * 100), 1) if total_predictions > 0 else 0 
                for row in score_ranges
            ]
        }
        
        # Prepare comprehensive stats
        stats = {
            "total_users": total_users,
            "total_predictions": total_predictions,
            "disease_data": disease_data,
            "monthly_data": {
                "labels": [row["month"] for row in monthly_data][::-1],
                "values": [row["count"] for row in monthly_data][::-1]
            },
            "risk_data": risk_stats,
            "week_data": {
                "labels": [row["date"] for row in week_data],
                "values": [row["count"] for row in week_data]
            },
            "prediction_types": {
                "blood_count": blood_count,
                "image_count": image_count,
                "blood_percentage": round(blood_percentage, 1),
                "image_percentage": round(image_percentage, 1)
            },
            "score_stats": score_stats
        }
        
        return render_template("analytics.html", stats=stats)

# ========================================
# NEW FEATURE 9: HEALTH SCORE TRACKER
# ========================================

@app.route("/health_tracker")
def health_tracker():
    """Track health score over time"""
    if session.get("role") != "user":
        return redirect(url_for("login"))
    
    with get_db() as con:
        scores = con.execute("""
            SELECT health_score, DATE(created_at) as date
            FROM records
            WHERE user_id = ?
            ORDER BY created_at ASC
        """, (session["user_id"],)).fetchall()
    
    # Prepare data for chart
    dates = [row["date"] for row in scores]
    values = [row["health_score"] for row in scores]
    
    return render_template("health_tracker.html", dates=dates, values=values)

# ========================================
# 🔧 FIX #1: SEARCH RECORDS WITH PROPER FILTER LOGIC
# ========================================
# ========================================
# 🔧 FIXED: SEARCH RECORDS WITH PROPER FILTER LOGIC
# ========================================
@app.route("/search_records", methods=["GET", "POST"])
def search_records():
    """Advanced search and filter for records - FIXED VERSION"""
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    
    # Base query
    query = """
        SELECT r.*, u.name as patient_name 
        FROM records r 
        JOIN users u ON r.user_id = u.id 
        WHERE 1=1
    """
    params = []
    
    # Get all unique filter options
    with get_db() as con:
        diseases = [row['disease'] for row in con.execute("SELECT DISTINCT disease FROM records WHERE disease IS NOT NULL").fetchall()]
        risks = [row['risk'] for row in con.execute("SELECT DISTINCT risk FROM records WHERE risk IS NOT NULL").fetchall()]
    
    # Store current filter values for template
    current_filters = {
        'disease': '',
        'risk': '',
        'date_from': '',
        'date_to': ''
    }
    
    # Apply filters if POST request
    if request.method == "POST":
        disease = request.form.get("disease", "").strip()
        risk = request.form.get("risk", "").strip()
        date_from = request.form.get("date_from", "").strip()
        date_to = request.form.get("date_to", "").strip()
        
        # Store for template (to keep selected values)
        current_filters['disease'] = disease
        current_filters['risk'] = risk
        current_filters['date_from'] = date_from
        current_filters['date_to'] = date_to
        
        # 🔧 FIX: Check if value is NOT empty (not just truthy)
        # This allows "No Risk" and other actual values to be filtered
        if disease != "":
            query += " AND r.disease = ?"
            params.append(disease)
            
        if risk != "":  # This now properly handles "No Risk"
            query += " AND r.risk = ?"
            params.append(risk)
            
        if date_from != "":
            query += " AND DATE(r.created_at) >= ?"
            params.append(date_from)
            
        if date_to != "":
            query += " AND DATE(r.created_at) <= ?"
            params.append(date_to)
    
    query += " ORDER BY r.id DESC"
    
    with get_db() as con:
        records = con.execute(query, params).fetchall()
    
    return render_template(
        "search_records.html",
        records=records,
        diseases=diseases,
        risks=risks,
        filters=current_filters  # Pass current filters to keep selections
    )




# ========================================
# AUTHENTICATION ROUTES
# ========================================
@app.route("/")
def index():
    user = session.get("user")
    return render_template("index.html", user=user)

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")
        phone = request.form.get("phone")
        guardian_name = request.form.get("guardian_name")
        guardian_phone = request.form.get("guardian_phone")

        if not name or not email or not password:
            return render_template("register.html", error="Name, Email, and Password are required.")

        hashed_password = generate_password_hash(password)

        try:
            with get_db() as con:
                con.execute("""
                    INSERT INTO users(name, email, password, phone, guardian_name, guardian_phone)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, email, hashed_password, phone, guardian_name, guardian_phone))
                con.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already registered.")
        except Exception as e:
            return render_template("register.html", error=f"Error: {e}")

    return render_template("register.html")

# ========================================
# FIXED LOGIN ROUTE - Replace your current login route with this
# ========================================

from flask import Flask, render_template, request, session, redirect, url_for
from werkzeug.security import check_password_hash, generate_password_hash
import sqlite3

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()  # Convert to lowercase
        password = request.form.get("password", "").strip()
        
        # Clear any previous session
        session.clear()

        # -------- ADMIN LOGIN (Hardcoded) --------
        if email == "admin@liver.com" and password == "admin123":
            session["role"] = "admin"
            session["user"] = "Admin"
            print("✓ Admin login successful")  # Debug log
            return redirect(url_for("admin_dashboard"))

        # -------- USER LOGIN (Database) --------
        try:
            with get_db() as con:
                con.row_factory = sqlite3.Row
                
                # Try to find user (case-insensitive email)
                user = con.execute(
                    "SELECT * FROM users WHERE LOWER(email) = ?", 
                    (email,)
                ).fetchone()
                
                # Debug: Print what we found
                if user:
                    print(f"✓ User found: {user['email']}")
                    print(f"  Password in DB: {user['password'][:20]}...")  # First 20 chars
                    print(f"  Is hashed: {user['password'].startswith('pbkdf2:')}")
                else:
                    print(f"✗ No user found with email: {email}")
                    return render_template("login.html", error="Invalid email or password")
                
                # Check password
                stored_password = user["password"]
                
                # Check if password is hashed (starts with pbkdf2:)
                if stored_password.startswith("pbkdf2:") or stored_password.startswith("scrypt:"):
                    # Hashed password - use check_password_hash
                    valid = check_password_hash(stored_password, password)
                    print(f"  Checking hashed password: {valid}")
                else:
                    # Plain text password (for backward compatibility)
                    valid = (stored_password == password)
                    print(f"  Checking plain text password: {valid}")
                    
                    # OPTIONAL: Upgrade to hashed password
                    if valid:
                        print("  Upgrading to hashed password...")
                        hashed = generate_password_hash(password)
                        con.execute(
                            "UPDATE users SET password = ? WHERE id = ?",
                            (hashed, user["id"])
                        )
                        con.commit()

                if valid:
                    # Login successful
                    session["user_id"] = user["id"]
                    session["role"] = "user"
                    session["user"] = user["name"]
                    print(f"✓ Login successful for user: {user['name']}")
                    return redirect(url_for("index"))
                else:
                    print(f"✗ Invalid password for user: {email}")
                    return render_template("login.html", error="Invalid email or password")
                    
        except Exception as e:
            print(f"✗ Database error: {e}")
            return render_template("login.html", error=f"Login error: {str(e)}")

    # GET request - show login form
    return render_template("login.html")


# ========================================
# HELPER: Check if user exists and password format
# ========================================

@app.route("/check_user/<email>")
def check_user(email):
    """Debug route - Remove this in production!"""
    try:
        with get_db() as con:
            con.row_factory = sqlite3.Row
            user = con.execute(
                "SELECT email, password FROM users WHERE LOWER(email) = ?",
                (email.lower(),)
            ).fetchone()
            
            if user:
                is_hashed = user['password'].startswith('pbkdf2:') or user['password'].startswith('scrypt:')
                return {
                    "found": True,
                    "email": user['email'],
                    "password_format": "hashed" if is_hashed else "plain_text",
                    "password_preview": user['password'][:30] + "..."
                }
            else:
                return {"found": False, "message": "User not found"}
    except Exception as e:
        return {"error": str(e)}


# ========================================
# HELPER: Fix all existing passwords
# ========================================

@app.route("/fix_passwords")
def fix_passwords():
    """
    One-time fix: Hash all plain text passwords
    Visit this URL once, then remove this route!
    """
    try:
        with get_db() as con:
            users = con.execute("SELECT id, email, password FROM users").fetchall()
            
            fixed_count = 0
            for user in users:
                password = user[2]  # password column
                
                # If password is not hashed, hash it
                if not (password.startswith('pbkdf2:') or password.startswith('scrypt:')):
                    hashed = generate_password_hash(password)
                    con.execute(
                        "UPDATE users SET password = ? WHERE id = ?",
                        (hashed, user[0])
                    )
                    fixed_count += 1
                    print(f"✓ Hashed password for: {user[1]}")
            
            con.commit()
            return f"Fixed {fixed_count} passwords. All passwords are now hashed."
            
    except Exception as e:
        return f"Error: {str(e)}"
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        email = request.form.get("email")
        phone = request.form.get("phone")
        new_password = request.form.get("new_password")

        with get_db() as con:
            user = con.execute(
                "SELECT id FROM users WHERE email=? AND phone=?",
                (email, phone)
            ).fetchone()

            if user:
                hashed_password = generate_password_hash(new_password)
                con.execute("UPDATE users SET password=? WHERE id=?", (hashed_password, user["id"]))
                con.commit()
                return render_template("login.html", success="Password updated successfully!")
            else:
                return render_template("forgot.html", error="Email and phone do not match.")

    return render_template("forgot.html")

# ========================================
# PREDICTION ROUTES
# ========================================
@app.route("/predict_form")
def predict_form():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("predict.html")

# ========================================
# ENHANCED predict_blood FUNCTION
# Replace your entire predict_blood function with this
# ========================================

@app.route("/predict_blood", methods=["POST"])
def predict_blood():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if not blood_model or not scaler:
        return render_template("predict.html", error="Blood test model not loaded")

    try:
        # Get form data
        age = float(request.form.get("Age", 0))
        tb = float(request.form.get("Total_Bilirubin", 0))
        alk = float(request.form.get("Alkaline_Phosphotase", 0))
        sgpt = float(request.form.get("SGPT", 0))
        sgot = float(request.form.get("SGOT", 0))
        tp = float(request.form.get("Total_Proteins", 0))
        alb = float(request.form.get("Albumin", 0))
        agr = float(request.form.get("A_G_Ratio", 0))

        # Prepare input for ANN
        input_features = np.array([[age, tb, alk, sgpt, sgot, tp, alb, agr]])
        input_scaled = scaler.transform(input_features)
        
        # Make prediction using ANN
        prediction_prob = blood_model.predict(input_scaled, verbose=0)[0][0]
        model_pred = 1 if prediction_prob > 0.5 else 0
        confidence = prediction_prob * 100 if model_pred == 1 else (1 - prediction_prob) * 100

        # ========================================
        # ENHANCED DISEASE CLASSIFICATION
        # ========================================
        
        # Count abnormal parameters
        abnormal_count = 0
        severity_score = 0
        
        # Bilirubin analysis
        if tb > 1.2:
            abnormal_count += 1
            if tb > 3.0:
                severity_score += 3
            elif tb > 2.0:
                severity_score += 2
            else:
                severity_score += 1
        
        # Liver enzymes (SGPT/ALT)
        if sgpt > 55:
            abnormal_count += 1
            if sgpt > 200:
                severity_score += 3
            elif sgpt > 150:
                severity_score += 2
            else:
                severity_score += 1
        
        # Liver enzymes (SGOT/AST)
        if sgot > 40:
            abnormal_count += 1
            if sgot > 200:
                severity_score += 3
            elif sgot > 150:
                severity_score += 2
            else:
                severity_score += 1
        
        # Alkaline Phosphatase
        if alk > 140:
            abnormal_count += 1
            if alk > 400:
                severity_score += 2
            else:
                severity_score += 1
        
        # Albumin (low is bad)
        if alb < 3.5:
            abnormal_count += 1
            if alb < 2.5:
                severity_score += 3
            elif alb < 3.0:
                severity_score += 2
            else:
                severity_score += 1
        
        # A/G Ratio
        if agr < 1.0:
            abnormal_count += 1
            if agr < 0.8:
                severity_score += 2
            else:
                severity_score += 1


        # Check No Disease first (all parameters normal)
                # No Disease
                # Advanced/End-Stage
        # ----------------------------
# 1️⃣ No Disease (highest priority)
# ----------------------------
        if model_pred == 0 and abnormal_count <= 1:
            disease = "No Liver Disease"
            health_score = 95

        # ----------------------------
        # 2️⃣ Advanced/End-Stage
        # ----------------------------
        elif alb < 2.5 and tb > 3.0 and abnormal_count >= 5:
            disease = "Advanced Liver Disease"
            health_score = 20

        # ----------------------------
        # 3️⃣ Severe Inflammation
        # ----------------------------
        elif sgpt > 200 and sgot > 200 and tb > 2.5:
            disease = "Severe Liver Inflammation"
            health_score = 25

        # ----------------------------
        # 4️⃣ Moderate Stage - Structural Damage
        # ----------------------------
        elif alb < 3.0 and abnormal_count >= 4:
            disease = "Cirrhosis"
            health_score = 35

        # ----------------------------
        # 5️⃣ Moderate Stage - Inflammation
        # ----------------------------
        elif (sgpt > 150 or sgot > 150) and tb < 3.0:
            disease = "Hepatitis"
            health_score = 55

        # ----------------------------
        # 6️⃣ Mild/Early Stage
        # ----------------------------
        elif abnormal_count <= 2 and severity_score <= 3:
            disease = "Fatty Liver Disease"
            health_score = 75

        else:
            disease = "Fatty Liver Disease"
            health_score = 65       

        # Get disease information
        info = BLOOD_STAGE_INFO.get(disease, BLOOD_STAGE_INFO["No Liver Disease"])
        
        risk = info["risk"]
        stage = info["stage"]
        health_score_range = info["health_score_range"]
        symptoms = info["symptoms"]
        diet = info["diet"]
        doctor_advice = info["doctor"]

        # Store parameters
        blood_params = f"Age:{age}|TB:{tb}|ALP:{alk}|SGPT:{sgpt}|SGOT:{sgot}|TP:{tp}|Alb:{alb}|AGR:{agr}"

        # Save to database
        # Save to database
        with get_db() as con:
            cur = con.cursor()
            cur.execute("""
                INSERT INTO records(user_id, prediction_type, disease, risk, health_score, stage, confidence, blood_params, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (session["user_id"], "blood_test", disease, risk, health_score, stage, 
                  round(confidence, 2), blood_params, datetime.now().strftime("%Y-%m-%d %H:%M")))
            record_id = cur.lastrowid
            con.commit()
            
            # ⭐ GET USER INFO FOR EMAIL ⭐
            user = con.execute("SELECT name, email FROM users WHERE id = ?", (session["user_id"],)).fetchone()

        # ⭐ SEND EMAIL NOTIFICATION ⭐
        email_subject = f"🏥 Your Liver Test Results - {disease}"
        email_html = generate_email_html(
            patient_name=user['name'],
            disease=disease,
            risk=risk,
            health_score=health_score,
            stage=stage,
            confidence=f"{confidence:.2f}",
            prediction_type="Blood Test Analysis"
        )
        send_email(email_subject, user['email'], email_html)
        # ⭐ ADD THIS RIGHT BEFORE return render_template ⭐
        # Check if appointment is needed
        needs_apt = needs_appointment(risk, disease)

        # Return result with all information
        return render_template("result.html", 
            prediction_type="Blood Test Analysis (ANN Model - 95% Accuracy)",
            advice={
                "disease": disease,
                "risk": risk,
                "health_score": health_score,
                "health_score_range": health_score_range,
                "stage": stage,
                "confidence": f"{confidence:.2f}%",
                "symptoms": symptoms,
                "diet": diet,
                "doctor_recommendation": doctor_advice,
                "disclaimer": "⚠️ IMPORTANT: This is an AI-based prediction and NOT a medical diagnosis. Always consult a qualified healthcare professional for proper diagnosis and treatment."
            },
            record_id=record_id,
            email_sent=True,
            needs_appointment=needs_apt  # ⭐ NEW LINE
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("predict.html", error=f"Prediction Error: {str(e)}")


# ========================================
# SUMMARY OF CHANGES
# ========================================
"""
ENHANCED FEATURES:

1. ✅ Six Disease Classifications:
   - No Liver Disease (Normal)
   - Fatty Liver Disease (Mild)
   - Hepatitis (Moderate - Inflammation)
   - Cirrhosis (Severe - Structural Damage)
   - Severe Liver Inflammation (Critical)
   - Advanced Liver Disease (End-Stage)

2. ✅ Detailed Risk Levels:
   - 🟢 Low Risk (90-100 score)
   - 🟡 Moderate Risk (70-85 score)
   - 🟠 High Risk (40-65 score)
   - 🔴 Critical Risk (0-35 score)

3. ✅ Comprehensive Information for Each:
   - Specific symptoms list
   - Detailed diet recommendations
   - Doctor consultation advice
   - Health score range

4. ✅ Smart Disease Detection:
   - Uses severity scoring
   - Considers multiple parameters
   - Classifies based on biomarker patterns
   - Distinguishes between types of liver damage

5. ✅ Better Clinical Logic:
   - High enzymes + low bilirubin = Hepatitis
   - Low albumin + high bilirubin = Cirrhosis
   - Extremely high enzymes = Severe inflammation
   - Multiple severe markers = Advanced disease

6. 
    High SGPT/SGOT + Normal Bilirubin → Hepatitis
    Low Albumin + High Bilirubin → Cirrhosis  
    Very High Enzymes (200+) → Severe Inflammation
    Multiple Critical Markers → Advanced Disease
"""
# ========================================
# FIXED predict_image ROUTE
# Replace your existing predict_image route with this
# ========================================

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "user_id" not in session:
        return redirect(url_for("login"))
    
    if not image_model:
        return render_template("predict.html", error="Image model not loaded")
    
    if 'file' not in request.files:
        return render_template("predict.html", error="No file uploaded")
    
    file = request.files['file']
    if file.filename == '':
        return render_template("predict.html", error="No file selected")
    
    try:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['user_id']}_{file.filename}")
        file.save(filepath)
        
        # Load and preprocess image
        img = load_img(filepath, target_size=IMAGE_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = image_model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(prediction[0][predicted_index] * 100)
        
        # Get disease name and info
        stage_label = STAGE_MEANING[predicted_class]
        info = MEDICAL_INFO[predicted_class]
        severity = SEVERITY_LEVEL[predicted_class]  # This now has emojis and correct text
        
        # Save to database with CORRECT risk value
        with get_db() as con:
            cur = con.cursor()
            cur.execute("""
                INSERT INTO records(user_id, prediction_type, disease, risk, health_score, stage, confidence, image_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session["user_id"], 
                "image_scan", 
                stage_label,           # e.g., "No Fibrosis"
                severity,              # e.g., "🟢 No Risk" (from SEVERITY_LEVEL)
                info['score'], 
                predicted_class.upper(), 
                confidence, 
                filepath, 
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))
            record_id = cur.lastrowid
            con.commit()
            
            # Get user info for email
            user = con.execute("SELECT name, email FROM users WHERE id = ?", (session["user_id"],)).fetchone()
        
        # Send email notification
        email_subject = f"🏥 Your Liver Scan Results - {stage_label}"
        email_html = generate_email_html(
            patient_name=user['name'],
            disease=stage_label,
            risk=severity,
            health_score=info['score'],
            stage=predicted_class.upper(),
            confidence=f"{confidence:.2f}",
            prediction_type="Image Scan Analysis"
        )
        send_email(email_subject, user['email'], email_html)
        # ⭐ ADD THIS RIGHT BEFORE return render_template ⭐
        # Check if appointment is needed
        needs_apt = needs_appointment(severity, stage_label)
        
        # Return results
        return render_template("result.html",
            prediction_type="Image Scan Analysis (MobileNet Model - 88% Accuracy)",
            advice={
                "disease": stage_label,
                "risk": severity,
                "health_score": info['score'],
                "stage": predicted_class.upper(),
                "confidence": f"{confidence:.2f}%",
                "symptoms": info['symptoms'],
                "diet": info['diet'],
                "doctor_recommendation": info['doctor'],
                "disclaimer": "⚠️ IMPORTANT: This is an AI-based prediction and NOT a medical diagnosis. Always consult a qualified healthcare professional for proper diagnosis and treatment."
            },
            image_path=filepath,
            record_id=record_id,
            email_sent=True,
            needs_appointment=needs_apt  # ⭐ NEW LINE
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template("predict.html", error=f"Prediction Error: {str(e)}")
# ========================================
# USER HISTORY & SETTINGS
# ========================================
@app.route("/user_history")
def user_history():
    if session.get("role") != "user":
        return redirect(url_for("login"))

    with get_db() as con:
        records = con.execute("""
            SELECT id, prediction_type, disease, risk, health_score, stage, confidence, created_at
            FROM records WHERE user_id=? ORDER BY id DESC
        """, (session["user_id"],)).fetchall()

    return render_template("user_history.html", records=records)

@app.route("/view_result/<int:record_id>")
def view_result(record_id):
    with get_db() as con:
        r = con.execute("""
            SELECT prediction_type, disease, risk, health_score, stage, confidence, image_path
            FROM records WHERE id=?
        """, (record_id,)).fetchone()

    if not r:
        return redirect(url_for("user_history"))

    advice = {
        "disease": r["disease"], "risk": r["risk"], "health_score": r["health_score"],
        "stage": r["stage"], "confidence": r["confidence"],
        "disclaimer": "Consult a medical professional."
    }

    return render_template("result.html", 
        prediction_type=r["prediction_type"],
        advice=advice, 
        image_path=r["image_path"],
        record_id=record_id
    )

@app.route("/settings", methods=["GET", "POST"])
def settings():
    if session.get("role") != "user":
        return redirect(url_for("login"))

    user_id = session["user_id"]

    with get_db() as con:
        if request.method == "POST":
            name = request.form.get("name")
            phone = request.form.get("phone")
            guardian_name = request.form.get("guardian_name")
            guardian_phone = request.form.get("guardian_phone")

            con.execute("""
                UPDATE users SET name=?, phone=?, guardian_name=?, guardian_phone=? WHERE id=?
            """, (name, phone, guardian_name, guardian_phone, user_id))
            con.commit()

        user = con.execute("""
            SELECT name, email, phone, guardian_name, guardian_phone FROM users WHERE id=?
        """, (user_id,)).fetchone()

    return render_template("setting.html", user=user)

# ========================================
# ADMIN ROUTES
# ========================================
# ========================================
# FIXED ADMIN DASHBOARD ROUTE
# Replace your existing admin_dashboard route (around line 400)
# ========================================
@app.route("/admin_dashboard")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    
    with get_db() as con:
        # Total predictions
        total = con.execute("SELECT COUNT(*) AS count FROM records").fetchone()["count"]
        
        # Disease detected (anything except "No Liver Disease" and "No Fibrosis")
        disease = con.execute("""
            SELECT COUNT(*) AS count FROM records 
            WHERE disease != 'No Liver Disease' 
            AND disease != 'No Fibrosis'
        """).fetchone()["count"]
        
        # Healthy reports (specifically "No Liver Disease" or "No Fibrosis")
        healthy = con.execute("""
            SELECT COUNT(*) AS count FROM records 
            WHERE disease = 'No Liver Disease' 
            OR disease = 'No Fibrosis'
        """).fetchone()["count"]
        
        # Recent records with risk information
        records = con.execute("""
            SELECT 
                r.id, 
                u.name, 
                u.email, 
                r.prediction_type, 
                r.disease,
                r.risk,
                r.created_at, 
                u.id AS user_id
            FROM records r
            JOIN users u ON r.user_id = u.id
            ORDER BY r.id DESC 
            LIMIT 10
        """).fetchall()
        
        return render_template("admin_dashboard.html",
            total_patients=total,
            disease_count=disease,
            no_disease_count=healthy,
            records=records
        )

@app.route("/admin_history")
def admin_history():
    if session.get("role") != "admin":
        return redirect(url_for("login"))
    
    with get_db() as con:
        records = con.execute("SELECT r.*, u.name as name FROM records r JOIN users u ON r.user_id = u.id ORDER BY r.id DESC").fetchall()

        # Get unique diseases and risks dynamically
        diseases = [row['disease'] for row in records]
        diseases = sorted(list(set(diseases)))  # remove duplicates & sort
        risks = [row['risk'] for row in records]
        risks = sorted(list(set(risks)))        # remove duplicates & sort

    return render_template(
        "admin_history.html", 
        records=records,
        diseases=diseases,
        risks=risks
    )

@app.route("/delete_user/<int:user_id>")
def delete_user(user_id):
    if session.get("role") != "admin":
        return redirect(url_for("login"))

    with get_db() as con:
        con.execute("DELETE FROM records WHERE user_id = ?", (user_id,))
        con.execute("DELETE FROM users WHERE id = ?", (user_id,))
        con.commit()

    return redirect(url_for("admin_dashboard"))

@app.route("/export_excel")
def export_excel():
    if session.get("role") != "admin":
        return redirect(url_for("login"))

    with get_db() as con:
        df = pd.read_sql("SELECT * FROM records", con)

    file = "all_records.xlsx"
    df.to_excel(file, index=False)
    response = send_file(file, as_attachment=True)
    
    try:
        os.remove(file)
    except:
        pass

    return response

# ========================================
# STATIC PAGES
# ========================================
# 3. UPDATE FAQ ROUTE (Optional - Update accuracy info)
# ========================================
@app.route("/faq")
def faq():
    faqs = [
        {"q": "Is this a final diagnosis?", "a": "No. This provides AI-based decision support only."},
        {"q": "How accurate are the predictions?", "a": "ANN Blood model: ~95% accuracy, Image model: ~88% accuracy."},
        {"q": "What model is used for blood tests?", "a": "Artificial Neural Network (ANN) trained on 20,000+ samples."},
        {"q": "Can I use both prediction methods?", "a": "Yes! Use blood tests and image scans together for comprehensive assessment."}
    ]
    return render_template("faq.html", faqs=faqs)


@app.route("/contact", methods=["GET", "POST"])
def contact():
    # Contact info displayed on page
    contact_info = {
        "email": "support@liverhealth.com",
        "phone": "+91 12345 67890",
        "address": "Medical AI Center, Chennai, India"
    }
    
    success = None
    error = None

    # Ensure user is logged in
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()

        # Validate input
        if not name or not email or not message:
            error = "Please fill all required fields."
        else:
            try:
                with get_db() as con:
                    cur = con.cursor()
                    cur.execute("""
                        INSERT INTO messages (user_id, name, email, subject, message, sender)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (user_id, name, email, subject, message, 'user'))
                    con.commit()
                success = "Your message has been sent successfully!"
            except Exception as e:
                error = f"Error saving message: {str(e)}"

    return render_template("contact.html", contact=contact_info, success=success, error=error)

@app.route("/admin_messages")
def admin_messages():
    # Make sure only admin can access
    if session.get("role") != "admin":
        return redirect(url_for("login"))

    with get_db() as con:
        cur = con.cursor()
        # Fetch all messages, including admin replies, grouped by user
        cur.execute("""
        SELECT * FROM messages
        ORDER BY created_at ASC
        """)
        messages = cur.fetchall()

    return render_template("admin_messages.html", messages=messages)

@app.route("/admin_reply/<int:message_id>", methods=["POST"])
def admin_reply(message_id):
    if session.get("role") != "admin":
        return redirect(url_for("login"))

    reply_message = request.form.get("reply_message")

    with get_db() as con:
        cur = con.cursor()
        # Find the original message to get user_id, name, email
        cur.execute("SELECT user_id, name, email FROM messages WHERE id = ?", (message_id,))
        original = cur.fetchone()

        if original and reply_message:
            cur.execute("""
            INSERT INTO messages (user_id, name, email, message, sender)
            VALUES (?, ?, ?, ?, ?)
            """, (original['user_id'], "Admin", original['email'], reply_message, 'admin'))
            con.commit()

    return redirect(url_for("admin_messages"))


@app.route("/admin/messages/delete/<int:message_id>")
def delete_message(message_id):
    with get_db() as con:
        cur = con.cursor()
        cur.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        con.commit()
    return redirect(url_for('admin_messages'))

@app.route("/user/messages")
def user_messages():
    if session.get("role") != "user":
        return redirect(url_for("login"))

    user_id = session.get("user_id")  # Logged-in user's ID
    with get_db() as con:
        cur = con.cursor()
        cur.execute("""
            SELECT * FROM messages
            WHERE user_id = ?
            ORDER BY created_at ASC
        """, (user_id,))
        messages = cur.fetchall()

    return render_template("user_messages.html", messages=messages)



# ========================================
# ADDITIONAL PAGES
# ========================================

@app.route("/about")
def about():
    """About Us page"""
    return render_template("about.html")

@app.route("/privacy")
def privacy():
    """Privacy Policy page"""
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    """Terms of Service page"""
    return render_template("terms.html")

# ========================================
# ERROR HANDLERS
# ========================================

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 error page"""
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    """Custom 500 error page"""
    return render_template("500.html"), 500


# ========================================
# FIXED DOWNLOAD RESULT ROUTE
# ========================================

from flask import send_file
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
from datetime import datetime

@app.route("/download_result/<int:record_id>")
def download_result(record_id):
    """Generate and download PDF report for a prediction result"""
    
    # Check if user is logged in
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))
    
    try:
        # Fetch the record from database
        with get_db() as con:
            con.row_factory = sqlite3.Row
            record = con.execute("""
                SELECT r.*, u.name as patient_name, u.email as patient_email 
                FROM records r
                JOIN users u ON r.user_id = u.id
                WHERE r.id = ? AND r.user_id = ?
            """, (record_id, user_id)).fetchone()
            
            if not record:
                return "Record not found or access denied", 404
            
            # Convert to dictionary for easier access
            record_dict = dict(record)
        
        # Create PDF filename
        filename = f"liver_report_{record_id}_{datetime.now().strftime('%Y%m%d')}.pdf"
        filepath = os.path.join("static", "temp", filename)
        
        # Ensure temp directory exists
        os.makedirs(os.path.join("static", "temp"), exist_ok=True)
        
        # Create PDF
        doc = SimpleDocTemplate(filepath, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        story.append(Paragraph("Liver Disease Prediction Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Patient Information
        story.append(Paragraph("Patient Information", heading_style))
        patient_data = [
            ['Name:', record_dict.get('patient_name', 'N/A')],
            ['Email:', record_dict.get('patient_email', 'N/A')],
            ['Report ID:', str(record_id)],
            ['Test Date:', record_dict.get('created_at', 'N/A')],
            ['Prediction Type:', record_dict.get('prediction_type', 'N/A').replace('_', ' ').title()],
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Results
        story.append(Paragraph("Prediction Results", heading_style))
        
        # Determine risk color
        risk = record_dict.get('risk', '')
        risk_lower = risk.lower() if risk else ''
        if 'low' in risk_lower:
            risk_color = colors.green
        elif 'medium' in risk_lower or 'moderate' in risk_lower:
            risk_color = colors.orange
        else:
            risk_color = colors.red
        
        results_data = [
            ['Disease/Diagnosis:', record_dict.get('disease', 'N/A')],
            ['Risk Level:', record_dict.get('risk', 'N/A')],
            ['Health Score:', f"{record_dict.get('health_score', 0)}/100"],
            ['Stage:', record_dict.get('stage', 'N/A')],
        ]
        
        # Add confidence if available (for image scans)
        confidence = record_dict.get('confidence')
        if confidence is not None and confidence != '':
            results_data.append(['Confidence:', f"{confidence}%"])
        
        results_table = Table(results_data, colWidths=[2*inch, 4*inch])
        results_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (1, 1), (1, 1), risk_color),  # Risk level in color
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        story.append(results_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Medical Disclaimer
        story.append(Paragraph("Medical Disclaimer", heading_style))
        disclaimer_text = """
        This report is generated by an AI-powered clinical decision support system and 
        should NOT be used as a final medical diagnosis. All predictions and recommendations 
        must be reviewed by qualified healthcare professionals. Please consult your doctor 
        or a certified hepatologist for proper diagnosis and treatment.
        """
        story.append(Paragraph(disclaimer_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph("Generated by LiverAI - Liver Disease Prediction System", footer_style))
        story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", footer_style))
        
        # Build PDF
        doc.build(story)
        
        # Send file and cleanup
        response = send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
        # Optional: Delete file after sending
        # os.remove(filepath)
        
        return response
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return f"Error generating report: {str(e)}", 500


# ========================================
# SIMPLER ALTERNATIVE: Text File Download
# ========================================

@app.route("/download_result_txt/<int:record_id>")
def download_result_txt(record_id):
    """Simple text file download - No dependencies needed!"""
    
    user_id = session.get("user_id")
    if not user_id:
        return redirect(url_for("login"))
    
    try:
        with get_db() as con:
            con.row_factory = sqlite3.Row
            record = con.execute("""
                SELECT r.*, u.name as patient_name, u.email as patient_email 
                FROM records r
                JOIN users u ON r.user_id = u.id
                WHERE r.id = ? AND r.user_id = ?
            """, (record_id, user_id)).fetchone()
            
            if not record:
                return "Record not found", 404
            
            # Convert to dict
            r = dict(record)
        
        # Create text report
        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║        LIVER DISEASE PREDICTION REPORT                        ║
╚═══════════════════════════════════════════════════════════════╝

PATIENT INFORMATION
────────────────────────────────────────────────────────────────
Name:              {r.get('patient_name', 'N/A')}
Email:             {r.get('patient_email', 'N/A')}
Report ID:         {record_id}
Test Date:         {r.get('created_at', 'N/A')}
Prediction Type:   {r.get('prediction_type', 'N/A').replace('_', ' ').title()}

PREDICTION RESULTS
────────────────────────────────────────────────────────────────
Disease/Diagnosis: {r.get('disease', 'N/A')}
Risk Level:        {r.get('risk', 'N/A')}
Health Score:      {r.get('health_score', 0)}/100
Stage:             {r.get('stage', 'N/A')}
"""
        
        # Add confidence if available
        if r.get('confidence'):
            report += f"Confidence:        {r.get('confidence')}%\n"
        
        report += """
MEDICAL DISCLAIMER
────────────────────────────────────────────────────────────────
This report is generated by an AI-powered clinical decision 
support system and should NOT be used as a final medical diagnosis. 
All predictions and recommendations must be reviewed by qualified 
healthcare professionals. Please consult your doctor or a certified 
hepatologist for proper diagnosis and treatment.

────────────────────────────────────────────────────────────────
Generated by LiverAI - Liver Disease Prediction System
Report generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
════════════════════════════════════════════════════════════════
"""
        
        # Save to file
        filename = f"liver_report_{record_id}.txt"
        filepath = os.path.join("static", "temp", filename)
        os.makedirs(os.path.join("static", "temp"), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return send_file(filepath, as_attachment=True, download_name=filename)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500


# ========================================
# DELETE ACCOUNT ROUTE
# ========================================

@app.route("/delete_account", methods=["POST"])
def delete_account():
    """Delete user account and all associated data"""
    
    user_id = session.get("user_id")
    role = session.get("role")
    
    # Only regular users can delete their own accounts
    if not user_id or role != "user":
        return redirect(url_for("login"))
    
    try:
        with get_db() as con:
            # Delete user's records first (foreign key constraint)
            con.execute("DELETE FROM records WHERE user_id = ?", (user_id,))
            
            # Delete the user account
            con.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            con.commit()
        
        # Clear session
        session.clear()
        
        # Redirect to home with message
        return redirect(url_for("index"))
        
    except Exception as e:
        print(f"Error deleting account: {e}")
        return redirect(url_for("settings"))


@app.route("/test_email")
def test_email():
    """Test email - visit http://localhost:5000/test_email"""
    try:
        test_html = """
        <div style="font-family: Arial; padding: 30px; text-align: center;">
            <h1 style="color: #667eea;">✅ Email Working!</h1>
            <p style="font-size: 18px;">Your Gmail App Password is configured correctly.</p>
            <p>Email sent from: <strong>pavithra22123@gmail.com</strong></p>
            <hr>
            <p style="color: #666;">Liver Disease Prediction System - Email Test</p>
        </div>
        """
        
        send_email(
            subject="✅ Test Email - Liver App Working!",
            recipient='pavithra22123@gmail.com',  # Sends to yourself
            html_body=test_html
        )
        
        return """
        <html>
        <head><title>Email Test</title></head>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1 style="color: green;">✅ Test Email Sent!</h1>
            <p style="font-size: 18px;">Check your inbox: <strong>pavithra22123@gmail.com</strong></p>
            <p>If you don't see it in 1-2 minutes, check your spam folder.</p>
            <hr>
            <a href="/" style="padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px;">Go Home</a>
        </body>
        </html>
        """
    except Exception as e:
        return f"""
        <html>
        <body style="font-family: Arial; padding: 50px;">
            <h1 style="color: red;">❌ Email Error</h1>
            <pre style="background: #f5f5f5; padding: 20px; border-radius: 5px;">{str(e)}</pre>
            <a href="/">Go Home</a>
        </body>
        </html>
        """
# ========================================
# RUN APPLICATION
# ========================================

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)





