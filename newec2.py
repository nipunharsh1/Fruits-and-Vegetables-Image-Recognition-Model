import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from flask import Flask, jsonify, request
import os
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
try:
    model = load_model('FV.h5')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Labels dictionary
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalapeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'radish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path):
    try:
        if not model:
            return None, None
            
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        answer = model.predict(img)
        y_class = answer.argmax(axis=-1)
        y = int(y_class[0])
        res = labels[y]
        confidence = float(answer[0][y])
        
        logger.info(f"Prediction: {res}, Confidence: {confidence}")
        return res.capitalize(), confidence
    except Exception as e:
        logger.error(f"Error in prepare_image: {e}")
        return None, None

# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory
os.makedirs('./upload_images', exist_ok=True)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Fruit & Vegetable Classification API",
        "model_loaded": model is not None,
        "supported_classes": len(labels)
    })

@app.route('/predict', methods=['POST'])
def infer_image():
    try:
        # Check if model is loaded
        if not model:
            return jsonify({
                "error": "Model not loaded",
                "success": False
            }), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                "error": "No file provided",
                "success": False
            }), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "success": False
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type. Allowed types: png, jpg, jpeg, gif",
                "success": False
            }), 400
        
        # Save file securely
        filename = secure_filename(file.filename)
        img_path = os.path.join('./upload_images', filename)
        file.save(img_path)
        
        # Make prediction
        result, confidence = prepare_image(img_path)
        
        if result is None:
            return jsonify({
                "error": "Error processing image",
                "success": False
            }), 500
        
        # Clean up uploaded file
        try:
            os.remove(img_path)
        except:
            pass
        
        # Determine category
        fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalapeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
                 'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
        category = "fruit" if result in fruits else "vegetable"
        
        return jsonify({
            "success": True,
            "prediction": result,
            "confidence": confidence,
            "category": category,
            "message": f"Successfully classified as {result}"
        })
        
    except Exception as e:
        logger.error(f"Error in infer_image: {e}")
        return jsonify({
            "error": "Internal server error",
            "success": False
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "File too large. Maximum size is 16MB",
        "success": False
    }), 413

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
