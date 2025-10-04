import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os

# Page configuration
st.set_page_config(
    page_title="🍎 Fruit & Vegetable Classifier",
    page_icon="🥕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .calorie-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .category-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_classification_model():
    try:
        model = load_model('FV.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_classification_model()

# Labels dictionary
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalapeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'radish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bell Pepper', 'Chilli Pepper', 'Grapes', 'Jalapeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Radish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

@st.cache_data
def fetch_calories(prediction):
    try:
        app_id = 'af090f89'
        app_key = '7efd4f2fdc1e1fa089916e68c19feac8'
        url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
        headers = {
            'x-app-id': app_id,
            'x-app-key': app_key,
            'Content-Type': 'application/json'
        }
        data = {
            'query': prediction,
            'timezone': 'US/Eastern'
        }
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        if 'foods' in result and len(result['foods']) > 0:
            calories = result['foods'][0]['nf_calories']
            return f"{calories} kcal per serving"
        else:
            return "Calorie info not found"
    except Exception as e:
        return "Unable to fetch calorie information"

def prepare_image(img_path):
    if not model:
        return None
    
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(y_class[0])
    res = labels[y]
    confidence = float(answer[0][y])
    return res.capitalize(), confidence

def main():
    # Header
    st.markdown('<h1 class="main-header">🍎 Fruit & Vegetable Classifier 🥕</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.write("""
        This AI-powered app can identify 36 different types of fruits and vegetables from images.
        
        **Features:**
        - 🔍 Image classification
        - 📊 Confidence scores
        - 🍎 Fruit/Vegetable categorization
        - 🔥 Calorie information
        """)
        
        st.header("📝 Instructions")
        st.write("""
        1. Upload an image of a fruit or vegetable
        2. Wait for the AI to analyze it
        3. View the results and nutritional info
        """)
        
        st.header("🎯 Supported Items")
        with st.expander("View all 36 categories"):
            st.write("**Fruits:**", ", ".join(fruits))
            st.write("**Vegetables:**", ", ".join(vegetables))
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        img_file = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "png", "jpeg"],
            help="Upload a clear image of a fruit or vegetable"
        )
        
        if img_file is not None:
            # Create upload directory if it doesn't exist
            os.makedirs('./upload_images', exist_ok=True)
            
            img = Image.open(img_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            
            # Save image
            save_image_path = f'./upload_images/{img_file.name}'
            with open(save_image_path, "wb") as f:
                f.write(img_file.getbuffer())
            
            # Analysis
            with st.spinner('🤖 Analyzing image...'):
                result = prepare_image(save_image_path)
                
                if result:
                    prediction, confidence = result
                    
                    with col2:
                        st.header("📊 Results")
                        
                        # Category
                        category = "🍎 Fruit" if prediction in fruits else "🥕 Vegetable"
                        st.markdown(f"""
                        <div class="category-card">
                            <h3>{category}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Prediction
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>🎯 Prediction: {prediction}</h2>
                            <p>Confidence: {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence meter
                        st.progress(confidence)
                        
                        # Calorie information
                        with st.spinner('🔥 Fetching nutrition info...'):
                            cal = fetch_calories(prediction)
                            st.markdown(f"""
                            <div class="calorie-card">
                                <h3>🔥 Nutrition Info</h3>
                                <p>{cal} (per 100g)</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Additional info
                        st.success(f"✅ Successfully identified as {prediction}!")
                        
                        if confidence < 0.7:
                            st.warning("⚠️ Low confidence prediction. Try uploading a clearer image.")
                        elif confidence > 0.9:
                            st.info("🎉 High confidence prediction!")
                else:
                    st.error("❌ Error processing image. Please try again.")

if __name__ == "__main__":
    main()
