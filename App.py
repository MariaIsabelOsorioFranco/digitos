import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Función de predicción
def predictDigit(image):
    # Se espera que el modelo esté en la carpeta model/
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    return np.argmax(pred[0])

# Configuración de página
st.set_page_config(page_title='DigitVision Pink AI', page_icon="🌸", layout="wide")

# Estilos Rosados
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #fce7f3, #fbcfe8); }
    section[data-testid="stSidebar"] { background-color: #fdf2f8 !important; }
    h1, h2, h3 { color: #881337 !important; text-align: center; }
    .stButton>button { 
        background-color: #db2777 !important; 
        color: white !important; 
        border-radius: 20px; 
        border: none;
    }
    .stSlider label { color: #9f1239 !important; }
</style>
""", unsafe_allow_html=True)

st.title('🌸 DigitVision Pink AI')
st.subheader("Dibuja un dígito en el lienzo y descubre qué número es")

# Configuración del Canvas
stroke_width = st.slider('Ancho de línea', 1, 30, 15)
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    height=200,
    width=200,
    key="canvas",
)

# Botón de predicción
if st.button('✨ Predecir ahora'):
    if canvas_result.image_data is not None:
        # Procesar imagen del canvas
        img_array = canvas_result.image_data.astype('uint8')
        input_image = Image.fromarray(img_array, 'RGBA')
        
        # Predicción
        res = predictDigit(input_image)
        st.balloons()
        st.success(f'### ¡El dígito es: {res}!')
    else:
        st.warning('¡Por favor, dibuja algo primero!')

# Sidebar
st.sidebar.title("🎀 Acerca de")
st.sidebar.info("""
Esta aplicación utiliza una red neuronal 
para reconocer dígitos escritos a mano.

**Estilo:** Pink Edition 🌸
**Basado en:** Vinay Uniyal
""")
