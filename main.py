from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = FastAPI()
model = load_model('modelo_digit_recognition.keras')

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>AI Lab - Modelo de Deep Learning</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #ffffff;
                    color: #000000;
                    margin: 0;
                    padding: 0;
                }
                header {
                    background-color: #ffffff;
                    color: #ff6600;
                    padding: 20px 0;
                    text-align: center;
                    font-size: 48px;  /* Tamaño aún más grande */
                    font-weight: bold;
                }
                main {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    height: 70vh;
                    background-color: #ffffff;
                    color: #000000;
                }
                h1 {
                    font-size: 36px;  /* Manteniendo este tamaño */
                    margin-bottom: 20px;
                }
                form {
                    margin: 20px;
                }
                input[type="file"] {
                    padding: 15px;
                    margin-bottom: 20px;
                    font-size: 18px;
                    border: 2px solid #000000;
                    color: #000000;
                    background-color: #ffffff;
                    width: 100%;
                    box-sizing: border-box;
                }
                input[type="submit"] {
                    background-color: #ff6600;
                    border: none;
                    padding: 15px 30px;
                    color: white;
                    cursor: pointer;
                    font-size: 20px;
                    width: 100%;
                    box-sizing: border-box;
                }
                input[type="submit"]:hover {
                    background-color: #ff9933;
                }
                #result {
                    margin-top: 20px;
                    font-size: 24px;
                    font-weight: bold;
                    color: #ff6600;
                }
                footer {
                    background-color: #000000;
                    color: white;
                    padding: 40px 0;
                    position: fixed;
                    width: 100%;
                    bottom: 0;
                    display: flex;
                    justify-content: space-around; /* Cambiado para centrar y distanciar */
                    align-items: center;
                    font-size: 24px;    /* Tamaño de fuente más grande */
                    height: 80px;        /* Mayor altura del footer */
                }
                footer div {
                    font-size: 24px;
                    padding: 0 40px;    /* Aumentar espaciado de los textos */
                }
            </style>
        </head>
        <body>
            <header>Modelo de Deep Learning de Clasificación</header>
            <main>
                <h1>Sube una imagen de un número escrito a mano</h1>
                <form id="upload-form" action="/predict/" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*">
                    <input type="submit" value="Subir">
                </form>
                <div id="result"></div>
            </main>
            <footer>
                <div>AI Lab</div>
                <div>Samuel Dominguez</div>
            </footer>
            <script>
                const form = document.getElementById('upload-form');
                const resultDiv = document.getElementById('result');
                
                form.addEventListener('submit', async function(event) {
                    event.preventDefault();
                    
                    const formData = new FormData(form);
                    
                    const response = await fetch('/predict/', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    resultDiv.textContent = 'El número predicho es: ' + result.predicted_digit;
                });
            </script>
        </body>
    </html>
    """

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lee la imagen
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return {"error": "Error al leer la imagen. Asegúrate de que es una imagen válida."}

    # Inversión y binarización de la imagen para que sea similar a MNIST
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Redimensiona y normaliza la imagen
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0  # Normaliza entre 0 y 1
    img = np.expand_dims(img, axis=-1)  # Añadir dimensión del canal
    img = np.expand_dims(img, axis=0)   # Añadir dimensión de batch

    # Realiza la predicción
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions)

    return {"predicted_digit": int(predicted_digit)}

# Monta la carpeta estática para archivos HTML si es necesario
app.mount("/static", StaticFiles(directory="static"), name="static")
