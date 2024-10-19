from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = FastAPI()
model = load_model('modelo_digit_recognition.h5')

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>Reconocimiento de Números</title>
        </head>
        <body>
            <h1>Sube una imagen de un número escrito a mano</h1>
            <form action="/predict/" method="post" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Subir">
            </form>
        </body>
    </html>
    """

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lee la imagen y preprocesa
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    # Redimensiona y normaliza la imagen
    img = cv2.resize(img, (28, 28))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)  # Agrega una dimensión para el canal
    img = np.expand_dims(img, axis=0)   # Agrega una dimensión para el batch

    # Realiza la predicción
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions)

    return {"predicted_digit": int(predicted_digit)}

# Monta la carpeta estática para archivos HTML si es necesario
app.mount("/static", StaticFiles(directory="static"), name="static")
