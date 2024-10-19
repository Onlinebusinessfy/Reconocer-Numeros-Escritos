document.getElementById('predictButton').addEventListener('click', async () => {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Por favor, selecciona una imagen.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('predictionResult').innerText = `Predicción: ${result.predicción}`;
});
