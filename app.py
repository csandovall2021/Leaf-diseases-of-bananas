from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import sys

# Añadir el directorio actual al PATH para importar predict.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import predict_image

app = Flask(__name__)

# Configuración para guardar las imágenes subidas temporalmente
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ruta para servir el formulario HTML
@app.route('/')
def index():
    return render_template('upload.html')

# Ruta para manejar la subida de imágenes y realizar la predicción
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró el archivo en la solicitud"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Realizar la predicción
        prediction_result = predict_image(filepath)
        
        # Eliminar el archivo temporalmente después de la predicción
        os.remove(filepath)

        if "error" in prediction_result:
            return jsonify(prediction_result), 500
        else:
            return jsonify(prediction_result), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
