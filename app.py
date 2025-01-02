from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from inference import load_model_and_predict
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    # List all model files in the Models directory
    models = [f for f in os.listdir('Models') if f.endswith('.h5')]
    return render_template('index.html', models=models)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get the selected model name from the form
        model_name = request.form.get('model_name')

        # Call prediction function with file path and model name
        result = load_model_and_predict(file_path, model_name)
        is_cancerous, cancer_type = result.split("|")
        return render_template('result.html', is_cancerous=is_cancerous,
                               cancer_type=cancer_type, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
