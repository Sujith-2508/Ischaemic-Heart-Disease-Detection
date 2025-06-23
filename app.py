from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)

model = joblib.load("heart_detection.pkl")
stenosis_model = tf.keras.models.load_model("angiogram.h5")

def preprocess_image(file, size=224):
    img = Image.open(file).convert('RGB')
    resized = img.resize((size, size))
    display_img = img.resize((512, 512))
    return np.array(resized) / 255.0, display_img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        sex = request.form['sex']
        sex_encoded = 1 if sex == 'Male' else 0
        bmi = float(request.form['bmi'])
        temp = float(request.form['temp'])
        heart_rate = float(request.form['heart_rate'])
        spo2 = float(request.form['spo2'])
        ecg = int(request.form['ecg'])

        features = np.array([[age, weight, height, sex_encoded, bmi, temp, heart_rate, spo2, ecg]])
        cad_prob = model.predict_proba(features)[0][1]

        if cad_prob > 0.5:
            diagnosis = (
                "<p style='color:red;'>You've been diagnosed with Coronary Artery Disease (CAD).</p>"
                "<p>This means there is narrowing or blockage of the arteries that supply blood to your heart.</p>"
                "<p>CAD can be managed with lifestyle changes and sometimes medical procedures.</p>"
            )

            uploaded_files = request.files.getlist('angiogram')

            if uploaded_files:
                results = ""
                for file in uploaded_files:
                    filename = file.filename
                    ext = os.path.splitext(filename)[1].lower()

                    if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                        results += f"<p style='color:red;'>Unsupported file type: {filename}</p>"
                        continue

                    try:
                        img_arr, img_disp = preprocess_image(file)
                        input_batch = np.expand_dims(img_arr, axis=0)
                        prediction = stenosis_model.predict(input_batch, verbose=0)[0]
                        classes = ["Mild", "Moderate", "Severe"]
                        predicted_class = classes[np.argmax(prediction)]

                        buffered = io.BytesIO()
                        img_disp.save(buffered, format="PNG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()

                        results += (
                            f"<div style='margin-bottom:20px;'>"
                            f"<p><strong>Analyzed Severity:</strong> {predicted_class}</p>"
                            f"<img src='data:image/png;base64,{img_str}' style='max-width:400px;'/>"
                            f"</div>"
                        )
                    except Exception as e:
                        results += f"<p style='color:red;'>Error processing {filename}: {str(e)}</p>"

                diagnosis += f"<h3>Stenosis Severity</h3>{results}"
            else:
                diagnosis += "<p style='color:orange;'>Upload angiogram images.</p>"

        else:
            diagnosis = (
                "<p style='color:green;'>No evidence of CAD detected.</p>"
                "<p>Continue a heart-healthy lifestyle with a balanced diet and regular activity.</p>"
            )

        return f"""<div style='display:flex; justify-content:center; align-items:center; height:100vh;'>
                            <div style='max-width:600px; padding:20px; border:1px solid #ccc; border-radius:10px; box-shadow:0 0 10px rgba(0,0,0,0.1); text-align:center; background-color:#f9f9f9;'>
                                {diagnosis}
                                <br>
                                <a href='/' style='display:inline-block; margin-top:20px; text-decoration:none; color:orange;'>Back</a>
                            </div>
                    </div>
                """


    except Exception as e:
        return f"<p style='color:red;'>Error: {str(e)}</p><br><a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(debug=True)
