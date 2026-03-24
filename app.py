import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from flask import Flask, request, render_template_string, url_for, send_file
import tensorflow as tf
import numpy as np
import cv2
from google import genai
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__)

# --- 1. CONFIGURATION ---
model = tf.keras.models.load_model('pneumonia_model.h5')

# ⚠️ PASTE YOUR ACTUAL API KEY RIGHT HERE ⚠️
client = genai.Client(api_key="AIzaSyA84qAtJO_8FargXm_Z0kJp-jA1i-3H2pQ")

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

last_report = {}


# --- 2. TACTICAL INSIGHT ENGINE ---
def get_clinical_insights(result, confidence):
    prompt = f"""
    Context: Act as a Tactical Medical AI (System: INTELLIGENCE PORTAL V4.2.1).
    Target: Chest X-Ray Scan. 
    Status: {result} detected with {confidence}% confidence.

    Provide a concise, highly technical intelligence dossier including:
    1. PATHOLOGY LOG: Describe the visual evidence (e.g., focal opacities, infiltrates).
    2. SYSTEM RECOMMENDATIONS: Immediate clinical countermeasures based on IDSA/CDC protocols.
    3. METRIC SCAN MARKERS: List 3 key indicators verified in this scan.

    Format without markdown stars. Use a cold, precise, and professional clinical tone.
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return "INTELLIGENCE LINK OFFLINE. VERIFY API CREDENTIALS."


# --- 3. CRASH-PROOF SMOOTH HEATMAP ---
def generate_heatmap(img_array, original_img_path, save_path, base_pred):
    step, box_size = 32, 64
    masked_images, coords = [], []

    # 1. Slide the evaluation box across the image
    for y in range(0, 224, step):
        for x in range(0, 224, step):
            masked_img = np.copy(img_array)
            y_end, x_end = min(224, y + box_size), min(224, x + box_size)
            masked_img[0, y:y_end, x:x_end, :] = 0.5
            masked_images.append(masked_img)
            coords.append((y, y_end, x, x_end))

    # 2. Predict all variations
    batch_images = np.vstack(masked_images)
    preds = model(batch_images, training=False).numpy()

    # 3. Calculate importance map
    heatmap = np.zeros((224, 224))
    for i, (y, y_end, x, x_end) in enumerate(coords):
        heatmap[y:y_end, x:x_end] += (base_pred - preds[i])

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)

    # 4. THE MAGIC: Apply heavy Gaussian Blur to make it look smooth and biological
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # 5. Overlay on original image
    original_img = cv2.imread(original_img_path)
    h, w = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap_colored, 0.4, original_img, 0.6, 0)

    cv2.imwrite(save_path, superimposed_img)


# --- 4. EXPORT DOSSIER (PDF) ---
def create_pdf(data, output_path):
    pdf = FPDF();
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 22);
    pdf.cell(0, 15, "INTELLIGENCE DOSSIER", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font("Helvetica", size=10);
    pdf.cell(0, 10, f"CHRONO: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT", align='C');
    pdf.ln(10)
    pdf.set_font("Helvetica", 'B', 12);
    pdf.cell(0, 10, f"DESIGNATION: {data['name']} | AGE: {data['age']}", new_x="LMARGIN", new_y="NEXT");
    pdf.ln(5)
    pdf.image(data['orig'], x=10, y=55, w=90);
    pdf.image(data['heat'], x=110, y=55, w=90);
    pdf.ln(100)
    pdf.set_font("Helvetica", 'B', 14);
    pdf.cell(0, 10, f"CONCLUSION: {data['res']} ({data['conf']}%)", new_x="LMARGIN", new_y="NEXT");
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    clean_text = data['ins'].replace('**', '').replace('*', '-')
    clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, clean_text)
    pdf.output(output_path)


# --- 5. THE NEW "DARK TACTICAL" UI ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Intelligence Portal V4</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Rajdhani:wght@500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0a0f;
            --surface: #12131a;
            --surface-light: #1c1d26;
            --neon-purple: #9d4edd;
            --neon-red: #ff2a4b;
            --neon-cyan: #00f0ff;
            --text-main: #e0e6ed;
            --text-muted: #6b7280;
            --border: #2a2b36;
        }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px 50px;
            background-image: radial-gradient(rgba(255,255,255,0.03) 1px, transparent 1px);
            background-size: 20px 20px;
        }
        h1, h2, h3, h4, .cyber-text { font-family: 'Rajdhani', sans-serif; letter-spacing: 1.5px; text-transform: uppercase; }

        /* HEADER */
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 15px; margin-bottom: 40px; }
        .header-title h1 { margin: 0; font-size: 28px; color: #fff; }
        .header-title h1 span { color: var(--neon-purple); }
        .header-title p { margin: 5px 0 0 0; font-size: 11px; color: var(--neon-cyan); letter-spacing: 2px; }
        .operator { display: flex; align-items: center; gap: 15px; text-align: right; }
        .operator p { margin: 0; font-size: 12px; color: var(--text-muted); }
        .operator strong { color: #fff; font-family: 'Rajdhani'; font-size: 16px; letter-spacing: 1px;}

        /* CARD STYLES */
        .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 25px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 25px; }
        .card-header h3 { margin: 0; font-size: 18px; }

        /* FORM SECTION */
        .form-grid { display: grid; grid-template-columns: 2fr 1fr 1.5fr 1.5fr; gap: 20px; align-items: end; }
        .input-group label { display: block; font-size: 10px; color: var(--text-muted); margin-bottom: 8px; letter-spacing: 1px; text-transform: uppercase; }
        input[type="text"], input[type="number"] { width: 100%; background: var(--surface-light); border: 1px solid var(--border); color: #fff; padding: 15px; border-radius: 8px; font-family: 'Inter'; box-sizing: border-box;}
        input[type="text"]:focus, input[type="number"]:focus { outline: none; border-color: var(--neon-purple); }

        /* BUTTONS */
        .btn { padding: 15px 25px; border-radius: 8px; border: none; cursor: pointer; font-family: 'Rajdhani'; font-size: 16px; font-weight: 700; transition: 0.3s; display: flex; align-items: center; justify-content: center; width: 100%; box-sizing: border-box;}
        .btn-upload { background: transparent; border: 1px dashed var(--neon-purple); color: var(--neon-purple); }
        .btn-upload:hover { background: rgba(157, 78, 221, 0.1); }
        .btn-primary { background: linear-gradient(90deg, #9d4edd, #b57eed); color: white; box-shadow: 0 0 15px rgba(157, 78, 221, 0.4); }
        .btn-primary:hover { box-shadow: 0 0 25px rgba(157, 78, 221, 0.7); transform: translateY(-2px); }
        .btn-export { background: transparent; border: 1px solid var(--text-muted); color: var(--text-main); width: auto; font-size: 14px;}
        .btn-export:hover { border-color: #fff; color: #fff; }

        /* RESULTS GRID */
        .results-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 30px; }
        .image-box { background: var(--bg); border: 1px solid var(--border); border-radius: 12px; padding: 15px; position: relative; }
        .image-box-header { display: flex; justify-content: space-between; margin-bottom: 15px; font-size: 12px; font-family: 'Rajdhani'; }
        .image-box img { width: 100%; border-radius: 8px; display: block; }
        .badge { background: rgba(255, 42, 75, 0.15); color: var(--neon-red); padding: 5px 10px; border-radius: 4px; border: 1px solid rgba(255, 42, 75, 0.3); font-weight: 600; font-size: 12px;}

        /* DOSSIER SECTION */
        .dossier-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 30px; }
        .brief-card { border-color: rgba(255, 42, 75, 0.3); background: linear-gradient(180deg, var(--surface) 0%, rgba(255, 42, 75, 0.05) 100%); text-align: center; padding: 40px 20px;}
        .brief-card h2 { font-size: 32px; margin: 10px 0; color: #fff;}
        .brief-card .prob { font-size: 64px; font-weight: 700; color: var(--neon-red); margin: 0; line-height: 1;}
        .brief-card .prob span { font-size: 32px; }
        .log-box { font-size: 13px; line-height: 1.6; color: #a0aec0; white-space: pre-line; padding-left: 15px; border-left: 2px solid var(--neon-purple);}
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>RADIOLOGY <span>REPORT</span></h1>
            <p>● SYSTEM LINK: ACTIVE / MODEL V2.1.0</p>
        </div>
        <div class="operator">
            <div>
                <p>OPERATOR</p>
                <strong>SR. OFFICE STAFF </strong>
            </div>
            <div style="width: 40px; height: 40px; border-radius: 50%; background: #2a2b36; border: 2px solid var(--neon-cyan);"></div>
        </div>
    </div>

    <div class="card">
        <div class="card-header">
            <div style="width: 8px; height: 8px; background: var(--neon-purple); border-radius: 50%;"></div>
            <h3>PHASE I: TARGET IDENTIFICATION</h3>
        </div>
        <form method="post" enctype="multipart/form-data" class="form-grid">
            <div class="input-group">
                <label>■ PATIENT DESIGNATION</label>
                <input type="text" name="p_name" placeholder="E.g., Your Name" required>
            </div>
            <div class="input-group">
                <label>■ AGE</label>
                <input type="number" name="p_age" placeholder="69" required>
            </div>
            <div class="input-group">
                <label style="opacity: 0;">UPLOAD</label>
                <label for="file-upload" class="btn btn-upload">☁ UPLOAD X-RAY / JPEG</label>
                <input id="file-upload" type="file" name="image" accept="image/*" required style="display: none;">
            </div>
            <div class="input-group">
                <label style="opacity: 0;">SUBMIT</label>
                <button type="submit" class="btn btn-primary">🚀 RUN TESTS</button>
            </div>
        </form>
    </div>

    {% if result %}
    <div class="results-grid">
        <div class="image-box">
            <div class="image-box-header">
                <span style="color: var(--text-muted);">● SPECTRAL CAPTURE: RAW</span>
                <span style="color: #4299e1;">NODE: PX-9928-A</span>
            </div>
            <img src="{{ original_url }}">
        </div>
        <div class="image-box">
            <div class="image-box-header">
                <span style="color: var(--neon-red);">● NEURAL ANOMALY DETECTION</span>
                <span class="badge">⚠ {{ confidence }}% CONFIDENCE</span>
            </div>
            <img src="{{ heatmap_url }}">
        </div>
    </div>

    <div class="dossier-grid">
        <div class="card brief-card" style="margin-bottom: 0;">
            <p style="color: var(--neon-red); font-size: 12px; letter-spacing: 2px; margin-top:0;">■ INTELLIGENCE BRIEF</p>
            <p style="font-size: 10px; color: var(--text-muted); margin-bottom: 0; letter-spacing: 1px;">CONCLUSION</p>
            <h2 style="color: {{ color }};">{{ result|upper }}</h2>
            <p class="prob" style="color: {{ color }};">{{ confidence }}<span>%</span></p>
            <p style="font-size: 10px; color: var(--text-muted); letter-spacing: 2px; margin-top: 10px;">PROBABILITY FACTOR</p>
        </div>

        <div class="card" style="margin-bottom: 0;">
            <div class="card-header" style="justify-content: space-between; border-bottom: 1px solid var(--border); padding-bottom: 15px;">
                <h3 style="display: flex; align-items: center; gap: 10px;">
                    <span style="color: var(--neon-purple);">📄</span> EXPERT INTELLIGENCE DOSSIER
                </h3>
                <a href="/download_pdf" style="text-decoration: none;">
                    <button class="btn btn-export">↓ EXPORT DOSSIER</button>
                </a>
            </div>

            <p style="font-size: 12px; font-family: 'Rajdhani'; color: var(--neon-purple); letter-spacing: 1px;">PATHOLOGY LOG & SYSTEMS RECOMMENDATION</p>
            <div class="log-box">
                {{ insights }}
            </div>
        </div>
    </div>
    {% endif %}
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    global last_report
    if request.method == 'POST':
        file = request.files['image'];
        name = request.form.get('p_name');
        age = request.form.get('p_age')
        if file:
            filename = file.filename;
            orig_p = os.path.join(UPLOAD_FOLDER, filename);
            heat_p = os.path.join(UPLOAD_FOLDER, 'h_' + filename)
            file.save(orig_p)
            img = cv2.resize(cv2.imread(orig_p), (224, 224)) / 255.0;
            img_array = np.expand_dims(img, axis=0)
            pred = float(model(img_array, training=False).numpy().item())

            res = "PNEUMONIA DETECTED" if pred > 0.5 else "NOMINAL (HEALTHY)"
            color = "#ff2a4b" if pred > 0.5 else "#00f0ff"
            conf = round((pred if pred > 0.5 else 1.0 - pred) * 100, 2)

            generate_heatmap(img_array, orig_p, heat_p, pred)
            ins = get_clinical_insights(res, conf)
            last_report = {'orig': orig_p, 'heat': heat_p, 'res': res, 'conf': conf, 'ins': ins, 'name': name,
                           'age': age}
            return render_template_string(HTML_PAGE, result=res, confidence=conf, insights=ins, color=color,
                                          original_url=url_for('static', filename='uploads/' + filename),
                                          heatmap_url=url_for('static', filename='uploads/h_' + filename))
    return render_template_string(HTML_PAGE)


@app.route('/download_pdf')
def download_pdf():
    try:
        if not last_report: return "No report data found.", 400
        pdf_path = "Intelligence_Dossier.pdf"
        create_pdf(last_report, pdf_path)
        return send_file(pdf_path, as_attachment=True)
    except Exception as e:
        print(f"\n--- PDF CRASH ERROR: {e} ---\n")
        return f"PDF Error: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)