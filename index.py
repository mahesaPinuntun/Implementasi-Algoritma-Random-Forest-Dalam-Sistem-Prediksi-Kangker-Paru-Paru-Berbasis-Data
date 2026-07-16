import os
import joblib
import pandas as pd
import traceback
import requests
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
# LOAD HTML TEMPLATES FROM GITHUB UI REPO
# ══════════════════════════════════════════════════════════════
BASE_URL = 'https://raw.githubusercontent.com/mahesaPinuntun/UIforRandomForestLungCancerClassifier/main'

_cache = {}

def get_html(filename):
    if filename not in _cache:
        url = f'{BASE_URL}/{filename}'
        print(f'[OK] Fetching {filename} from GitHub...')
        resp = requests.get(url, timeout=10)
        _cache[filename] = resp.text
        print(f'[OK] {filename} cached ({len(resp.text)} chars)')
    return _cache[filename]

# ══════════════════════════════════════════════════════════════
# LOAD CANCER MODEL BUNDLE
# ══════════════════════════════════════════════════════════════
bundle = joblib.load('trainedmodel/cancer_model_bundle.joblib')
print(f"[OK] Model bundle loaded")
print(f"[OK] Test accuracy : {bundle['metadata']['test_accuracy']:.2%}")
print(f"[OK] Features      : {bundle['feature_count']}")
print(f"[OK] Classes       : {list(bundle['label_encoder'].classes_)}")

# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return get_html('dashboard.html')


@app.route('/dashboard')
def dashboard():
    return render_template_string(get_html('dashboard.html'))


@app.route('/layout')
def layout():
    return render_template_string(get_html('layout.html'))


# ══════════════════════════════════════════════════════════════
# CANCER PREDICTION ROUTE
# ══════════════════════════════════════════════════════════════

@app.route('/cekpotensikangker', methods=['GET', 'POST'])
def cekpotensikangker():

    features = bundle['feature_names']

    # Default values
    prediction    = None
    label_id      = None
    color         = None
    icon          = None
    advice        = None
    confidence    = 0
    prob_high     = 0
    prob_medium   = 0
    prob_low      = 0
    error_message = None

    if request.method == 'POST':
        try:
            # Step 1 — collect 23 form values
            input_data = [int(request.form[f]) for f in features]

            # Step 2 — clip to valid training ranges
            input_clipped = []
            for i, val in enumerate(input_data):
                if i == 0:    input_clipped.append(val)                 # Age
                elif i == 1:  input_clipped.append(max(1, min(2, val))) # Gender
                else:         input_clipped.append(max(1, min(8, val))) # Features 1-8

            # Step 3 — build DataFrame with feature names
            input_df = pd.DataFrame([input_clipped], columns=features)

            # Step 4 — predict (convert numpy.int64 → plain int)
            pred_encoded = int(bundle['model'].predict(input_df)[0])

            # Step 5 — decode number → text label
            prediction = bundle['label_encoder'].inverse_transform([pred_encoded])[0]

            # Step 6 — get probabilities for all 3 classes
            proba   = bundle['model'].predict_proba(input_df)[0]
            classes = bundle['label_encoder'].classes_

            prob_dict   = {cls: round(float(proba[i]) * 100, 2) for i, cls in enumerate(classes)}
            prob_high   = prob_dict.get('High',   0)
            prob_medium = prob_dict.get('Medium', 0)
            prob_low    = prob_dict.get('Low',    0)
            confidence  = prob_dict.get(prediction, 0)

            # Step 7 — pull display data from bundle
            color    = bundle['class_colors'][prediction]
            icon     = bundle['class_icons'][prediction]
            advice   = bundle['class_advice'][prediction]
            label_id = bundle['class_labels_id'][pred_encoded]

            print(f"[PREDICT] {prediction} ({confidence}%) High:{prob_high} Med:{prob_medium} Low:{prob_low}")

        except KeyError as e:
            error_message = f"Input tidak lengkap — field hilang: {e}"
            print(f"[ERROR] KeyError: {e}")
        except ValueError as e:
            error_message = f"Nilai tidak valid — harap isi semua field: {e}"
            print(f"[ERROR] ValueError: {e}")
        except Exception as e:
            error_message = f"Terjadi kesalahan: {str(e)}"
            print(f"[ERROR] Unexpected:")
            traceback.print_exc()

    # render_template_string works exactly like render_template
    # all {{ }} and {% %} Jinja2 tags in the HTML work normally
    return render_template_string(
        get_html('cekpotensikangker.html'),
        prediction    = prediction,
        label_id      = label_id,
        color         = color,
        icon          = icon,
        advice        = advice,
        confidence    = confidence,
        prob_high     = prob_high,
        prob_medium   = prob_medium,
        prob_low      = prob_low,
        error_message = error_message,
    )


# ══════════════════════════════════════════════════════════════
# DEBUG ROUTE — visit /debug to verify everything loaded OK
# ══════════════════════════════════════════════════════════════

@app.route('/debug')
def debug():
    return jsonify({
        'status'          : 'OK',
        'model_accuracy'  : bundle['metadata']['test_accuracy'],
        'classes'         : list(bundle['label_encoder'].classes_),
        'feature_count'   : bundle['feature_count'],
        'templates_cached': list(_cache.keys()),
        'cv_mean'         : bundle['metadata']['cv_mean'],
        'overfit_gap'     : bundle['metadata']['overfit_gap'],
    })
