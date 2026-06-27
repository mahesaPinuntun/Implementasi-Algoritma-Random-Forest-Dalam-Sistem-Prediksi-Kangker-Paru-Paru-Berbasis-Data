from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
import os
import joblib
import pandas as pd
import traceback

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ══════════════════════════════════════════════════════════════
# LOAD CANCER MODEL BUNDLE — loaded once at startup
#
# bundle is a Python DICTIONARY containing:
#   bundle['model']           → trained RandomForest object
#   bundle['label_encoder']   → converts 0/1/2 → High/Low/Medium
#   bundle['class_labels']    → {0:'High', 1:'Low', 2:'Medium'}
#   bundle['class_labels_id'] → {0:'Risiko Tinggi', ...}
#   bundle['class_colors']    → {'High':'red', ...}
#   bundle['class_icons']     → {'High':'⚠', ...}
#   bundle['class_advice']    → {'High':'Segera konsultasikan...', ...}
#   bundle['feature_names']   → list of 23 feature names in order
#   bundle['feature_count']   → 23
#   bundle['metadata']        → accuracy, overfit, cv scores etc.
# ══════════════════════════════════════════════════════════════
bundle = joblib.load('trainedmodel/cancer_model_bundle.joblib')
print(f"[OK] Model bundle loaded")
print(f"[OK] Test accuracy : {bundle['metadata']['test_accuracy']:.2%}")
print(f"[OK] Features      : {bundle['feature_count']}")
print(f"[OK] Classes       : {list(bundle['label_encoder'].classes_)}")


# ══════════════════════════════════════════════════════════════
# ROUTES — General
# ══════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return redirect(url_for('cekpotensikangker'))
#



@app.route('/about')
def about():
    return render_template('about.html')



# ══════════════════════════════════════════════════════════════
# CANCER PREDICTION ROUTE
# ══════════════════════════════════════════════════════════════

@app.route('/cekpotensikangker', methods=['GET', 'POST'])
def cekpotensikangker():

    # Feature list pulled from bundle — guaranteed correct order
    # ['Age','Gender','Air Pollution','Alcohol use','Dust Allergy',
    #  'OccuPational Hazards','Genetic Risk','chronic Lung Disease',
    #  'Balanced Diet','Obesity','Smoking','Passive Smoker','Chest Pain',
    #  'Coughing of Blood','Fatigue','Weight Loss','Shortness of Breath',
    #  'Wheezing','Swallowing Difficulty','Clubbing of Finger Nails',
    #  'Frequent Cold','Dry Cough','Snoring']
    features = bundle['feature_names']

    # Default values — nothing predicted yet
    prediction    = None   # 'High', 'Low', or 'Medium'
    label_id      = None   # 'Risiko Tinggi', 'Risiko Rendah', 'Risiko Sedang'
    color         = None   # 'red', 'green', 'yellow'
    icon          = None   # '⚠', '✅', '⚡'
    advice        = None   # Indonesian advice text
    confidence    = 0      # highest probability %
    prob_high     = 0      # probability of High class
    prob_medium   = 0      # probability of Medium class
    prob_low      = 0      # probability of Low class
    error_message = None   # error string if something goes wrong

    if request.method == 'POST':
        try:
            # ── Step 1: Collect all 23 form values ───────────
            input_data = [int(request.form[f]) for f in features]

            # ── Step 2: Clip to valid training ranges ─────────
            # Model was trained on: Age (unrestricted), Gender (1-2),
            # all other features (1-8)
            input_clipped = []
            for i, val in enumerate(input_data):
                if i == 0:    input_clipped.append(val)                 # Age
                elif i == 1:  input_clipped.append(max(1, min(2, val))) # Gender
                else:         input_clipped.append(max(1, min(8, val))) # Features 1-8

            # ── Step 3: Build DataFrame with feature names ────
            # Prevents sklearn warning and ensures correct feature order
            input_df = pd.DataFrame([input_clipped], columns=features)

            # ── Step 4: Predict ───────────────────────────────
            # bundle['model'].predict() returns numpy.int64
            # Convert to plain int to avoid dict lookup issues
            pred_encoded = int(bundle['model'].predict(input_df)[0])
            # pred_encoded is now 0, 1, or 2

            # ── Step 5: Decode number → text label ───────────
            # 0 → 'High', 1 → 'Low', 2 → 'Medium'
            prediction = bundle['label_encoder'].inverse_transform([pred_encoded])[0]

            # ── Step 6: Get probabilities for all 3 classes ───
            # classes order: ['High', 'Low', 'Medium'] (alphabetical)
            proba   = bundle['model'].predict_proba(input_df)[0]
            classes = bundle['label_encoder'].classes_

            prob_dict   = {cls: round(float(proba[i]) * 100, 2)
                          for i, cls in enumerate(classes)}
            prob_high   = prob_dict.get('High',   0)
            prob_medium = prob_dict.get('Medium', 0)
            prob_low    = prob_dict.get('Low',    0)
            confidence  = prob_dict.get(prediction, 0)

            # ── Step 7: Pull display data from bundle ─────────
            color    = bundle['class_colors'][prediction]
            icon     = bundle['class_icons'][prediction]
            advice   = bundle['class_advice'][prediction]
            label_id = bundle['class_labels_id'][pred_encoded]

            print(f"[PREDICT] {prediction} ({confidence}%) "
                  f"High:{prob_high} Med:{prob_medium} Low:{prob_low}")

        except KeyError as e:
            error_message = f"Input tidak lengkap — field hilang: {e}"
            print(f"[ERROR] KeyError: {e}")
        except ValueError as e:
            error_message = f"Nilai tidak valid — harap isi semua field: {e}"
            print(f"[ERROR] ValueError: {e}")
        except Exception as e:
            error_message = f"Terjadi kesalahan: {str(e)}"
            print(f"[ERROR] Unexpected error:")
            traceback.print_exc()

    return render_template(
        'cekpotensikangker.html',
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
# DEBUG ROUTE — visit /debug-bundle to verify model loaded OK
# Remove this before final production deployment
# ══════════════════════════════════════════════════════════════
@app.route('/debug-bundle')
def debug_bundle():
    return jsonify({
        'status'       : 'OK',
        'bundle_keys'  : list(bundle.keys()),
        'feature_names': bundle['feature_names'],
        'feature_count': bundle['feature_count'],
        'classes'      : list(bundle['label_encoder'].classes_),
        'class_labels' : {str(k): v for k, v in bundle['class_labels'].items()},
        'class_colors' : bundle['class_colors'],
        'test_accuracy': bundle['metadata']['test_accuracy'],
        'cv_mean'      : bundle['metadata']['cv_mean'],
        'overfit_gap'  : bundle['metadata']['overfit_gap'],
        'n_estimators' : bundle['metadata']['n_estimators'],
    })


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════
#port = int(os.environ.get('PORT', 5000))
#print(f'Starting Flask on port {port}')
#app.run(host='0.0.0.0', port=port, debug=False)
# Required for Vercel
app = app
