from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
import csv
import os
import joblib
import pandas as pd
import traceback

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# ══════════════════════════════════════════════════════════════
# LOAD CSV DATA — loaded once at startup
# ══════════════════════════════════════════════════════════════
DATA_DIR = 'excelfiles'

gejala_map = {}
with open(os.path.join(DATA_DIR, 'Kode_Gejala_Laptop.csv'), newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        gejala_map[row['kode_gejala']] = row['nama_gejala']

kerusakan_map = {}
with open(os.path.join(DATA_DIR, 'Kode_Kerusakan_Laptop.csv'), newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        kerusakan_map[row['kode_kerusakan']] = row['kerusakan']

rules_list = []
with open(os.path.join(DATA_DIR, 'Rule_Kerusakan_Laptop.csv'), newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        codes = [c.strip() for c in row['kode_gejala'].split(',') if c.strip()]
        rules_list.append({'kode_kerusakan': row['kode_kerusakan'], 'kode_gejala_list': codes})

perbaikan_map = {}
with open(os.path.join(DATA_DIR, 'Kode_perbaikan_laptop.csv'), newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        kode_p  = row.get('kode_perbaikan') or row.get('Kode_perbaikan')
        langkah = row.get('langkah_perbaikan')
        if kode_p and langkah:
            perbaikan_map[kode_p] = langkah

perbaikan_rules = {}
with open(os.path.join(DATA_DIR, 'Rule_perbaikan.csv'), newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        kode_k = row.get('kode_kerusakan')
        kode_p = row.get('kode_perbaikan') or row.get('Kode_perbaikan')
        if kode_k and kode_p:
            perbaikan_rules.setdefault(kode_k, []).append(kode_p)


# ══════════════════════════════════════════════════════════════
# LOAD CANCER MODEL BUNDLE — loaded once at startup
# ══════════════════════════════════════════════════════════════
bundle = joblib.load('trainedmodel/cancer_model_bundle.joblib')
print(f"[OK] Model bundle loaded")
print(f"[OK] Test accuracy : {bundle['metadata']['test_accuracy']:.2%}")
print(f"[OK] Features      : {bundle['feature_count']}")
print(f"[OK] Classes       : {list(bundle['label_encoder'].classes_)}")


# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════
def build_rule_data():
    rd = []
    for rule in rules_list:
        kode           = rule['kode_kerusakan']
        kerusakan_name = kerusakan_map.get(kode, kode)
        gejala_names   = [gejala_map.get(c, c) for c in rule['kode_gejala_list']]
        rd.append({
            'kode_kerusakan': kode,
            'kerusakan_name': kerusakan_name,
            'gejala_list'   : gejala_names
        })
    return rd


# ══════════════════════════════════════════════════════════════
# ROUTES — General
# ══════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return redirect(url_for('dashboard'))


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dictionary')
def dictionary():
    return render_template('dictionary.html', rules=build_rule_data())


@app.route('/diagnosis', methods=['GET', 'POST'])
def diagnose():
    if 'username' not in session:
        flash('Please log in.', 'warning')
        return redirect(url_for('login'))

    result   = None
    selected = request.form.getlist('gejala') if request.method == 'POST' else []
    matches  = []

    for rule in rules_list:
        if set(rule['kode_gejala_list']).issubset(set(selected)):
            matches.append(rule['kode_kerusakan'])

    if matches:
        best   = max(set(matches), key=matches.count)
        result = f"{best} - {kerusakan_map.get(best, best)}"
    elif request.method == 'POST':
        result = "No matching diagnosis found."

    return render_template('diagnosis.html',
                           gejala_list=[{'kode_gejala': k, 'nama_gejala': v}
                                        for k, v in gejala_map.items()],
                           selected_gejala=selected,
                           result=result,
                           rules=build_rule_data())


@app.route('/get_steps/<kode_kerusakan>')
def get_steps(kode_kerusakan):
    codes = perbaikan_rules.get(kode_kerusakan, [])
    steps = [perbaikan_map.get(c, f"Step for {c} not found") for c in codes]
    return jsonify(steps=steps)


@app.route('/edit-dataset')
def edit_dataset():
    if 'username' not in session:
        return redirect(url_for('login'))

    def read_csv(fname):
        path = os.path.join(DATA_DIR, fname)
        try:
            with open(path, newline='', encoding='utf-8') as f:
                return list(csv.reader(f))
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")
            return []

    return render_template('edit_dataset.html',
                           gejala         = read_csv('Kode_Gejala_Laptop.csv'),
                           kerusakan      = read_csv('Kode_Kerusakan_Laptop.csv'),
                           perbaikan      = read_csv('Kode_perbaikan_laptop.csv'),
                           rule_kerusakan = read_csv('Rule_Kerusakan_Laptop.csv'),
                           rule_perbaikan = read_csv('Rule_perbaikan.csv'))


@app.route('/edit-row')
def edit_row():
    return render_template('edit_row.html',
                           dataset = request.args.get('dataset'),
                           row_id  = request.args.get('id'))


@app.route('/update_row', methods=['POST'])
def update_row():
    dataset = request.form.get('dataset')
    row_id  = request.form.get('id')

    updated_row, i = [], 0
    while True:
        val = request.form.get(f'col{i}')
        if val is None:
            break
        updated_row.append(val)
        i += 1

    dataset_map = {
        'gejala'        : os.path.join(DATA_DIR, 'Kode_Gejala_Laptop.csv'),
        'kerusakan'     : os.path.join(DATA_DIR, 'Kode_Kerusakan_Laptop.csv'),
        'perbaikan'     : os.path.join(DATA_DIR, 'Kode_perbaikan_laptop.csv'),
        'rule_kerusakan': os.path.join(DATA_DIR, 'Rule_Kerusakan_Laptop.csv'),
        'rule_perbaikan': os.path.join(DATA_DIR, 'Rule_perbaikan.csv'),
    }
    filename = dataset_map.get(dataset)
    if not filename:
        flash('Invalid dataset type.')
        return redirect(url_for('edit_dataset'))

    rows, found = [], False
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row and row[0] == row_id:
                    rows.append(updated_row)
                    found = True
                else:
                    rows.append(row)
    except FileNotFoundError:
        flash('Data file not found.')
        return redirect(url_for('edit_dataset'))

    if not found:
        flash('Row not found.')
        return redirect(url_for('edit_dataset'))

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)

    flash(f'Successfully updated row in {dataset}.')
    return redirect(url_for('edit_dataset'))


@app.route('/add-row', methods=['POST'])
def add_row():
    dataset    = request.form.get('dataset')
    new_row, i = [], 0
    while True:
        val = request.form.get(f'col{i}')
        if val is None:
            break
        new_row.append(val)
        i += 1

    dataset_map = {
        'gejala'        : os.path.join(DATA_DIR, 'Kode_Gejala_Laptop.csv'),
        'kerusakan'     : os.path.join(DATA_DIR, 'Kode_Kerusakan_Laptop.csv'),
        'perbaikan'     : os.path.join(DATA_DIR, 'Kode_perbaikan_laptop.csv'),
        'rule_kerusakan': os.path.join(DATA_DIR, 'Rule_Kerusakan_Laptop.csv'),
        'rule_perbaikan': os.path.join(DATA_DIR, 'Rule_perbaikan.csv'),
    }
    filename = dataset_map.get(dataset)
    if not filename:
        flash('Invalid dataset.')
        return redirect(url_for('edit_dataset'))

    try:
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(new_row)
        flash(f'Successfully added row to {dataset}.')
    except Exception as e:
        flash(f'Failed to add row: {e}')

    return redirect(url_for('edit_dataset'))


# ══════════════════════════════════════════════════════════════
# CANCER PREDICTION ROUTE
# ══════════════════════════════════════════════════════════════

@app.route('/cekpotensikangker', methods=['GET', 'POST'])
def cekpotensikangker():

    # Feature list from bundle — guaranteed correct order
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
            # ── Step 1: Collect form values ───────────────────
            input_data = [int(request.form[f]) for f in features]

            # ── Step 2: Clip to valid training ranges ─────────
            input_clipped = []
            for i, val in enumerate(input_data):
                if i == 0:    input_clipped.append(val)                 # Age
                elif i == 1:  input_clipped.append(max(1, min(2, val))) # Gender
                else:         input_clipped.append(max(1, min(8, val))) # Features 1-8

            # ── Step 3: Build DataFrame with feature names ────
            # This prevents the sklearn warning about feature names
            # and ensures features are in exactly the right order
            input_df = pd.DataFrame([input_clipped], columns=features)

            # ── Step 4: Predict ───────────────────────────────
            # Returns numpy.int64 — MUST convert to plain int
            # otherwise bundle dict lookup fails silently
            pred_encoded = int(bundle['model'].predict(input_df)[0])
            # pred_encoded is now a plain Python int: 0, 1, or 2

            # ── Step 5: Decode to text ────────────────────────
            # inverse_transform converts: 0→'High', 1→'Low', 2→'Medium'
            prediction = bundle['label_encoder'].inverse_transform([pred_encoded])[0]

            # ── Step 6: Get probabilities ──────────────────────
            proba   = bundle['model'].predict_proba(input_df)[0]
            classes = bundle['label_encoder'].classes_  # ['High','Low','Medium']

            prob_dict   = {cls: round(float(proba[i]) * 100, 2)
                          for i, cls in enumerate(classes)}
            prob_high   = prob_dict.get('High',   0)
            prob_medium = prob_dict.get('Medium', 0)
            prob_low    = prob_dict.get('Low',    0)
            confidence  = prob_dict.get(prediction, 0)

            # ── Step 7: Pull display values from bundle ───────
            # pred_encoded is now plain int — dict lookup works correctly
            color    = bundle['class_colors'][prediction]
            icon     = bundle['class_icons'][prediction]
            advice   = bundle['class_advice'][prediction]
            label_id = bundle['class_labels_id'][pred_encoded]

            print(f"[PREDICT] {prediction} ({confidence}%) — High:{prob_high} Med:{prob_medium} Low:{prob_low}")

        except KeyError as e:
            error_message = f"Input tidak lengkap — field hilang: {e}"
            print(f"[ERROR] KeyError: {e}")
        except ValueError as e:
            error_message = f"Nilai tidak valid — harap isi semua field: {e}"
            print(f"[ERROR] ValueError: {e}")
        except Exception as e:
            # Print full traceback to terminal so you can see exact error
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
# DEBUG ROUTE — visit localhost:5000/debug-bundle to verify
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
port = int(os.environ.get('PORT', 5000))
print(f'Starting Flask on port {port}')
app.run(host='0.0.0.0', port=port, debug=True)