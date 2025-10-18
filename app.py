import os
import io
import base64
from typing import Dict, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer

from config import Config
from flask_cors import CORS
from models import db, User
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Initialize DB and Migrate
db.init_app(app)
migrate = Migrate(app, db)

# In-memory dataframe store keyed by user_id
USER_DATAFRAMES: Dict[int, pd.DataFrame] = {}


def _get_user_df() -> pd.DataFrame | None:
    user_id = session.get('user_id')
    if user_id is None:
        return None
    return USER_DATAFRAMES.get(user_id)


def _set_user_df(df: pd.DataFrame) -> None:
    user_id = session.get('user_id')
    if user_id is None:
        return
    USER_DATAFRAMES[user_id] = df


def _figure_to_base64() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _compute_overview_and_stats(df: pd.DataFrame) -> Dict[str, Any]:
    overview = {
        'total_rows': int(df.shape[0]),
        'total_columns': int(df.shape[1]),
        'missing_values': int(df.isna().sum().sum()),
        'numeric_columns': int(df.select_dtypes(include=[np.number]).shape[1]),
    }

    column_info = []
    for col in df.columns:
        series = df[col]
        missing_percent = float(series.isna().mean() * 100.0)
        dtype_str = str(series.dtype)
        unique_values = int(series.nunique(dropna=True))
        column_info.append({
            'name': col,
            'dtype': dtype_str,
            'missing_percent': missing_percent,
            'unique_values': unique_values,
        })

    numeric_df = df.select_dtypes(include=[np.number])
    stats: Dict[str, Dict[str, float]] = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().to_dict()
        # transpose to {col: {metric: value}}
        for col, metrics in desc.items():
            stats[col] = {}
            for metric, value in metrics.items():
                # Cast numpy types to python native for JSON
                if pd.isna(value):
                    stats[col][metric] = None
                else:
                    stats[col][metric] = float(value)

    return {
        'data_overview': overview,
        'column_info': column_info,
        'statistics': stats,
    }


def _generate_plots(df: pd.DataFrame) -> Dict[str, Any]:
    plots: Dict[str, Any] = {}
    numeric_df = df.select_dtypes(include=[np.number])

    # Correlation heatmap
    if numeric_df.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        corr = numeric_df.corr(numeric_only=True)
        sns.heatmap(corr, cmap='coolwarm', annot=False)
        plots['heatmap'] = _figure_to_base64()

    # Histograms and Boxplots for up to 6 numeric columns
    cols = list(numeric_df.columns)[:6]
    histograms: Dict[str, str] = {}
    boxplots: Dict[str, str] = {}
    for col in cols:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        # Distribution plot (hist + kde)
        plt.figure(figsize=(5, 3))
        sns.histplot(series, kde=True, bins=30, color='#6366F1')
        plt.title(f'Distribution - {col}')
        histograms[col] = _figure_to_base64()

        # Boxplot
        plt.figure(figsize=(5, 2.5))
        sns.boxplot(x=series, color='#22C55E')
        plt.title(f'Boxplot - {col}')
        boxplots[col] = _figure_to_base64()

    if histograms:
        plots['histograms'] = histograms
    if boxplots:
        plots['boxplots'] = boxplots

    return plots


def _apply_preprocessing(df: pd.DataFrame, steps: list[str]) -> pd.DataFrame:
    result = df.copy()
    numeric_cols = list(result.select_dtypes(include=[np.number]).columns)
    cat_cols = list(result.select_dtypes(exclude=[np.number]).columns)

    if 'drop_missing' in steps:
        result = result.dropna()
        numeric_cols = list(result.select_dtypes(include=[np.number]).columns)
        cat_cols = list(result.select_dtypes(exclude=[np.number]).columns)

    if 'fill_mean' in steps and numeric_cols:
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())

    if 'fill_median' in steps and numeric_cols:
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())

    if 'fill_mode' in steps and cat_cols:
        for c in cat_cols:
            mode_val = result[c].mode(dropna=True)
            if not mode_val.empty:
                result[c] = result[c].fillna(mode_val.iloc[0])

    # One-hot encoding
    if 'one_hot' in steps and cat_cols:
        result = pd.get_dummies(result, columns=cat_cols, drop_first=True)
        numeric_cols = list(result.select_dtypes(include=[np.number]).columns)

    # Scaling / Normalization (apply to numeric columns only)
    numeric_cols = list(result.select_dtypes(include=[np.number]).columns)
    if numeric_cols:
        if 'standardize' in steps:
            scaler = StandardScaler()
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        if 'minmax' in steps:
            scaler = MinMaxScaler()
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        if 'robust' in steps:
            scaler = RobustScaler()
            result[numeric_cols] = scaler.fit_transform(result[numeric_cols])
        if 'normalize_l2' in steps:
            normalizer = Normalizer(norm='l2')
            result[numeric_cols] = normalizer.fit_transform(result[numeric_cols])

    return result

# Routes
@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        mobile_number = request.form['mobile_number']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        profile_pic = request.files.get('profile_pic')
        address = request.form['address']
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        pincode = request.form['pincode']
        password = request.form['password']

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        profile_pic_path = None
        if profile_pic:
            os.makedirs('static/profile_pics', exist_ok=True)
            profile_pic_path = os.path.join('static/profile_pics', profile_pic.filename)
            profile_pic.save(profile_pic_path)

        new_user = User(
            email=email,
            mobile_number=mobile_number,
            first_name=first_name,
            last_name=last_name,
            profile_pic_path=profile_pic_path,
            address=address,
            city=city,
            state=state,
            country=country,
            pincode=pincode,
            password_hash=hashed_password
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            return redirect(url_for('dashboard'))  # redirect to dashboard
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for('login'))

    # GET request → render login page
    return render_template('login.html')  # login.html does NOT expect 'user'


@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Fetch user from session
    user = User.query.get(session['user_id'])
    df = _get_user_df()
    template_kwargs: Dict[str, Any] = {'user': user}
    if df is not None:
        computed = _compute_overview_and_stats(df)
        template_kwargs.update(computed)
    return render_template('dashboard.html', **template_kwargs)  # dashboard.html expects 'user'


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
def upload():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No file provided', 'danger')
        return redirect(url_for('dashboard'))
    try:
        filename_lower = file.filename.lower()
        if filename_lower.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename_lower.endswith('.xlsx') or filename_lower.endswith('.xls'):
            df = pd.read_excel(file)
        else:
            flash('Unsupported file format. Please upload CSV or Excel.', 'danger')
            return redirect(url_for('dashboard'))
        _set_user_df(df)
        flash('File uploaded successfully.', 'success')
    except Exception as e:
        flash(f'Failed to read file: {e}', 'danger')
    return redirect(url_for('dashboard'))


@app.route('/api/analytics', methods=['GET'])
def api_analytics():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    df = _get_user_df()
    if df is None:
        return jsonify({'error': 'No dataset uploaded yet'}), 400
    computed = _compute_overview_and_stats(df)
    plots = _generate_plots(df)
    return jsonify({**computed, 'plots': plots})


@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    df = _get_user_df()
    if df is None:
        return jsonify({'error': 'No dataset uploaded yet'}), 400

    # Read steps from either form or JSON
    steps = request.json.get('steps') if request.is_json else request.form.getlist('preprocessing')
    if not steps:
        steps = []
    try:
        new_df = _apply_preprocessing(df, steps)
        _set_user_df(new_df)
        computed = _compute_overview_and_stats(new_df)
        plots = _generate_plots(new_df)
        return jsonify({'success': True, **computed, 'plots': plots})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
