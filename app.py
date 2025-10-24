import os
import matplotlib
import pandas as pd
from config import Config
from models import db, User
from flask_cors import CORS
from typing import Dict, Any
import matplotlib.pyplot as plt
from flask_migrate import Migrate
from plot_utils import _generate_plots
from werkzeug.security import generate_password_hash, check_password_hash
from preprocess_utils import _compute_overview_and_stats, _apply_preprocessing
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

matplotlib.use('Agg')
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
            save_dir = os.path.join('static', 'profile_pics')
            os.makedirs(save_dir, exist_ok=True)

            filename = profile_pic.filename
            profile_pic_path = filename  # <-- store only the filename
            profile_pic.save(os.path.join(save_dir, filename))

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
