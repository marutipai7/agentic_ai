import os
from config import Config
from flask_cors import CORS
from models import db, User
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Initialize DB and Migrate
db.init_app(app)
migrate = Migrate(app, db)

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

        hashed_password = generate_password_hash(password, method='sha256')

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
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
