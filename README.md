# Agentic AI Data Analytics Platform
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/marutipai7/agentic_ai)

This repository contains a web-based data analytics and preprocessing platform built with a Flask backend. It provides a user-friendly interface for uploading datasets, performing exploratory data analysis, applying various preprocessing steps, and visualizing the results. The system is designed with a foundation for integrating future agentic AI capabilities using Langchain.

## Features

- **User Authentication:** Secure user registration, login, and session management.
- **Profile Management:** Users have profiles with personal details and profile pictures.
- **Dataset Upload:** Supports uploading data in CSV and Excel formats.
- **Automated Data Analysis:**
    - **Data Overview:** Instantly view total rows, columns, missing values, and column types.
    - **Statistical Summary:** Get descriptive statistics (mean, std, min, max, etc.) for all numeric columns.
- **Data Visualization:**
    - **Histograms:** View the distribution of numeric features.
    - **Boxplots:** Identify outliers and data spread for numeric features.
    - **Correlation Heatmap:** Understand relationships between numeric variables.
- **Data Preprocessing:** Apply a suite of preprocessing techniques:
    - Missing value imputation (mean, median, mode).
    - Outlier treatment using the IQR method.
    - Feature scaling (Standardization, Min-Max, Robust).
    - Categorical data encoding (One-Hot Encoding).
- **Database Connectivity:** Connect to PostgreSQL, MySQL, or MongoDB databases to inspect schemas and tables.
- **Data Export:** Download the processed dataset as a CSV file.

## Tech Stack

- **Backend:** Flask, Flask-SQLAlchemy, Flask-Migrate
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Database:** PostgreSQL (for user management)
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** HTML, Tailwind CSS (via CDN), JavaScript

## Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL database

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/marutipai7/agentic_ai.git
    cd agentic_ai
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your database configuration and a secret key. Use the following template:

    ```env
    SECRET_KEY='your_super_secret_key'
    DB_USER='your_postgres_user'
    DB_PASSWORD='your_postgres_password'
    DB_HOST='localhost'
    DB_PORT='5432'
    DB_NAME='agentic_ai'
    ```

5.  **Initialize and migrate the database:**
    Make sure your PostgreSQL server is running and the database specified in `.env` exists.
    ```bash
    flask db init  # Run only if the 'migrations' folder doesn't exist
    flask db migrate -m "Initial migration"
    flask db upgrade
    ```

6.  **Run the application:**
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

## Usage

1.  Navigate to `http://127.0.0.1:5000/register` to create a new user account.
2.  Log in with your credentials.
3.  On the dashboard, use the "Upload Dataset" section to upload a CSV or Excel file.
4.  Once uploaded, the dashboard will automatically display the data overview, column information, statistical summaries, and visualizations.
5.  Use the "Data Preprocessing" section to select and apply various data cleaning and transformation steps. The analytics will refresh to reflect the changes.
6.  Click "Download CSV" to save the processed data to your local machine.

## Core File Structure

```
├── app.py              # Main Flask application with routes and API endpoints
├── config.py           # Configuration setup loading from .env
├── models.py           # SQLAlchemy User model
├── plot_utils.py       # Helper functions for generating Matplotlib/Seaborn plots
├── preprocess_utils.py # Functions for computing stats and applying preprocessing
├── requirements.txt    # Python dependencies
└── templates/
    ├── dashboard.html  # Main user dashboard
    ├── login.html      # User login page
    └── register.html   # User registration page