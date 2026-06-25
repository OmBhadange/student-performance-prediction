# 🎓 AI Student Performance Predictor

An end-to-end Machine Learning web application that predicts student exam scores based on academic, social, and personal factors.

🌐 **Live Demo:** https://student-performance-prediction-nslwcykuszugztasnyr6tz.streamlit.app/

---

# 🚀 Project Overview

The **AI Student Performance Predictor** is a Machine Learning-based web application that predicts a student's exam score using various performance-related factors such as study hours, attendance, sleep, previous scores, motivation level, and more.

The application helps educational institutions, teachers, and parents identify students who may require additional academic support.

---

# ✨ Features

✅ Predict student exam scores instantly

✅ Interactive and premium Streamlit dashboard

✅ Animated particle background

✅ Student performance gauge chart

✅ Performance progress indicator

✅ Analytics dashboard with visualizations

✅ Downloadable student report

✅ Beautiful modern UI

✅ Fully deployed online

---

# 🖥️ Live Application

🔗 **Live Demo:**

https://student-performance-prediction-nslwcykuszugztasnyr6tz.streamlit.app/

---

# 📂 Project Structure

```bash
student-performance-prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── models/
│   └── student_score_predictor.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── requirements.txt
│
├── README.md
│
└── .venv/
```

---

# 📊 Dataset Information

Dataset: **StudentPerformanceFactors.csv**

| Attribute       | Value      |
| --------------- | ---------- |
| Total Records   | 6607       |
| Total Features  | 20         |
| Target Variable | Exam_Score |

---

# 📈 Features Used

### Numerical Features

* Hours_Studied
* Attendance
* Sleep_Hours
* Previous_Scores
* Tutoring_Sessions
* Physical_Activity

### Categorical Features

* Parental_Involvement
* Access_to_Resources
* Extracurricular_Activities
* Motivation_Level
* Internet_Access
* Family_Income
* Teacher_Quality
* School_Type
* Peer_Influence
* Learning_Disabilities
* Parental_Education_Level
* Distance_from_Home
* Gender

---

# 🔄 Machine Learning Workflow

```text
Data Collection
       ↓
Data Cleaning
       ↓
Missing Value Handling
       ↓
Feature Encoding
       ↓
Exploratory Data Analysis
       ↓
Train-Test Split
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Saving
       ↓
Web Application Deployment
```

---

# 🧠 Machine Learning Model

### Model Used

* Linear Regression

### Libraries Used

* Scikit-Learn
* Pandas
* NumPy

---

# 📊 Model Performance

| Metric   | Score |
| -------- | ----- |
| MAE      | ~1.01 |
| MSE      | ~4.39 |
| R² Score | ~0.68 |

---

# 🛠️ Data Preprocessing

### Missing Values Handling

| Column                   | Technique |
| ------------------------ | --------- |
| Teacher_Quality          | Mode      |
| Parental_Education_Level | Mode      |
| Distance_from_Home       | Mode      |

### Encoding

Categorical features were converted into numerical values using label encoding techniques.

Examples:

```text
Male → 1
Female → 0

Yes → 1
No → 0
```

---

# 📈 Exploratory Data Analysis

Performed:

* Dataset inspection
* Missing value analysis
* Statistical summary
* Correlation analysis
* Feature distribution analysis

---

# ⚙️ Technologies Used

## Programming Language

* Python

## Machine Learning

* Scikit-Learn

## Data Analysis

* Pandas
* NumPy

## Visualization

* Plotly
* Matplotlib
* Seaborn

## Web Framework

* Streamlit

## Deployment

* Streamlit Community Cloud

## Version Control

* Git
* GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/OmBhadange/student-performance-prediction.git
```

Move into the project folder:

```bash
cd student-performance-prediction
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate virtual environment:

### Mac/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---

# 🌟 Future Improvements

* CSV Upload for batch predictions
* Student ranking system
* Grade prediction
* PDF report generation
* AI-based study recommendations
* Authentication system
* Database integration
* Teacher dashboard
* Student history tracking
* Advanced ML models (XGBoost, Random Forest)
* Deep Learning implementation

---

# 💡 Real-World Applications

* Schools
* Colleges
* Coaching Institutes
* EdTech Platforms
* Student Monitoring Systems

---

# 👨‍💻 Author

## Om Avinash Bhadange

🎓 B.Tech Artificial Intelligence & Data Science

🏫 Sanjivani University

🔗 GitHub: https://github.com/OmBhadange

---

# ⭐ Support

If you like this project, please give it a ⭐ on GitHub.
