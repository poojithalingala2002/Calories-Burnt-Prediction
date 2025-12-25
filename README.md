<!DOCTYPE html>
<html lang="en">
<body>

<h1>🔥 Calories Burnt Prediction – Machine Learning Project</h1>

<hr>

<h2>📌 Project Overview</h2>
<p>
This project focuses on predicting the <b>number of calories burnt</b> during physical activities using
<b>Machine Learning regression techniques</b>.
The solution includes an end-to-end ML pipeline covering data preprocessing, feature engineering,
model training, evaluation, and deployment using a <b>Flask web application</b>.
</p>

<p>
The model predicts calories burnt based on user and exercise parameters such as
<b>age, height, weight, duration, heart rate, body temperature, and gender</b>.
</p>

<hr>

<h2>🚀 Key Features</h2>
<ul>
    <li>End-to-end Machine Learning pipeline</li>
    <li>Robust data preprocessing and feature engineering</li>
    <li>Outlier handling and variable transformation</li>
    <li>Feature scaling using StandardScaler</li>
    <li>Model training and evaluation</li>
    <li>Model persistence using Pickle</li>
    <li>Flask-based web application for real-time prediction</li>
</ul>

<hr>

<h2>📂 Project Structure</h2>
<pre>
Calories-Burnt-Prediction/
│
├── app.py
├── main.py
├── allalgo.py
├── cat_to_num.py
├── data_scaling.py
├── feat_select.py
├── missing_value_handling.py
├── variable_trans_out_handle.py
├── log_code.py
├── requirements.txt
│
├── calories.csv
├── exercise.csv
│
├── reg_model.pkl
├── scalar.pkl
│
├── templates/
│   └── index.html
│
└── logs/
</pre>

<hr>

<h2>📊 Dataset Description</h2>
<p>
The project uses two datasets:
</p>
<ul>
    <li><b>calories.csv</b> – contains calories burnt values</li>
    <li><b>exercise.csv</b> – contains user and exercise-related information</li>
</ul>

<h3>Input Features</h3>
<ul>
    <li>Age</li>
    <li>Height</li>
    <li>Weight</li>
    <li>Exercise Duration</li>
    <li>Heart Rate</li>
    <li>Body Temperature</li>
    <li>Gender</li>
</ul>

<h3>Target Variable</h3>
<ul>
    <li>Calories Burnt</li>
</ul>

<hr>

<h2>⚙️ Data Preprocessing Pipeline</h2>

<h3>1️⃣ Data Merging</h3>
<p>
The datasets are merged using <b>User_ID</b> to align exercise details with calorie output.
</p>

<h3>2️⃣ Missing Value Handling</h3>
<p>
Missing values are handled using <b>Random Sample Imputation</b>.
Null values are replaced with randomly sampled existing values from the same feature
to preserve data distribution.
</p>

<h3>3️⃣ Variable Transformation & Outlier Handling</h3>
<ul>
    <li>Log transformation (<code>log1p</code>) to reduce skewness</li>
    <li>Quantile capping (1%–99%) to treat extreme outliers</li>
    <li>KDE and boxplots generated before and after transformation</li>
</ul>

<h3>4️⃣ Categorical Encoding</h3>
<p>
Categorical features such as <b>Gender</b> are converted into numeric format using
<b>One-Hot Encoding</b>.
</p>

<h3>5️⃣ Feature Scaling</h3>
<p>
Numerical features are scaled using <b>StandardScaler</b>.
The scaler is saved as <code>scalar.pkl</code> to ensure consistency during deployment.
</p>

<h3>6️⃣ Feature Selection</h3>
<ul>
    <li>Constant and quasi-constant feature removal</li>
    <li>Pearson correlation hypothesis testing</li>
</ul>

<p>
(Note: Feature selection is optional and can be enabled as required.)
</p>

<hr>

<h2>🤖 Model Training & Evaluation</h2>

<h3>Algorithm Used</h3>
<ul>
    <li>Linear Regression</li>
</ul>

<h3>Evaluation Metrics</h3>
<ul>
    <li>R² Score</li>
    <li>Mean Squared Error (MSE)</li>
</ul>

<h3>Model Persistence</h3>
<p>
The trained model is saved as:
</p>
<pre>
reg_model.pkl
</pre>

<hr>

<h2>🌐 Flask Web Application</h2>
<p>
A Flask-based web application is built to provide real-time calorie prediction.
</p>

<h3>Application Workflow</h3>
<ol>
    <li>User enters input values through the UI</li>
    <li>Inputs are scaled using the saved scaler</li>
    <li>The trained model predicts calories burnt</li>
    <li>Prediction is displayed on the webpage</li>
</ol>

<h3>User Inputs</h3>
<ul>
    <li>Age</li>
    <li>Height</li>
    <li>Weight</li>
    <li>Duration</li>
    <li>Heart Rate</li>
    <li>Body Temperature</li>
    <li>Gender</li>
</ul>

<hr>

<h2>🖥️ How to Run the Project</h2>

<h3>1️⃣ Clone the Repository</h3>
<pre>
git clone https://github.com/Balavenu123/Calories-Burnt-Prediction.git
cd Calories-Burnt-Prediction
</pre>

<h3>2️⃣ Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>3️⃣ Train the Model</h3>
<pre>
python main.py
</pre>

<h3>4️⃣ Run the Flask App</h3>
<pre>
python app.py
</pre>

<p>
Open your browser and navigate to:
</p>
<pre>
http://127.0.0.1:5000/
</pre>

<hr>

<h2>📈 Results</h2>
<ul>
    <li>The model successfully predicts calories burnt</li>
    <li>Exercise duration and heart rate strongly influence calorie expenditure</li>
    <li>The pipeline is modular, reusable, and deployment-ready</li>
</ul>

<hr>

<h2>🛠️ Technologies Used</h2>
<ul>
    <li>Python</li>
    <li>Pandas, NumPy</li>
    <li>Scikit-learn</li>
    <li>Matplotlib, Seaborn</li>
    <li>Flask</li>
    <li>Pickle</li>
</ul>

<hr>

<h2>📌 Future Enhancements</h2>
<ul>
    <li>Implement advanced models (Random Forest, XGBoost)</li>
    <li>Add cross-validation</li>
    <li>Improve UI design</li>
    <li>Deploy on cloud platforms</li>
    <li>Create REST API endpoints</li>
</ul>

<hr>

<h2>👤 Author</h2>
<p>
<b>Bala Venu Balineni</b><br>
Machine Learning & Data Science Enthusiast
</p>

</body>
</html>
