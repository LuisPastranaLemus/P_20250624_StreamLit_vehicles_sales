# Web Dashboard with Streamlit and Python

This repository contains a project designed to strengthen software engineering skills through hands-on tasks. It focuses on setting up Python environments (Conda), performing exploratory data analysis (EDA), and developing a web dashboard using Streamlit.

The project specifically involves analyzing a dataset of vehicle sales in the US to explore pricing trends, vehicle types, and other attributes through interactive visualizations.

## 🧠 Project Objective

The main goal is to provide more opportunities to practice typical software engineering tasks and enhance data skills.

Key tasks include:

- Setting up and managing a virtual Python environment (Conda).
- Performing basic data visualization with `pandas`, `plotly-express` and `streamlit`.
- Developing and deploying a web application using `streamlit`.

---

## 📁 Repository Structure

```
project-name/
│
├── Datasets/
│   └── vehicles_us.csv      # Dataset for vehicle sales ads
│
├── notebooks/
│   └── EDA.ipynb            # Jupyter notebook with initial data analysis
│
├── app.py                   # Streamlit web app
├── requirements.txt         # Required Python packages
│
├── .gitignore               # Python-specific Git ignore rules
└── README.md                # Project description and instructions
```

---

## ⚙️ Setup Instructions

1. **Create Python Virtual Environment**

   ```bash
   conda create --name base python=3.12
   conda activate base
   ```

2. **Install Required Libraries**

   ```bash
   conda install pandas plotly streamlit
   ```

3. **Create**

   ```text
   kaleido
   nbformat
   numpy
   pandas
   streamlit
   plotly==6.1.2
   plotly-express
   ```

4. **Clone the Repository & Open in VS Code**

   - Set the Python interpreter to virtual environment.

---

## 📊 Data

Provided dataset (`vehicles_us.csv`). Place it in the project root directory.

### About `vehicles_us.csv`

This dataset contains information about used vehicle listings in the US, including attributes such as price, model year, condition, and type. It is suitable for learning basic EDA techniques and visualization.

---

## 📈 EDA Notebook

Use the `notebooks/eda.ipynb` notebook to perform initial exploratory data analysis using `plotly-express (plotly, import plotly.express as px)`. This helps understand the dataset before building the app.

Suggested EDA tasks:

- Distribution of prices and mileage
- Comparison by vehicle type or condition
- Scatter plots for price vs. model year or mileage
- Price Trends Over Time: Explore how vehicle prices change by model year or over calendar years if such data is available.
- Condition Impact: Analyze how the vehicle condition affects pricing across different models or brands.
- Brand vs. Price: Compare average prices between brands to identify high-end vs. economy segments.
- Mileage vs. Price (Depreciation Curve): Use scatter plots and trend lines to see how mileage impacts vehicle value.
- Top Models Sold: Use a bar chart to display the most listed models or makes.
- Transmission Type Analysis: See how manual vs. automatic transmission affects price and popularity.
- Fuel Type Preference: Compare price distributions across fuel types (gas, diesel, electric, hybrid).
- Correlation Analysis of Vehicle Features: examine the linear relationships between key vehicle attributes and listing price,includes both numerical variables and encoded categorical features such as vehicle condition, fuel type, and transmission type..

---

## 🌐 Streamlit Web App

The web dashboard (`app.py`) should:

- Display a header using `st.header()`.
- Include a button or checkbox to generate visualizations (histograms, scatter plots) using `plotly-express`.
- Show charts using `st.plotly_chart()`.

Run the app locally with:

```bash
streamlit run app.py
```

---

## 🚀 Deployment with Render

1. Connect GitHub to [Render](https://render.com).
2. Create a new web service and connect the repository.
3. Use the following settings:
   - **Build Command**: `pip install --upgrade pip && pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py`
4. Access app at: `https://p-20250624-streamlit-vehicles-sales.onrender.com`

Note: The app may take a few minutes to wake up if inactive.

---

## 📄 License

This project is open-source and available for any educational or demonstration purposes.

---

🛠️ Tools & Libraries

- Python 3.11
- pandas, NumPy, IPython.display, scipy, streamlit, ipywidgets, plotly.express, plotly
- Streamlit
- Jupyter Notebook
- Git & GitHub for version control

---

## 📌 Notes

This project is part of a personal learning portfolio focused on developing strong skills in data analysis, statistical thinking, and communication of insights. Constructive feedback is welcome.

---

## 👤 Author   
##### Luis Sergio Pastrana Lemus   
##### Engineer pivoting into Data Science | Passionate about insights, structure, and solving real-world problems with data.   
##### [GitHub Profile](https://github.com/LuisPastranaLemus)   
##### 📍 Querétaro, México     
##### 📧 Contact: luis.pastrana.lemus@engineer.com   
---

