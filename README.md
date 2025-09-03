# 📺 Netflix Data Analysis Dashboard

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Data Analysis Components](#data-analysis-components)
- [Machine Learning Models](#machine-learning-models)
- [Web Application Features](#web-application-features)
- [Technical Implementation](#technical-implementation)
- [Results & Insights](#results--insights)
- [Future Enhancements](#future-enhancements)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License & Acknowledgments](#license--acknowledgments)
- [Contact Information](#contact-information)

---

## Project Overview

### 🎯 Objective
A **Netflix Data Analysis Dashboard** built with **Streamlit** that provides insights into content distribution, growth trends, geographic patterns, and genre preferences. It also includes:
- A **content-based recommendation system**
- **Machine learning predictions** for content growth
- **WordCloud visualization** for text-based insights

### 🔧 Technology Stack
- **Backend**: Python 3.8+
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Text Analysis**: TF-IDF, WordCloud
- **Data Science**: Linear Regression, Cosine Similarity

### 🌟 Key Features
- Interactive dashboard with 8 analysis sections
- Content-based recommendation system
- Predictive modeling for future content growth
- Geographic and genre analysis
- WordCloud visualization

---

## Dataset Information

### 📊 Data Source
- **File**: `netflix1.csv`
- **Format**: CSV
- **Content**: Netflix titles metadata

### 📋 Schema
| Column       | Description                   | Example |
|--------------|-------------------------------|---------|
| show_id      | Unique identifier             | s1 |
| type         | Content type                  | Movie / TV Show |
| title        | Title name                    | Stranger Things |
| director     | Director name(s)              | Steven Spielberg |
| cast         | Cast members                  | Actor1, Actor2 |
| country      | Production country            | United States |
| date_added   | Date added to Netflix         | January 1, 2021 |
| release_year | Release year                  | 2021 |
| rating       | Content rating                | TV-MA |
| duration     | Duration                      | 120 min / 3 Seasons |
| listed_in    | Genres                        | Drama, Thriller |
| description  | Content description           | Plot summary... |

### 🔧 Preprocessing Steps
- Convert `date_added` to datetime
- Handle missing values (replace with `Unknown`)
- Extract year, month, day from dates
- Convert duration to minutes
- Process genres into lists

---

## Project Structure
```
netflix-analysis-project/
├── app.py               # Main Streamlit application
├── netflix1.csv         # Dataset file
├── Netflix.ipynb        # Jupyter notebook with analysis
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation

```

---

## Installation & Setup

### 🐍 Prerequisites
- Python 3.8+
- pip
- Git (optional)

### ⚙️ Steps
```bash
# Clone repo
git clone https://github.com/your-username/netflix-analysis-project.git
cd netflix-analysis-project

# Create virtual environment
python -m venv netflix_env
source netflix_env/bin/activate   # Windows: netflix_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 📋 requirements.txt
```
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
numpy>=1.24.0
scikit-learn>=1.3.0
wordcloud>=1.9.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

### 🚀 Run Application
```bash
streamlit run app.py
```
Open in browser: [http://localhost:8501](http://localhost:8501)

---

## Data Analysis Components
1. **Overview Analysis** → Dataset stats, Movies vs TV Shows, unique countries
2. **Content Distribution** → Pie chart, ratings distribution, duration histograms
3. **Geographic Analysis** → Top content-producing countries, bar charts
4. **Genre Analysis** → Top genres, combinations, distribution per title
5. **Content Growth** → Time-series trends, seasonal patterns
6. **WordCloud** → Frequent words in titles, directors, countries, genres

---

## Machine Learning Models

### 🎯 Content-Based Recommendation System
- **Algorithm**: TF-IDF + Cosine Similarity
- **Features**: Title, director, genres, country
- **Output**: Top-N recommendations

### 📊 Linear Regression (Growth Prediction)
- **Input**: Year
- **Target**: Content additions per year
- **Output**: Predicted future growth trends

---

## Web Application Features
- **UI Layout**: Sidebar navigation, main content, responsive design
- **Navigation**: Overview, Distribution, Geographic, Genre, Growth, WordCloud, Recommendations, Predictions
- **Interactivity**: Dropdowns, sliders, multi-select filters
- **Visualizations**: Interactive charts, hover effects, zoom, export
- **Performance**: Cached data loading, optimized rendering

---

## Technical Implementation
- **Architecture**:
  - Streamlit UI → Pandas/NumPy → ML Models → Plotly/Matplotlib
- **Caching**: `@st.cache_data`
- **Core Functions**:
  - `load_data()` – Loads and preprocesses dataset
  - `get_recommendations()` – Generates recommendations
  - `train_prediction_model()` – Trains linear regression model

---

## Results & Insights
- **Content Distribution**: More movies than TV shows, mature audience focus
- **Geographic Trends**: US leads, followed by India and UK
- **Genres**: Drama & Comedy dominate, diverse international genres
- **Growth Trends**: Exponential growth 2015–2020, seasonal patterns
- **Predictions**: Linear growth expected, R² ≈ 0.7–0.9

---

## Future Enhancements
- **UI**: Dark mode, better filters, downloadable reports
- **Analytics**: Collaborative filtering, sentiment analysis, network analysis
- **ML**: Time series models (ARIMA, Prophet), classification & clustering
- **Deployment**: Cloud hosting, Docker, CI/CD

---

## Troubleshooting
- **Module not found** → `pip install <module>`
- **CSV file not found** → Place `netflix1.csv` in same directory
- **Encoding issues** → Use `encoding='utf-8'`
- **Slow loading** → Use `@st.cache_data`

---

## Contributing
1. Fork repo & create a branch
2. Add feature/fix with tests
3. Submit PR with detailed description

### 📋 Guidelines
- Follow **PEP 8**
- Add **docstrings & type hints**
- Write **unit tests**
- Update **documentation**

---

## License & Acknowledgments
- **License**: MIT
- **Acknowledgments**:
  - Dataset contributors & Kaggle community
  - Streamlit, Pandas, Plotly, Scikit-learn teams
  - Data science community inspiration

---

## Contact Information
- **Maintainer**: Amit Khotele
- **Email**: amitkhotele2@gmail.com
- **GitHub**: github.com/amitkhotele
