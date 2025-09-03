import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Page configuration
st.set_page_config(page_title="Netflix Data Analysis Dashboard", page_icon="📺", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("netflix1.csv")
    
    # Convert date_added to datetime
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    
    # Fill missing values
    df['country'] = df['country'].fillna("Unknown")
    df['rating'] = df['rating'].fillna("Unknown")
    df['duration'] = df['duration'].fillna("Unknown")
    df['director'] = df['director'].fillna("Unknown")
    df['listed_in'] = df['listed_in'].fillna("Unknown")
    
    # Feature Engineering
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    df['day_added'] = df['date_added'].dt.day
    
    # Number of genres
    df['num_genres'] = df['listed_in'].apply(lambda x: len(str(x).split(",")))
    
    # Duration in minutes (for movies)
    def convert_duration(x):
        if "min" in str(x):
            return int(str(x).replace("min","").strip())
        elif "Season" in str(x):
            return int(str(x).split()[0]) * 60  # approx. 1 season = 60 min
        else:
            return np.nan
    
    df['duration_mins'] = df['duration'].apply(convert_duration)
    
    # Flag long movies (>120 min)
    df['long_movie'] = df['duration_mins'].apply(lambda x: 1 if pd.notna(x) and x > 120 else 0)
    
    return df

# Load data
df = load_data()

# Title and sidebar
st.title("📺 Netflix Data Analysis Dashboard")
st.sidebar.title("Navigation")

# Sidebar options
analysis_option = st.sidebar.selectbox(
    "Choose Analysis",
    ["Overview", "Content Distribution", "Geographic Analysis", "Genre Analysis", 
     "Content Growth", "WordCloud", "Recommendations", "ML Predictions"]
)

# Overview Section
if analysis_option == "Overview":
    st.header("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Titles", len(df))
    with col2:
        st.metric("Movies", len(df[df['type'] == 'Movie']))
    with col3:
        st.metric("TV Shows", len(df[df['type'] == 'TV Show']))
    with col4:
        st.metric("Countries", df['country'].nunique())
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Statistics")
    st.write(df.describe())

# Content Distribution
elif analysis_option == "Content Distribution":
    st.header("🎬 Content Distribution Analysis")
    
    # Movies vs TV Shows
    col1, col2 = st.columns(2)
    
    with col1:
        type_count = df['type'].value_counts()
        fig1 = px.pie(names=type_count.index, values=type_count.values, 
                     title="Movies vs TV Shows Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Rating distribution
        rating_count = df['rating'].value_counts().head(10)
        fig2 = px.bar(x=rating_count.index, y=rating_count.values,
                     title="Content by Rating")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Duration analysis
    st.subheader("Duration Analysis")
    movie_durations = df[df['type'] == 'Movie']['duration_mins'].dropna()
    
    if len(movie_durations) > 0:
        fig3 = px.histogram(x=movie_durations, nbins=30, 
                           title="Movie Duration Distribution (Minutes)")
        st.plotly_chart(fig3, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Average Movie Duration", f"{movie_durations.mean():.0f} min")
        with col4:
            long_movies = df['long_movie'].sum()
            st.metric("Long Movies (>120 min)", long_movies)

# Geographic Analysis
elif analysis_option == "Geographic Analysis":
    st.header("🌍 Geographic Content Analysis")
    
    # Top countries
    top_countries = df['country'].value_counts().head(15)
    fig1 = px.bar(x=top_countries.values, y=top_countries.index, 
                 orientation='h', title="Top 15 Content Producing Countries")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Content by country and type
    country_type = df.groupby(['country', 'type']).size().reset_index(name='count')
    country_type = country_type[country_type['country'].isin(top_countries.head(10).index)]
    
    fig2 = px.bar(country_type, x='country', y='count', color='type',
                 title="Content Type Distribution by Top Countries")
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

# Genre Analysis
elif analysis_option == "Genre Analysis":
    st.header("🎭 Genre Analysis")
    
    # Extract all genres
    genre_list = []
    for s in df['listed_in'].dropna():
        genre_list.extend([g.strip() for g in s.split(",")])
    
    genre_count = Counter(genre_list).most_common(20)
    
    # Top genres
    fig1 = px.bar(x=[c[1] for c in genre_count], y=[c[0] for c in genre_count],
                 orientation='h', title="Top 20 Genres on Netflix")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Genre combinations
    st.subheader("Genre Combinations")
    genre_combo = df['listed_in'].value_counts().head(15)
    fig2 = px.bar(x=genre_combo.values, y=genre_combo.index, 
                 orientation='h', title="Top 15 Genre Combinations")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Number of genres per title
    st.subheader("Number of Genres per Title")
    fig3 = px.histogram(df, x='num_genres', title="Distribution of Number of Genres per Title")
    st.plotly_chart(fig3, use_container_width=True)

# Content Growth
elif analysis_option == "Content Growth":
    st.header("📈 Content Growth Analysis")
    
    # Content added over years
    yearly_content = df['year_added'].value_counts().sort_index()
    
    fig1 = px.line(x=yearly_content.index, y=yearly_content.values,
                  title="Netflix Content Added Over Years", markers=True)
    fig1.update_layout(xaxis_title="Year", yaxis_title="Number of Titles Added")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Monthly trends
    st.subheader("Monthly Addition Patterns")
    monthly_content = df['month_added'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig2 = px.bar(x=[month_names[i-1] for i in monthly_content.index], 
                 y=monthly_content.values,
                 title="Content Added by Month")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Year-wise content by type
    yearly_type = df.groupby(['year_added', 'type']).size().reset_index(name='count')
    fig3 = px.line(yearly_type, x='year_added', y='count', color='type',
                  title="Content Growth by Type Over Years", markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# WordCloud
elif analysis_option == "WordCloud":
    st.header("☁️ WordCloud Analysis")
    
    # Generate WordCloud
    text = ' '.join(df['title'].astype(str)) + ' ' \
           + ' '.join(df['director'].astype(str)) + ' ' \
           + ' '.join(df['country'].astype(str)) + ' ' \
           + ' '.join(df['listed_in'].astype(str))
    
    # Create WordCloud
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white',
                          colormap='viridis',
                          max_words=200,
                          random_state=42).generate(text)
    
    # Display WordCloud
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    plt.title("Netflix Content WordCloud", fontsize=16, fontweight='bold')
    
    st.pyplot(fig)
    
    # Most common words
    st.subheader("Most Frequent Words")
    word_freq = wordcloud.words_
    word_df = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    word_df = word_df.head(20)
    
    fig2 = px.bar(word_df, x='Frequency', y='Word', orientation='h',
                 title="Top 20 Most Frequent Words")
    st.plotly_chart(fig2, use_container_width=True)

# Recommendations
elif analysis_option == "Recommendations":
    st.header("🎯 Content Recommendation System")
    
    # Prepare recommendation system
    @st.cache_data
    def prepare_recommendations():
        text_features = ['title', 'director', 'listed_in', 'country']
        df['combined'] = df[text_features].fillna('').agg(' '.join, axis=1)
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['combined'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return cosine_sim
    
    def get_recommendations(title, cosine_sim, n=10):
        if title not in df['title'].values:
            return ["Title not found in dataset."]
        
        idx = df[df['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = df.iloc[movie_indices][['title', 'type', 'listed_in', 'country']].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations
    
    cosine_sim = prepare_recommendations()
    
    # User interface
    selected_title = st.selectbox("Select a Movie/Show for Recommendations", 
                                 df['title'].dropna().unique())
    
    num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)
    
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(selected_title, cosine_sim, num_recommendations)
        
        if isinstance(recommendations, list):
            st.warning("Title not found!")
        else:
            st.subheader(f"Recommendations for: {selected_title}")
            
            # Display selected title info
            selected_info = df[df['title'] == selected_title].iloc[0]
            st.write(f"**Type:** {selected_info['type']}")
            st.write(f"**Genres:** {selected_info['listed_in']}")
            st.write(f"**Country:** {selected_info['country']}")
            
            st.subheader("Recommended Titles:")
            for idx, rec in recommendations.iterrows():
                with st.expander(f"{rec['title']} (Similarity: {rec['similarity_score']:.3f})"):
                    st.write(f"**Type:** {rec['type']}")
                    st.write(f"**Genres:** {rec['listed_in']}")
                    st.write(f"**Country:** {rec['country']}")

# ML Predictions
elif analysis_option == "ML Predictions":
    st.header("🤖 Machine Learning Predictions")
    
    st.subheader("Netflix Content Growth Prediction")
    
    # Prepare data for prediction
    df_clean = df.dropna(subset=['year_added'])
    yearly = df_clean['year_added'].value_counts().sort_index().reset_index()
    yearly.columns = ['year', 'count']
    
    # Train model
    X = yearly['year'].values.reshape(-1, 1)
    y = yearly['count'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    current_year = 2022
    prediction_years = st.slider("Predict until year:", current_year, 2035, 2030)
    
    future_years = np.array(range(current_year, prediction_years + 1)).reshape(-1, 1)
    predictions = model.predict(future_years)
    
    # Create prediction visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(x=yearly['year'], y=yearly['count'],
                            mode='markers+lines', name='Historical Data',
                            line=dict(color='blue')))
    
    # Trend line
    trend_predictions = model.predict(X)
    fig.add_trace(go.Scatter(x=yearly['year'], y=trend_predictions,
                            mode='lines', name='Trend Line',
                            line=dict(color='green', dash='dash')))
    
    # Future predictions
    fig.add_trace(go.Scatter(x=future_years.flatten(), y=predictions,
                            mode='markers+lines', name='Predictions',
                            line=dict(color='red')))
    
    fig.update_layout(title="Netflix Content Growth Prediction",
                     xaxis_title="Year",
                     yaxis_title="Number of Titles Added")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display predictions
    st.subheader("Prediction Results")
    pred_df = pd.DataFrame({
        'Year': future_years.flatten(),
        'Predicted Titles': predictions.astype(int)
    })
    st.dataframe(pred_df)
    
    # Model performance
    st.subheader("Model Performance")
    train_score = model.score(X, y)
    st.metric("R² Score", f"{train_score:.3f}")
    
    # Additional insights
    st.subheader("Key Insights")
    avg_growth = predictions[-1] - predictions[0]
    st.write(f"• Predicted average annual growth: {avg_growth/len(predictions):.0f} titles")
    st.write(f"• Total predicted content by {prediction_years}: {predictions[-1]:.0f} titles")
    
    # Content type predictions
    st.subheader("Content Type Growth Trends")
    
    # Separate predictions for movies and TV shows
    movies_yearly = df_clean[df_clean['type'] == 'Movie']['year_added'].value_counts().sort_index()
    tv_yearly = df_clean[df_clean['type'] == 'TV Show']['year_added'].value_counts().sort_index()
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=movies_yearly.index, y=movies_yearly.values,
                             mode='lines+markers', name='Movies'))
    fig2.add_trace(go.Scatter(x=tv_yearly.index, y=tv_yearly.values,
                             mode='lines+markers', name='TV Shows'))
    
    fig2.update_layout(title="Content Type Trends",
                      xaxis_title="Year",
                      yaxis_title="Number of Titles")
    
    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("This dashboard provides comprehensive analysis of Netflix content data including visualizations, recommendations, and ML predictions.")