📚 Audible Book Clustering and Recommender System

This interactive Streamlit web app analyzes audiobook metadata from Audible to:

    Explore book trends and genres via EDA

    Compare clustering models (KMeans vs DBSCAN)

    Recommend books based on genre preferences

🔍 Features
1. Exploratory Data Analysis (EDA)

    Understand distribution of ratings, reviews, top genres & authors

    Visualized using bar charts, histograms, and KDE plots

2. Clustering Model Comparison

    Clustering of book descriptions using:

        KMeans: distance-based clustering

        DBSCAN: density-based clustering, identifies noise points

    Visualized via PCA plots

    Evaluated using Silhouette Score, cluster count, and noise detection

3. Genre-Based Book Recommendation

    Enter your preferred genre to get the Top 10 books

    Uses clustering results to suggest similar books

    Available for both KMeans and DBSCAN

🗂️ Project Structure

📁 Project Root
│
├── app.py                  # Main Streamlit app (Navigation: EDA, Model Comparison, Recommenders)
├── EDA_vs_Comparison.py    # Streamlit app with side-by-side EDA and clustering comparison
├── COMPARISON.py           # Standalone comparison module for KMeans vs DBSCAN
├── KMEANS.py               # KMeans recommender module
├── DBSCAN.py               # DBSCAN recommender module
├── dummy6.py               # Alternate version of KMeans recommender
├── Audible_Merged_Cleaned.csv  # Cleaned dataset with book metadata

📊 Dataset Info

    Audible_Merged_Cleaned.csv includes:

        Book Name, Author, Description

        Rating_adv, Number of Reviews_adv, Price_adv

        Genre info inside Ranks and Genre

🚀 How to Run
1. Clone the Repo

git clone https://github.com/your-username/audible-book-clustering.git
cd audible-book-clustering

2. Install Requirements

pip install -r requirements.txt

3. Launch the App

streamlit run app.py

💡 Tools & Libraries Used

    Python, Pandas, NumPy

    Scikit-learn for clustering & vectorization

    Matplotlib, Seaborn for visualization

    Streamlit for UI

🎯 Use Cases

    Explore how genres differ in popularity and user ratings

    Understand how machine learning clusters similar audiobooks

    Receive genre-specific book recommendations

