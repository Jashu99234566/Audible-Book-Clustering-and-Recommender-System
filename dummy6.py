import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --------- Extract genres ----------
def extract_genres(text):
    return [match.strip() for match in re.findall(r'#\d+ in ([^#,(]+)', str(text))]

# --------- Load and clean data ----------
@st.cache_data
def load_data():
    path = r"D:\D DRIVE DATA\Auidible_book_Project\Audible_Merged_Cleaned.csv"
    df = pd.read_csv(path)
    df.dropna(subset=["Description"], inplace=True)
    df["Genres"] = df["Ranks and Genre"].apply(extract_genres)
    return df

# --------- Cluster book descriptions ----------
def cluster_books(descriptions, num_clusters=10):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(descriptions)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return kmeans, clusters, X

# --------- Recommend books ----------
def get_recommendations(df, genre, model, tfidf_matrix, clusters, top_n=10):
    genre_books = df[df["Genres"].apply(lambda g: genre in g)]
    if genre_books.empty:
        return pd.DataFrame()

    genre_indices = genre_books.index
    genre_clusters = clusters[genre_indices]
    most_common_cluster = pd.Series(genre_clusters).mode()[0]

    cluster_books = df[(df["Cluster"] == most_common_cluster) & (df["Genres"].apply(lambda g: genre in g))]
    return cluster_books.sort_values("Rating_adv", ascending=False).head(top_n)

# --------- Plot clusters using PCA ----------
def plot_clusters(X, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
    ax.set_title("📍 Book Description Clusters")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

# --------- Streamlit App ----------
st.title("📚 Genre-Based Book Recommender")

# Step 1: Load data
df = load_data()

# Step 2: Clustering
model, cluster_labels, tfidf_matrix = cluster_books(df["Description"], num_clusters=10)
df["Cluster"] = cluster_labels

# Step 3: Filter valid genres
genre_count = {}
for genres in df["Genres"]:
    for genre in genres:
        genre_count[genre] = genre_count.get(genre, 0) + 1
valid_genres = sorted([g for g, count in genre_count.items() if count >= 10])

# Step 4: UI - genre dropdown
selected_genre = st.selectbox("🎯 Choose a Genre:", valid_genres)

# Step 5: Show recommendations
if selected_genre:
    recommendations = get_recommendations(df, selected_genre, model, tfidf_matrix, cluster_labels)

    if not recommendations.empty:
        st.subheader(f"📘 Top 10 Books in '{selected_genre}' Genre")
        st.dataframe(recommendations[["Book Name", "Author", "Rating_adv", "Genres"]])
    else:
        st.warning("No books found for this genre.")

# Step 6: Show cluster plot
st.subheader("📍 Book Description Clusters")
plot_clusters(tfidf_matrix, cluster_labels)
