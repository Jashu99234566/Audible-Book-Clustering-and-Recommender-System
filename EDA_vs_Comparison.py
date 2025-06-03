import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
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

# --------- Run clustering ----------
def run_clustering_models(descriptions, n_clusters=10):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(descriptions)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    dbscan = DBSCAN(eps=1.0, min_samples=5, metric='cosine')
    dbscan_labels = dbscan.fit_predict(X)

    return X, kmeans_labels, dbscan_labels

# --------- Silhouette score ----------
def try_silhouette(X, labels):
    try:
        mask = labels != -1
        return round(silhouette_score(X[mask], labels[mask]) if not all(mask) else silhouette_score(X, labels), 3)
    except:
        return "NA"

# --------- PCA plot ----------
def plot_pca(X, labels, title):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    return fig

# --------- Sidebar Navigation ----------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to:", ["📊 EDA", "📈 Model Comparison"])

# --------- Load Data ----------
df = load_data()

# --------- EDA Page ----------
if page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    st.write("### Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())

    st.write("### Null Values")
    st.write(df.isnull().sum())

    st.write("### Genre Frequency")
    genre_series = pd.Series([g for genres in df["Genres"] for g in genres])
    top_genres = genre_series.value_counts().head(10)
    st.bar_chart(top_genres)

    st.write("### Rating Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Rating_adv"], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.write("### Reviews Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Number of Reviews_adv"], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.write("### Most Frequent Authors")
    top_authors = df["Author"].value_counts().head(10)
    st.bar_chart(top_authors)

# --------- Model Comparison Page ----------
elif page == "📈 Model Comparison":
    st.title("📈 Clustering Model Comparison: KMeans vs DBSCAN")

    # Run clustering
    X, kmeans_labels, dbscan_labels = run_clustering_models(df["Description"])

    # Evaluation
    kmeans_sil = try_silhouette(X, kmeans_labels)
    dbscan_sil = try_silhouette(X, dbscan_labels)
    dbscan_noise = sum(dbscan_labels == -1)
    kmeans_clusters = len(set(kmeans_labels))
    dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    # Side-by-side comparison
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 KMeans")
        st.write(f"Silhouette Score: `{kmeans_sil}`")
        st.write(f"Number of Clusters: `{kmeans_clusters}`")
        st.pyplot(plot_pca(X, kmeans_labels, "KMeans Clustering (PCA)"))

    with col2:
        st.markdown("### 🟠 DBSCAN")
        st.write(f"Silhouette Score: `{dbscan_sil}`")
        st.write(f"Number of Clusters: `{dbscan_clusters}`")
        st.write(f"Noise Points: `{dbscan_noise}`")
        st.pyplot(plot_pca(X, dbscan_labels, "DBSCAN Clustering (PCA)"))
