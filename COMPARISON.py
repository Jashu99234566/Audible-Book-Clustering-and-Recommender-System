import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# --------- Run both clustering models ----------
def run_clustering_models(descriptions, n_clusters=10):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(descriptions)

    # --- KMeans ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # --- DBSCAN ---
    dbscan = DBSCAN(eps=1.0, min_samples=5, metric='cosine')
    dbscan_labels = dbscan.fit_predict(X)

    return X, kmeans_labels, dbscan_labels

# --------- Silhouette score ----------
def try_silhouette(X, labels, model_name):
    try:
        mask = labels != -1  # exclude noise if any
        score = silhouette_score(X[mask], labels[mask]) if not all(mask) else silhouette_score(X, labels)
        return round(score, 3)
    except:
        return "NA"

# --------- Visualize clusters with PCA ----------
def plot_pca(X, labels, title):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X.toarray())

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    return fig

# --------- Streamlit App ----------
st.title("🔍 Clustering Model Comparison: KMeans vs DBSCAN")

df = load_data()
X, kmeans_labels, dbscan_labels = run_clustering_models(df["Description"])

# --- Evaluation ---
kmeans_sil = try_silhouette(X, kmeans_labels, "KMeans")
dbscan_sil = try_silhouette(X, dbscan_labels, "DBSCAN")

kmeans_clusters = len(set(kmeans_labels))
dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
dbscan_noise = sum(dbscan_labels == -1)

# --- Results ---
st.subheader("📈 Evaluation Metrics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔵 KMeans")
    st.write(f"Silhouette Score: `{kmeans_sil}`")
    st.write(f"Number of Clusters: `{kmeans_clusters}`")
    st.write("Noise Points: `0` (KMeans doesn't detect noise)")
    st.pyplot(plot_pca(X, kmeans_labels, "KMeans Clustering (PCA)"))

with col2:
    st.markdown("### 🟠 DBSCAN")
    st.write(f"Silhouette Score: `{dbscan_sil}`")
    st.write(f"Number of Clusters: `{dbscan_clusters}`")
    st.write(f"Noise Points: `{dbscan_noise}`")
    st.pyplot(plot_pca(X, dbscan_labels, "DBSCAN Clustering (PCA)"))
