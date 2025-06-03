import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# --------- Extract genres ----------
def extract_genres(text):
    return [match.strip() for match in re.findall(r'#\d+ in ([^#,(]+)', str(text))]

# --------- Load and clean data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("Audible_Merged_Cleaned.csv")
    df.dropna(subset=["Description"], inplace=True)
    df["Genres"] = df["Ranks and Genre"].apply(extract_genres)
    return df

# --------- Run clustering ----------
def run_clustering_models(descriptions, n_clusters=10):
    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(descriptions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dbscan = DBSCAN(eps=1.0, min_samples=5, metric='cosine')
    kmeans_labels = kmeans.fit_predict(X)
    dbscan_labels = dbscan.fit_predict(X)
    return X, kmeans_labels, dbscan_labels

# --------- Recommender Function ----------
def get_recommendations(df, genre, clusters, model_name="KMeans", top_n=10):
    genre_books = df[df["Genres"].apply(lambda g: genre in g)]
    if genre_books.empty:
        return pd.DataFrame()

    genre_indices = genre_books.index
    genre_clusters = clusters[genre_indices]

    valid_clusters = genre_clusters[genre_clusters != -1]
    if len(valid_clusters) == 0:
        return pd.DataFrame()

    most_common_cluster = pd.Series(valid_clusters).mode()[0]
    cluster_books = df[(df["Cluster"] == most_common_cluster) & (df["Genres"].apply(lambda g: genre in g))]

    return cluster_books.sort_values("Rating_adv", ascending=False).head(top_n)

# --------- Silhouette Score ----------
def try_silhouette(X, labels):
    try:
        mask = labels != -1
        return round(silhouette_score(X[mask], labels[mask]) if not all(mask) else silhouette_score(X, labels), 3)
    except:
        return "NA"

# --------- PCA Plot ----------
def plot_clusters(X, labels, df, selected_genre, title):
    genre_mask = df["Genres"].apply(lambda g: selected_genre in g)
    filtered_X = X[genre_mask]
    filtered_labels = labels[genre_mask]
    if filtered_X.shape[0] < 2:
        st.info("Not enough books in this genre to plot clusters.")
        return
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(filtered_X.toarray())
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=filtered_labels, cmap='tab10', alpha=0.7)
    ax.set_title(f"{title}: {selected_genre}")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

# --------- Sidebar Navigation ----------
st.sidebar.title("📘 Navigation")
page = st.sidebar.radio("Choose Section", [
    "📊 EDA",
    "📈 Model Comparison",
    "🤖 KMeans Recommender",
    "🧠 DBSCAN Recommender"
])

df = load_data()

# --------- EDA ---------
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

    st.write("### Top Authors")
    top_authors = df["Author"].value_counts().head(10)
    st.bar_chart(top_authors)

# --------- Model Comparison ---------
elif page == "📈 Model Comparison":
    st.title("📈 KMeans vs DBSCAN")

    X, kmeans_labels, dbscan_labels = run_clustering_models(df["Description"])
    kmeans_sil = try_silhouette(X, kmeans_labels)
    dbscan_sil = try_silhouette(X, dbscan_labels)
    dbscan_noise = sum(dbscan_labels == -1)
    kmeans_clusters = len(set(kmeans_labels))
    dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 KMeans")
        st.write(f"Silhouette Score: `{kmeans_sil}`")
        st.write(f"Clusters: `{kmeans_clusters}`")
        st.pyplot(plot_clusters(X, kmeans_labels, df, "Fiction", "KMeans Clustering"))

    with col2:
        st.markdown("### 🟠 DBSCAN")
        st.write(f"Silhouette Score: `{dbscan_sil}`")
        st.write(f"Clusters: `{dbscan_clusters}`")
        st.write(f"Noise Points: `{dbscan_noise}`")
        st.pyplot(plot_clusters(X, dbscan_labels, df, "Fiction", "DBSCAN Clustering"))

# --------- KMeans Recommender ---------
elif page == "🤖 KMeans Recommender":
    st.title("📚 KMeans Book Recommender")
    model, cluster_labels, tfidf_matrix = run_clustering_models(df["Description"])
    df["Cluster"] = cluster_labels

    genre_counts = {}
    for genres in df["Genres"]:
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
    valid_genres = sorted([g for g, count in genre_counts.items() if count >= 10])
    selected_genre = st.selectbox("🎯 Choose Genre", valid_genres)

    if selected_genre:
        recommendations = get_recommendations(df, selected_genre, cluster_labels)
        if not recommendations.empty:
            st.dataframe(recommendations[["Book Name", "Author", "Rating_adv", "Genres"]])
        plot_clusters(tfidf_matrix, cluster_labels, df, selected_genre, "KMeans Genre Clusters")

# --------- DBSCAN Recommender ---------
elif page == "🧠 DBSCAN Recommender":
    st.title("📚 DBSCAN Book Recommender")
    model, cluster_labels, tfidf_matrix = run_clustering_models(df["Description"])
    df["Cluster"] = cluster_labels

    genre_counts = {}
    for genres in df["Genres"]:
        for g in genres:
            genre_counts[g] = genre_counts.get(g, 0) + 1
    valid_genres = sorted([g for g, count in genre_counts.items() if count >= 10])
    selected_genre = st.selectbox("🎯 Choose Genre", valid_genres)

    if selected_genre:
        recommendations = get_recommendations(df, selected_genre, cluster_labels, model_name="DBSCAN")
        if not recommendations.empty:
            st.dataframe(recommendations[["Book Name", "Author", "Rating_adv", "Genres"]])
        plot_clusters(tfidf_matrix, cluster_labels, df, selected_genre, "DBSCAN Genre Clusters")
