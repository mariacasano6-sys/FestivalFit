# 🎪 FestivalFit — a data-driven tool for music promoters

**Which artists fit which market?** FestivalFit groups 72 countries by the sound of what their audiences stream on Spotify, then matches artists to the markets whose taste is closest to theirs. It is built for promoters and festival bookers planning a line-up or a tour.

👉 **Live app:** https://festivalplan-psmrfoyocntp474oetcyah.streamlit.app/
*(If the app is asleep, click "Wake up": it takes about a minute.)*

---

## What the app does

| Page | What a promoter can do |
|---|---|
| 🗺️ **Country profile** | Pick a country and see its "sound profile" on a radar chart, its cluster, and the other countries with similar taste. |
| 🎤 **Artist Match** | Pick the festival's country, filter compatible artists by popularity and build a line-up, comparing artists' sound profiles. |
| 🌍 **Clustering** | See how the 72 countries group into 5 listening clusters and how the clusters differ. |
| 🔮 **Artist prediction** | Search an artist and get the cluster they belong to and the list of recommended markets. |

---

## Data

- **Source:** [Top Spotify Songs in 73 Countries (Daily Updated)](https://www.kaggle.com/datasets/asaniczka/top-spotify-songs-in-73-countries-daily-updated), Kaggle — daily Top 50 charts per country.
- **Size:** about 2.1 million chart entries; roughly October 2023 – June 2025.
- **Features used:** danceability, energy, valence, tempo, acousticness, speechiness, instrumentalness, liveness, explicit content, mode (major/minor).

**Cleaning:** removed rows without country, title or artist; removed tracks with tempo or time signature equal to 0 and tracks longer than 10 minutes; split collaborations so each featured artist is counted (11,890 unique artists).

**Processed datasets used by the app** (both derived from the Kaggle source above):

| File | Rows | Content |
|---|---|---|
| `data/paese_profilo.csv` | 72 countries | Rank-weighted average sound profile of each country, with its cluster |
| `data/artisti_profilo.csv` | 11,890 artists | Average sound profile of each artist, with the assigned cluster |

---

## Method

1. **Country sound profile.** For each country, the average of each audio feature, **weighted by chart position** (1 / daily rank), so the songs at the top count more.
2. **Clustering (K-Means).** Features standardised; number of clusters chosen by comparing the Elbow method and the Silhouette score for k = 2–10. Final model: **k = 5**.
3. **Artist profiles.** Each artist's average sound profile across their charting songs, assigned to the closest country cluster.
4. **Classification.** Nine classifiers compared (Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, Naive Bayes, KNN, SVM, XGBoost) on an 80/20 stratified split. **Random Forest** performed best and was tuned with **GridSearchCV** (5-fold cross-validation).
5. **App.** Interactive dashboard in **Streamlit** with **Plotly** charts.

---

## Results

**Five listening clusters**

| Cluster | Countries | Distinctive traits |
|---|---|---|
| Latin Groove | 31 (e.g. Spain, Italy, France, Colombia) | Most danceable, highest share of explicit tracks, more minor keys |
| Pop Mainstream | 20 (e.g. US, UK, Canada, Australia) | Most songs in major keys, mainstream pop |
| High Energy Pop | 15 (e.g. Germany, Japan, Netherlands, Mexico) | Highest energy and positive mood (valence) |
| Asian Acoustic | 5 (India, Indonesia, Malaysia, Pakistan, Israel) | Much more acoustic, calmer and less explicit |
| Brazilian Beat | 1 (Brazil) | A market of its own: high liveness and speechiness |

**Classifier:** Random Forest (200 trees, max depth 20) — **93% accuracy** on the test set (macro F1 0.92). Most important features: liveness, energy and valence.

---

## Limitations and next steps

- **The model reproduces the clustering.** Artist labels come from the K-Means clusters, so the classifier learns to replicate that assignment; it does not predict commercial success in a country.
- **Snapshot in time.** Spotify no longer provides audio features to new apps (since late 2024), so the data cannot be refreshed in the same way.
- **Sound is only part of the story.** Language, local scenes, ticket prices and marketing also decide where an artist sells.
- **Brazil** forms a cluster by itself, which shows how distinctive the market is but limits comparisons.

**Next steps:** rank recommended markets by the artist's actual popularity in each country; add trends over time; add language as a feature.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Streamlit · Plotly · matplotlib · seaborn · Google Colab

## Repository structure

```
├── app.py                  # Streamlit app
├── data/
│   ├── paese_profilo.csv   # sound profile and cluster for each country
│   └── artisti_profilo.csv # sound profile and cluster for each artist
├── models/
│   └── pipeline_rf.pkl     # trained Random Forest pipeline
├── notebooks/              # data cleaning, clustering and model selection
├── images/                 # screenshots for this README
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

**Author:** Maria Casano — Data Analyst with a background in music and live events · [LinkedIn](https://www.linkedin.com/in/maria-casano)
