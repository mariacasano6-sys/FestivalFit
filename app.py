import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
 
# Page configuration
st.set_page_config(
    page_title="FestivalFit",
    page_icon="🎵",
    layout="wide"
)
 
 
@st.cache_data
def load_data():
    df_paese = pd.read_csv('data/paese_profilo.csv', index_col=0)
    df_artisti = pd.read_csv('data/artisti_profilo.csv')
    return df_paese, df_artisti
 
 
@st.cache_resource
def load_model():
    return joblib.load('models/pipeline_rf.pkl')
 
 
df_paese, df_artisti = load_data()
pipeline = load_model()
 
# ===========================
# TRANSLATIONS
# ===========================
 
TESTI = {
    'en': {
        'sidebar_sub': "A tool for music promoters",
        'language': "Language",
        'navigate': "Navigate",
        'app_sub': "A data-driven tool for music promoters",
        'page_country': "🗺️ Country profile",
        'page_match': "🎤 Artist Match",
        'page_cluster': "🌍 Clustering",
        'page_predict': "🔮 Artist prediction",
        # Page 1
        'p1_header': "🗺️ Sound profile by country",
        'choose_country': "Choose a country",
        'countries_in_cluster': "🌍 **Countries in this cluster ({n}):**",
        'p1_lines_sub': "📈 Average profiles across clusters",
        'p1_lines_title': "Average feature values by cluster",
        'feature': "Feature",
        'avg_norm': "Average value (normalised 0-1)",
        'p1_radar_title': "Sound profile — {country} ({cluster})",
        # Page 2
        'p2_sub': "Find the artists that best fit your festival",
        'choose_fest_country': "Choose the festival's country",
        'audience_info': "🎵 The audience in **{country}** belongs to the **{cluster}** cluster",
        'n_artists': "**{n} artists** match this market",
        'min_pop': "Minimum popularity",
        'n_artists_pop': "**{n} artists** with popularity ≥ {pop}",
        'choose_lineup': "🎤 Choose the artists for your line-up",
        'your_lineup': "🎪 Your line-up",
        'compare_sub': "📊 Sound profile comparison",
        'compare_title': "Sound profile comparison — line-up",
        # Page 3
        'p3_header': "🌍 Global clustering",
        'p3_sub': "72 countries grouped by sound profile",
        'n_countries': "**{n} countries:**",
        'p3_radar_sub': "📊 Sound profile by cluster",
        'p3_radar_title': "Average sound profile by cluster",
        # Page 4
        'p4_sub': "Find out where an artist's sound fits around the world",
        'search_artist': "Search for an artist",
        'belongs': "**{artist}** belongs to the **{cluster}** cluster",
        'recommended': "🌍 **Recommended markets ({n} countries):**",
        'p4_radar_sub': "📊 Sound profile of the {cluster} cluster",
        'p4_radar_title': "Sound profile — {cluster}",
    },
    'it': {
        'sidebar_sub': "Tool per promoter musicali",
        'language': "Lingua",
        'navigate': "Naviga",
        'app_sub': "Data-driven tool per promoter musicali",
        'page_country': "🗺️ Profilo paese",
        'page_match': "🎤 Artist Match",
        'page_cluster': "🌍 Clustering",
        'page_predict': "🔮 Predizione artista",
        # Pagina 1
        'p1_header': "🗺️ Profilo sonoro per paese",
        'choose_country': "Scegli un paese",
        'countries_in_cluster': "🌍 **Paesi in questo cluster ({n}):**",
        'p1_lines_sub': "📈 Confronto profili medi tra cluster",
        'p1_lines_title': "Valori medi delle feature per cluster",
        'feature': "Feature",
        'avg_norm': "Valore medio (normalizzato 0-1)",
        'p1_radar_title': "Profilo sonoro — {country} ({cluster})",
        # Pagina 2
        'p2_sub': "Trova gli artisti più compatibili per il tuo festival",
        'choose_fest_country': "Scegli il paese del festival",
        'audience_info': "🎵 Il pubblico di **{country}** appartiene al cluster **{cluster}**",
        'n_artists': "**{n} artisti** compatibili con questo mercato",
        'min_pop': "Popolarità minima",
        'n_artists_pop': "**{n} artisti** con popolarità ≥ {pop}",
        'choose_lineup': "🎤 Scegli gli artisti per la tua line-up",
        'your_lineup': "🎪 La tua line-up",
        'compare_sub': "📊 Confronto profili sonori",
        'compare_title': "Confronto profili sonori — line-up",
        # Pagina 3
        'p3_header': "🌍 Clustering mondiale",
        'p3_sub': "I 72 paesi raggruppati per profilo sonoro",
        'n_countries': "**{n} paesi:**",
        'p3_radar_sub': "📊 Profilo sonoro per cluster",
        'p3_radar_title': "Profilo sonoro medio per cluster",
        # Pagina 4
        'p4_sub': "Scopri dove funziona il sound di un artista nel mondo",
        'search_artist': "Cerca un artista",
        'belongs': "**{artist}** appartiene al cluster **{cluster}**",
        'recommended': "🌍 **Mercati consigliati ({n} paesi):**",
        'p4_radar_sub': "📊 Profilo sonoro del cluster {cluster}",
        'p4_radar_title': "Profilo sonoro — {cluster}",
    },
}
 
NOMI_PAESI = {
    'en': {
        'AE': 'United Arab Emirates', 'AR': 'Argentina', 'AT': 'Austria',
        'AU': 'Australia', 'BE': 'Belgium', 'BG': 'Bulgaria',
        'BO': 'Bolivia', 'BR': 'Brazil', 'BY': 'Belarus',
        'CA': 'Canada', 'CH': 'Switzerland', 'CL': 'Chile',
        'CO': 'Colombia', 'CR': 'Costa Rica', 'CZ': 'Czech Republic',
        'DE': 'Germany', 'DK': 'Denmark', 'DO': 'Dominican Republic',
        'EC': 'Ecuador', 'EE': 'Estonia', 'EG': 'Egypt',
        'ES': 'Spain', 'FI': 'Finland', 'FR': 'France',
        'GB': 'United Kingdom', 'GR': 'Greece', 'GT': 'Guatemala',
        'HK': 'Hong Kong', 'HN': 'Honduras', 'HU': 'Hungary',
        'ID': 'Indonesia', 'IE': 'Ireland', 'IL': 'Israel',
        'IN': 'India', 'IS': 'Iceland', 'IT': 'Italy',
        'JP': 'Japan', 'KR': 'South Korea', 'KZ': 'Kazakhstan',
        'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia',
        'MA': 'Morocco', 'MX': 'Mexico', 'MY': 'Malaysia',
        'NG': 'Nigeria', 'NI': 'Nicaragua', 'NL': 'Netherlands',
        'NO': 'Norway', 'NZ': 'New Zealand', 'PA': 'Panama',
        'PE': 'Peru', 'PH': 'Philippines', 'PK': 'Pakistan',
        'PL': 'Poland', 'PT': 'Portugal', 'PY': 'Paraguay',
        'RO': 'Romania', 'SA': 'Saudi Arabia', 'SE': 'Sweden',
        'SG': 'Singapore', 'SK': 'Slovakia', 'SV': 'El Salvador',
        'TH': 'Thailand', 'TR': 'Turkey', 'TW': 'Taiwan',
        'UA': 'Ukraine', 'US': 'United States', 'UY': 'Uruguay',
        'VE': 'Venezuela', 'VN': 'Vietnam', 'ZA': 'South Africa'
    },
    'it': {
        'AE': 'Emirati Arabi', 'AR': 'Argentina', 'AT': 'Austria',
        'AU': 'Australia', 'BE': 'Belgio', 'BG': 'Bulgaria',
        'BO': 'Bolivia', 'BR': 'Brasile', 'BY': 'Bielorussia',
        'CA': 'Canada', 'CH': 'Svizzera', 'CL': 'Cile',
        'CO': 'Colombia', 'CR': 'Costa Rica', 'CZ': 'Repubblica Ceca',
        'DE': 'Germania', 'DK': 'Danimarca', 'DO': 'Rep. Dominicana',
        'EC': 'Ecuador', 'EE': 'Estonia', 'EG': 'Egitto',
        'ES': 'Spagna', 'FI': 'Finlandia', 'FR': 'Francia',
        'GB': 'Regno Unito', 'GR': 'Grecia', 'GT': 'Guatemala',
        'HK': 'Hong Kong', 'HN': 'Honduras', 'HU': 'Ungheria',
        'ID': 'Indonesia', 'IE': 'Irlanda', 'IL': 'Israele',
        'IN': 'India', 'IS': 'Islanda', 'IT': 'Italia',
        'JP': 'Giappone', 'KR': 'Corea del Sud', 'KZ': 'Kazakistan',
        'LT': 'Lituania', 'LU': 'Lussemburgo', 'LV': 'Lettonia',
        'MA': 'Marocco', 'MX': 'Messico', 'MY': 'Malaysia',
        'NG': 'Nigeria', 'NI': 'Nicaragua', 'NL': 'Olanda',
        'NO': 'Norvegia', 'NZ': 'Nuova Zelanda', 'PA': 'Panama',
        'PE': 'Perù', 'PH': 'Filippine', 'PK': 'Pakistan',
        'PL': 'Polonia', 'PT': 'Portogallo', 'PY': 'Paraguay',
        'RO': 'Romania', 'SA': 'Arabia Saudita', 'SE': 'Svezia',
        'SG': 'Singapore', 'SK': 'Slovacchia', 'SV': 'El Salvador',
        'TH': 'Tailandia', 'TR': 'Turchia', 'TW': 'Taiwan',
        'UA': 'Ucraina', 'US': 'Stati Uniti', 'UY': 'Uruguay',
        'VE': 'Venezuela', 'VN': 'Vietnam', 'ZA': 'Sud Africa'
    },
}
 
# ===========================
# SIDEBAR AND LANGUAGE
# ===========================
 
st.sidebar.title("🎵 FestivalFit")
 
lingua = st.sidebar.radio(
    "Language / Lingua",
    options=['en', 'it'],
    format_func=lambda x: "🇬🇧 English" if x == 'en' else "🇮🇹 Italiano",
    horizontal=True
)
T = TESTI[lingua]
nomi_paesi = NOMI_PAESI[lingua]
 
st.sidebar.subheader(T['sidebar_sub'])
 
# Pages are stored as fixed keys, so changing language keeps the current page
pagina = st.sidebar.radio(
    T['navigate'],
    options=['country', 'match', 'cluster', 'predict'],
    format_func=lambda p: T[f'page_{p}']
)
 
st.title("🎪 FestivalFit")
st.subheader(T['app_sub'])
 
features_cluster = ['danceability', 'energy', 'valence', 'tempo',
                    'acousticness', 'speechiness', 'instrumentalness',
                    'liveness', 'is_explicit', 'mode']
 
nomi_cluster = {
    0: 'High Energy Pop',
    1: 'Pop Mainstream',
    2: 'Latin Groove',
    3: 'Asian Acoustic',
    4: 'Brazilian Beat'
}
 
colori_cluster = {
    'High Energy Pop': '#FF6600',
    'Pop Mainstream': '#4488FF',
    'Latin Groove': '#FF4444',
    'Asian Acoustic': '#44BB44',
    'Brazilian Beat': '#FF9900'
}
 
# Country codes sorted by name in the selected language
paesi_ordinati = sorted(nomi_paesi.keys(), key=lambda x: nomi_paesi[x])
 
 
def normalizza_paesi():
    """Scale the country profiles to 0-1 so features are comparable on charts."""
    scaler_viz = MinMaxScaler()
    return pd.DataFrame(
        scaler_viz.fit_transform(df_paese[features_cluster]),
        index=df_paese.index,
        columns=features_cluster
    )
 
 
def rgba(colore, alpha=0.3):
    """Convert a hex colour into an rgba string with transparency."""
    r, g, b = (int(colore.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r}, {g}, {b}, {alpha})'
 
 
# ===========================
# PAGE 1 - COUNTRY PROFILE
# ===========================
 
if pagina == 'country':
    st.header(T['p1_header'])
 
    paese_nome = st.selectbox(
        T['choose_country'],
        options=paesi_ordinati,
        format_func=lambda x: nomi_paesi.get(x, x)
    )
    profilo = df_paese.loc[paese_nome]
    cluster_id = int(profilo['cluster'])
    cluster_name = nomi_cluster[cluster_id]
    colore = colori_cluster[cluster_name]
 
    st.subheader(f"Cluster: **{cluster_name}**")
 
    # Countries in the same cluster
    paesi_cluster = df_paese[df_paese['cluster'] == cluster_id].index.tolist()
    nomi_cluster_paesi = sorted([nomi_paesi.get(p, p) for p in paesi_cluster])
    st.markdown(T['countries_in_cluster'].format(n=len(paesi_cluster)))
    st.markdown(', '.join(nomi_cluster_paesi))
 
    # Line chart with the normalised average profile of every cluster
    st.subheader(T['p1_lines_sub'])
 
    df_paese_norm = normalizza_paesi()
    df_paese_norm_linee = df_paese_norm.copy()
    df_paese_norm_linee['cluster'] = df_paese['cluster']
 
    medie_cluster = df_paese_norm_linee.groupby('cluster')[features_cluster].mean()
    medie_cluster.index = medie_cluster.index.map(nomi_cluster)
 
    fig_linee = go.Figure()
    for cname, row in medie_cluster.iterrows():
        fig_linee.add_trace(go.Scatter(
            x=features_cluster,
            y=row.values,
            mode='lines+markers',
            name=cname,
            line=dict(color=colori_cluster[cname], width=3 if cname == cluster_name else 1.5),
            opacity=1.0 if cname == cluster_name else 0.4
        ))
 
    fig_linee.update_layout(
        title=T['p1_lines_title'],
        xaxis_title=T['feature'],
        yaxis_title=T['avg_norm'],
        showlegend=True
    )
    st.plotly_chart(fig_linee, width='stretch')
 
    # Radar chart of the selected country
    valori = df_paese_norm.loc[paese_nome][features_cluster].tolist()
    valori += valori[:1]
 
    fig = go.Figure(data=go.Scatterpolar(
        r=valori,
        theta=features_cluster + [features_cluster[0]],
        fill='toself',
        fillcolor=rgba(colore),
        line=dict(color=colore)
    ))
 
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=T['p1_radar_title'].format(country=nomi_paesi[paese_nome], cluster=cluster_name),
        showlegend=False
    )
 
    st.plotly_chart(fig, width='stretch')
 
# ==================================
# PAGE 2 - ARTIST MATCH
# ==================================
 
elif pagina == 'match':
    st.header(T['page_match'])
    st.subheader(T['p2_sub'])
 
    # Country selection
    paese_nome = st.selectbox(
        T['choose_fest_country'],
        options=paesi_ordinati,
        format_func=lambda x: nomi_paesi.get(x, x)
    )
 
    profilo = df_paese.loc[paese_nome]
    cluster_id = int(profilo['cluster'])
    cluster_name = nomi_cluster[cluster_id]
 
    st.info(T['audience_info'].format(country=nomi_paesi[paese_nome], cluster=cluster_name))
 
    # Artists in the same cluster
    artisti_cluster = df_artisti[df_artisti['cluster'] == cluster_id]\
        .sort_values('popularity', ascending=False).copy()
 
    st.write(T['n_artists'].format(n=len(artisti_cluster)))
 
    # Popularity slider
    pop_min = st.slider(T['min_pop'], 0, 100, 50)
    artisti_filtrati = artisti_cluster[artisti_cluster['popularity'] >= pop_min].copy()
 
    st.write(T['n_artists_pop'].format(n=len(artisti_filtrati), pop=pop_min))
 
    # Line-up selection
    artisti_scelti = st.multiselect(
        T['choose_lineup'],
        options=artisti_filtrati['artists'].tolist()
    )
 
    # Table of filtered artists
    st.dataframe(
        artisti_filtrati[['artists', 'popularity', 'cluster_name'] + features_cluster]
        .reset_index(drop=True),
        width='stretch'
    )
 
    # Selected line-up
    if artisti_scelti:
        st.subheader(T['your_lineup'])
        lineup = artisti_filtrati[artisti_filtrati['artists'].isin(artisti_scelti)]
        st.dataframe(lineup[['artists', 'popularity'] + features_cluster],
                     width='stretch')
 
        # Radar chart comparing the selected artists
        st.subheader(T['compare_sub'])
 
        fig = go.Figure()
 
        for _, artista in lineup.iterrows():
            # Normalise each feature to 0-1
            valori = []
            for f in features_cluster:
                min_val = df_artisti[f].min()
                max_val = df_artisti[f].max()
                val_norm = (artista[f] - min_val) / (max_val - min_val)
                valori.append(round(max(0, min(1, val_norm)), 3))
            valori += valori[:1]
 
            fig.add_trace(go.Scatterpolar(
                r=valori,
                theta=features_cluster + [features_cluster[0]],
                fill='toself',
                name=artista['artists'],
                opacity=0.6
            ))
 
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=T['compare_title'],
            showlegend=True
        )
 
        st.plotly_chart(fig, width='stretch')
 
# ==============================
# PAGE 3 - CLUSTERING
# ==============================
 
elif pagina == 'cluster':
    st.header(T['p3_header'])
    st.subheader(T['p3_sub'])
 
    # Countries in each cluster
    for cluster_id, cluster_name in nomi_cluster.items():
        paesi_cluster = df_paese[df_paese['cluster'] == cluster_id].index.tolist()
        nomi = [nomi_paesi.get(p, p) for p in paesi_cluster]
 
        st.markdown(f"### {cluster_name}")
        st.markdown(f"{T['n_countries'].format(n=len(paesi_cluster))} {', '.join(sorted(nomi))}")
        st.markdown("---")
 
    # Radar chart for every cluster
    st.subheader(T['p3_radar_sub'])
 
    fig = go.Figure()
    df_paese_norm = normalizza_paesi()
 
    for cluster_id, cluster_name in nomi_cluster.items():
        colore = colori_cluster[cluster_name]
 
        media_cluster = df_paese_norm[df_paese['cluster'] == cluster_id].mean()
        valori = media_cluster.tolist()
        valori += valori[:1]
 
        fig.add_trace(go.Scatterpolar(
            r=valori,
            theta=features_cluster + [features_cluster[0]],
            fill='toself',
            name=cluster_name,
            line=dict(color=colore),
            opacity=0.6
        ))
 
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=T['p3_radar_title'],
        showlegend=True
    )
 
    st.plotly_chart(fig, width='stretch')
 
# ===========================
# PAGE 4 - ARTIST PREDICTION
# ===========================
 
elif pagina == 'predict':
    st.header(T['page_predict'])
    st.subheader(T['p4_sub'])
 
    artisti_ordinati = df_artisti.dropna(subset=['artists'])\
        .sort_values('popularity', ascending=False)['artists'].tolist()
 
    nome_artista = st.multiselect(
        T['search_artist'],
        options=artisti_ordinati,
        max_selections=1
    )
 
    if nome_artista:
        artista = df_artisti[df_artisti['artists'] == nome_artista[0]].iloc[0]
        input_data = pd.DataFrame([artista[features_cluster].values],
                                  columns=features_cluster)
        cluster_pred = int(pipeline.predict(input_data)[0])
        cluster_pred_name = nomi_cluster[cluster_pred]
        colore_pred = colori_cluster[cluster_pred_name]
 
        # Predicted cluster
        st.success(T['belongs'].format(artist=nome_artista[0], cluster=cluster_pred_name))
 
        # Countries in the cluster
        paesi_cluster = df_paese[df_paese['cluster'] == cluster_pred].index.tolist()
        nomi_consigliati = sorted([nomi_paesi.get(p, p) for p in paesi_cluster])
 
        st.markdown(T['recommended'].format(n=len(paesi_cluster)))
        st.markdown(', '.join(nomi_consigliati))
 
        # Radar chart of the cluster
        st.subheader(T['p4_radar_sub'].format(cluster=cluster_pred_name))
 
        df_paese_norm = normalizza_paesi()
        media_cluster = df_paese_norm[df_paese['cluster'] == cluster_pred].mean()
        valori = media_cluster.tolist()
        valori += valori[:1]
 
        fig = go.Figure(data=go.Scatterpolar(
            r=valori,
            theta=features_cluster + [features_cluster[0]],
            fill='toself',
            fillcolor=rgba(colore_pred),
            line=dict(color=colore_pred)
        ))
 
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=T['p4_radar_title'].format(cluster=cluster_pred_name),
            showlegend=False
        )
 
        st.plotly_chart(fig, width='stretch')
