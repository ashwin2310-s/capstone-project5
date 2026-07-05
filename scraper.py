import os
import re
import time
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# Machine Learning, Scraping & Parsing Imports
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. SETUP & ENVIRONMENT CONFIGURATION

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = "models"
DATA_DIR = "dataset"
TFIDF_FILE = os.path.join(MODEL_DIR, "tfidf.pkl")
MATRIX_FILE = os.path.join(MODEL_DIR, "tfidf_matrix.pkl")
CSV_PATH = os.path.join(DATA_DIR, "imdb_movies.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CSS_CONTENT = """
.stApp { background: linear-gradient(135deg, #0f172a, #1e293b); color: white; }
section[data-testid="stSidebar"] { background: #111827 !important; }
h1, h2, h3 { color: #f8fafc !important; font-weight: 700; }
div[data-testid="metric-container"] { background: #1f2937; border: 1px solid #374151; border-radius: 16px; padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,.25); }
.stButton>button { width: 100%; border-radius: 12px; background: #ef4444; color: white; border: none; font-weight: bold; transition: .3s; }
.stButton>button:hover { background: #gradient-color-fix !important; background: #dc2626 !important; transform: scale(1.02); }
.stTextInput input, .stTextArea textarea { border-radius: 10px; }
[data-testid="stDataFrame"] { border-radius: 12px; }
hr { border: 1px solid #374151; }
"""
st.markdown(f"<style>{CSS_CONTENT}</style>", unsafe_allow_html=True)

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except:
        nltk.download(pkg, quiet=True)

STOP = set(stopwords.words("english"))
LEM = WordNetLemmatizer()

# 2. TEXT PREPROCESSING ENGINE

def clean_text(text: str) -> str:
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text) # Reverted to match your specific clean profile
    words = [LEM.lemmatize(w) for w in text.split() if w not in STOP and len(w) > 2]
    return " ".join(words)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_story"] = df["Storyline"].fillna("").apply(clean_text)
    df["story_length"] = df["clean_story"].str.split().str.len()
    return df

# 3. HYBRID SCRAPER (Selenium + BeautifulSoup Failback)

IMDB_URL = "https://www.imdb.com/search/title/?release_date=2024-01-01,2024-12-31"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

def scrape_with_selenium(limit):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(f"user-agent={USER_AGENT}")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    movies = []
    try:
        driver.get(IMDB_URL)
        
        # Priority targeting loop using your preferred selector architecture
        selectors = [".ipc-metadata-list-summary-item", "[data-testid='title-card']", ".lister-item"]
        cards = []
        for sel in selectors:
            try:
                WebDriverWait(driver, 6).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, sel)))
                cards = driver.find_elements(By.CSS_SELECTOR, sel)
                if cards: break
            except: continue
            
        for card in cards[:limit]:
            try:
                name = ""
                for heading in [".ipc-title__text", "h3", ".lister-item-header a"]:
                    try:
                        name = card.find_element(By.CSS_SELECTOR, heading).text
                        if name: break
                    except: continue
                
                # Cleanup indexing strings like "1. The Batman"
                if ". " in name: name = name.split(". ", 1)[1]
                
                story = ""
                for plot in [".ipc-html-content-inner-div", "[data-testid='plot']", ".plot"]:
                    try:
                        story = card.find_element(By.CSS_SELECTOR, plot).text
                        if story: break
                    except: continue
                
                if name and story:
                    movies.append({"Movie Name": name, "Storyline": story})
            except: continue
    finally:
        driver.quit()
    return pd.DataFrame(movies)

def scrape_with_requests():
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.5"}
    response = requests.get(IMDB_URL, headers=headers, timeout=10)
    movies = []
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        cards = soup.select(".ipc-metadata-list-summary-item") or soup.select("[data-testid='title-card']") or soup.select(".lister-item")
        for card in cards:
            try:
                name_el = card.select_one(".ipc-title__text") or card.select_one("h3")
                name = name_el.text.strip() if name_el else ""
                if ". " in name: name = name.split(". ", 1)[1]
                
                story_el = card.select_one(".ipc-html-content-inner-div") or card.select_one("[data-testid='plot']")
                story = story_el.text.strip() if story_el else ""
                
                if name and story:
                    movies.append({"Movie Name": name, "Storyline": story})
            except: continue
    return pd.DataFrame(movies)

def fetch_movie_dataset(limit=30):
    try:
        df = scrape_with_selenium(limit)
        if not df.empty: return df, "Selenium Engine (Webdriver Manager)"
    except Exception as e:
        print(f"Selenium execution failure alert: {e}")
        
    try:
        df = scrape_with_requests()
        if not df.empty: return df.head(limit), "BeautifulSoup Engine (Request Failback)"
    except Exception as e:
        print(f"Network request fallback crash: {e}")
        
    return pd.DataFrame(), "None"

# 4. RECOMMENDATION CORE ENGINE

def _build(df: pd.DataFrame):
    data = df.copy()
    data["clean_story"] = data["Storyline"].fillna("").apply(clean_text)
    
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(data["clean_story"])

    joblib.dump(tfidf, TFIDF_FILE)
    joblib.dump(matrix, MATRIX_FILE)
    return tfidf, matrix

def load_resources(df: pd.DataFrame):
    try:
        if os.path.exists(TFIDF_FILE) and os.path.exists(MATRIX_FILE):
            return joblib.load(TFIDF_FILE), joblib.load(MATRIX_FILE)
    except:
        pass
    return _build(df)

def recommend_movies_from_matrix(user_story, df, tfidf, tfidf_matrix, top_n=5):
    user_clean = clean_text(user_story)
    if not user_clean.strip():
        return pd.DataFrame(columns=["Movie Name", "Similarity Score", "Storyline"])
        
    user_vec = tfidf.transform([user_clean])
    cosine_sim = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    result = df.copy()
    result["Similarity Score"] = cosine_sim
    result = result.sort_values("Similarity Score", ascending=False).head(top_n)
    return result[["Movie Name", "Similarity Score", "Storyline"]].reset_index(drop=True)

# 5. DATA VISUALIZATION FUNCTIONS

def show_dashboard_cards(df):
    c1, c2, c3, c4 = st.columns(4)
    lengths = df["Storyline"].fillna("").str.split().str.len()
    c1.metric("Movies Available", len(df))
    c2.metric("Avg Story Length", int(lengths.mean()) if len(df) > 0 else 0)
    c3.metric("Longest Story", lengths.max() if len(df) > 0 else 0)
    c4.metric("Shortest Story", lengths.min() if len(df) > 0 else 0)

def plot_story_length_distribution(df):
    d = df.copy()
    d["Length"] = d["Storyline"].fillna("").str.split().str.len()
    st.plotly_chart(px.histogram(d, x="Length", title="Story Length Distribution", color_discrete_sequence=['#ef4444']), use_container_width=True)

def plot_top_words(df, n=20):
    cleaned_df = preprocess_dataframe(df)
    words = " ".join(cleaned_df["clean_story"]).split()
    cnt = Counter(words).most_common(n)
    cdf = pd.DataFrame(cnt, columns=["Word", "Count"]).sort_values(by="Count", ascending=True)
    st.plotly_chart(px.bar(cdf, x="Count", y="Word", orientation="h", title="Top Content Words", color_discrete_sequence=['#3b82f6']), use_container_width=True)

def plot_wordcloud(df):
    cleaned_df = preprocess_dataframe(df)
    text = " ".join(cleaned_df["clean_story"])
    if not text.strip():
        st.warning("No plot text available to build metrics.")
        return
    wc = WordCloud(width=900, height=400, background_color="#1e293b", colormap="YlOrRd").generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#1e293b')
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# 6. APPLICATION ROUTING LAYER

@st.cache_data
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    else:
        # Fallback entries matching initialization sequence
        return pd.DataFrame([
            {"Movie Name": "The Matrix", "Storyline": "A computer hacker learns from mysterious rebels about the true nature of his reality."},
            {"Movie Name": "Inception", "Storyline": "A thief who steals corporate secrets through the use of dream-sharing technology."},
            {"Movie Name": "Interstellar", "Storyline": "A team of explorers travel through a wormhole in space to ensure survival."},
            {"Movie Name": "The Dark Knight", "Storyline": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham."},
            {"Movie Name": "Avatar", "Storyline": "A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn."}
        ])

df = load_data()
tfidf, tfidf_matrix = load_resources(df)

st.sidebar.title("🎬 CineMatch AI")
page = st.sidebar.radio("Navigation", ["Home", "Dataset Explorer", "AI Recommendation", "Visual Analytics", "Live Data Scraper", "About"])

if page == "Home":
    st.title("🎬 CineMatch AI")
    st.subheader("IMDb Movie Recommendation System Using Storylines")
    show_dashboard_cards(df)

elif page == "Dataset Explorer":
    st.title("Dataset Explorer")
    search = st.text_input("Search Movie")
    data = df
    if search:
        data = df[df["Movie Name"].str.contains(search, case=False, na=False)]
    st.dataframe(data, use_container_width=True)

elif page == "AI Recommendation":
    st.title("Storyline Recommendation")
    story = st.text_area("What kind of story are you looking for?", placeholder="Example: A space explorer gets lost in a black hole...")
    if st.button("Get Recommendations"):
        if story.strip():
            with st.spinner("Analyzing matrix indices..."):
                results = recommend_movies_from_matrix(story, df, tfidf, tfidf_matrix)
                
                st.subheader("Top 5 Recommended Movies:")
                for index, row in results.iterrows():
                    with st.container():
                        score_label = f"Match Confidence: {round(row.get('Similarity Score', 0) * 100, 1)}%" if 'Similarity Score' in row else ""
                        st.markdown(f"### 🎥 {row['Movie Name']}  `{score_label}`")
                        st.write(f"**Plot:** {row['Storyline']}")
                        st.divider()
        else:
            st.warning("Please type a storyline first!")

elif page == "Visual Analytics":
    st.title("Visual Analytics")
    plot_story_length_distribution(df)
    plot_top_words(df)
    plot_wordcloud(df)

elif page == "Live Data Scraper":
    st.title("🌐 Live IMDb Data Scraper")
    st.info(f"Targeting Source URL: {IMDB_URL}")
    
    scrape_limit = st.slider("Select collection limit", 10, 100, 30)
    
    if st.button("Execute Scraper Integration"):
        with st.spinner("Processing network elements..."):
            scraped_df, engine_used = fetch_movie_dataset(limit=scrape_limit)
            
            if not scraped_df.empty:
                scraped_df.to_csv(CSV_PATH, index=False)
                st.success(f"Success! Captured {len(scraped_df)} items via **{engine_used}** and saved to disk!")
                
                st.cache_data.clear()
                if os.path.exists(TFIDF_FILE): os.remove(TFIDF_FILE)
                if os.path.exists(MATRIX_FILE): os.remove(MATRIX_FILE)
                
                st.info("Refreshing internal cache structures...")
                st.rerun()
            else:
                st.error("The system failed to extract entries across available extraction architectures. Please re-verify web connection integrity.")

else:
    st.title("About")
    st.write("CineMatch AI will get your dream movie in screen .")
