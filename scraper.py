import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def scrape_imdb():
    # Setup Chrome options for headless scraping
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Uncomment to run without opening browser
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    # IMDb 2024 Popular Movies
    url = "https://www.imdb.com/search/title/?release_date=2024-01-01,2024-12-31"
    driver.get(url)
    time.sleep(5) # Wait for page to load

    movie_data = []

    # Find movie containers
    items = driver.find_elements(By.CSS_SELECTOR, ".ipc-metadata-list-summary-item")

    for item in items:
        try:
            # Extract Title
            title = item.find_element(By.CSS_SELECTOR, ".ipc-title__text").text
            # Extract Storyline/Plot
            plot = item.find_element(By.CSS_SELECTOR, ".ipc-html-content-inner-div").text
            
            movie_data.append({"Movie Name": title, "Storyline": plot})
        except Exception as e:
            continue

    df = pd.DataFrame(movie_data)
    df.to_csv("imdb_2024_movies.csv", index=False)
    driver.quit()
    print("Scraping completed. File saved as 'imdb_2024_movies.csv'.")

if __name__ == "__main__":
    scrape_imdb()

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- STEP 1: LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("imdb_2024_movies.csv")
        return df
    except FileNotFoundError:
        st.error("CSV file not found! Please run the scraper first.")
        return None

# --- STEP 2: RECOMMENDATION ENGINE ---
def get_recommendations(user_input, df):
    def clean(text):
        return re.sub(r'[^a-zA-Z\s]', '', str(text).lower())

    df['clean_plot'] = df['Storyline'].apply(clean)
    user_clean = clean(user_input)

    tfidf = TfidfVectorizer(stop_words='english')
    
    all_plots = pd.concat([df['clean_plot'], pd.Series([user_clean])], ignore_index=True)
    tfidf_matrix = tfidf.fit_transform(all_plots)

    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:5]]

    return df.iloc[top_indices]

# --- STEP 3: STREAMLIT UI ---
st.set_page_config(page_title="IMDb 2024 Recommender", page_icon="🎬")

st.title("🎬 IMDb Movie Recommender (2024)")
st.markdown("Enter a **storyline** or **plot** below, and I'll find the best matches from 2024 movies.")

df = load_data()

if df is not None:
    user_query = st.text_area("What kind of story are you looking for?", 
                              placeholder="Example: A space explorer gets lost in a black hole...")

    if st.button("Get Recommendations"):
        if user_query.strip():
            with st.spinner("Analyzing storylines..."):
                results = get_recommendations(user_query, df)
                
                st.subheader("Top 5 Recommended Movies:")
                for index, row in results.iterrows():
                    with st.container():
                        st.markdown(f"### 🎥 {row['Movie Name']}")
                        st.write(f"**Plot:** {row['Storyline']}")
                        st.divider()
        else:
            st.warning("Please type a storyline first!")