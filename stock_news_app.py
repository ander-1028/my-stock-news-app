import subprocess
import sys

def install_and_import(pkg_name, import_name=None):
    # 如果沒有指定 import 名稱，就預設與安裝名稱相同
    if import_name is None:
        import_name = pkg_name
        
    try:
        # 檢查是否已經安裝
        __import__(import_name)
    except ImportError:
        print(f"找不到套件 {import_name}，正在為您自動安裝 {pkg_name}...")
        try:
            # 執行安裝指令
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
            print(f"{pkg_name} 安裝成功！")
        except Exception as e:
            print(f"安裝 {pkg_name} 失敗: {e}")
# 設定需要安裝的清單： (安裝套件名稱, 程式碼內import的名稱)
required_packages = [
    ("gnews", "gnews"),
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("jieba", "jieba"),
    ("lxml", "lxml"),
    ("twstock", "twstock"),
    ("scikit-learn", "sklearn")  # 這裡最重要：安裝叫 scikit-learn，程式內叫 sklearn
]

for pkg_name, import_name in required_packages:
    install_and_import(pkg_name, import_name)

print("\n--- 環境檢查完成，準備執行主程式 ---")

import streamlit as st
import pandas as pd
from gnews import GNews
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import twstock
import time



# --- Configuration & Setup ---
st.set_page_config(page_title="台灣新聞主題分群與個股識別", layout="wide")

class StockMatcher:
    def __init__(self):
        self.company_to_code = {}
        self._initialize_stock_data()

    def _initialize_stock_data(self):
        """Initialize stock mapping from twstock."""
        # twstock.codes contains details for all stocks
        # We want to map Company Name -> Stock Code
        # We will try to match full name and likely the stock name if available
        # Note: twstock.codes is a dict where key is code, value is tuple/namedtuple
        
        # Example structure update might be needed depending on twstock version, 
        # but generally keys are codes.
        
        for code, info in twstock.codes.items():
            # info usually has type, name, ISIN, start, market, group, etc.
            # We focus on 'name'
            if info.type in ['股票', 'ETF']: # Filter for stocks and ETFs if desired
                self.company_to_code[info.name] = code

    def extract_stocks(self, text):
        """
        Identify matches in text.
        Returns a list of tuples: (Stock Name, Stock Code)
        """
        matches = []
        # Naive matching: iterate all companies. 
        # Optimization: Could build a trie or use Aho-Corasick if performance issues arise.
        # For now, simple iteration is acceptable for POC.
        for name, code in self.company_to_code.items():
            if name in text:
                matches.append(f"{name}({code})")
        
        return list(set(matches)) # Deduplicate

class NewsClusterer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.cut, stop_words=None, max_features=1000)
        self.kmeans = None
    
    def fetch_news(self, period='24h'):
        google_news = GNews(language='zh-Hant', country='TW', period=period, max_results=100)
        news = google_news.get_news('台灣股市') # Search specifically for market relevant news or general '台灣'
        # If '台灣股市' is too narrow, we can try just getting top news, but GNews requires a topic or query often for best results.
        # Let's try to get general business/market news or just top news if query is empty.
        if not news:
           # Fallback or broader search
           news = google_news.get_news('財經')
        return news
    
    def cluster_news(self, df, n_clusters=5):
        if df.empty:
            return df
        
        # Vectorize
        # Preprocessing: remove non-chinese chars could help but might remove stock codes 
        # (though we match stock names which are usually Chinese).
        tfidf_matrix = self.vectorizer.fit_transform(df['title'])
        
        # Clustering
        # Ensure we don't ask for more clusters than samples
        true_k = min(n_clusters, len(df) - 1) if len(df) > 1 else 1
        if true_k < 2:
             df['cluster'] = 0
             return df

        self.kmeans = KMeans(n_clusters=true_k, random_state=42)
        self.kmeans.fit(tfidf_matrix)
        
        df['cluster'] = self.kmeans.labels_
        return df

# --- Main App Logic ---

def main():
    st.title("📰 台灣新聞主題分群與個股識別系統")
    st.markdown("""
    本系統自動抓取最新台灣新聞，利用 AI 技術進行主題分群，並自動識別新聞中提及的上市櫃公司。
    """)

    # Sidebar Controls
    with st.sidebar:
        st.header("設定")
        n_clusters = st.slider("分群數量 (Topics)", min_value=3, max_value=12, value=6)
        fetch_btn = st.button("🔄 抓取最新新聞")
        
        st.info("資料來源: GNews (24h)\nNLP: Jieba + TF-IDF + K-Means\n股票資料: twstock")

    if 'news_data' not in st.session_state:
        st.session_state.news_data = pd.DataFrame()

    if fetch_btn:
        with st.spinner("正在抓取新聞並進行分析..."):
            # 1. Initialize
            matcher = StockMatcher()
            clusterer = NewsClusterer()
            
            # 2. Fetch
            raw_news = clusterer.fetch_news()
            
            if not raw_news:
                st.error("未能抓取到新聞，請稍後再試。")
            else:
                # Convert to DataFrame
                df = pd.DataFrame(raw_news)
                # Keep relevant columns: title, published date, url
                df = df[['title', 'published date', 'url']]
                
                # 3. Cluster
                df = clusterer.cluster_news(df, n_clusters)
                
                # 4. Stock Match
                df['related_stocks'] = df['title'].apply(lambda x: matcher.extract_stocks(x))
                df['has_stock'] = df['related_stocks'].apply(lambda x: len(x) > 0)
                
                st.session_state.news_data = df
                st.success(f"成功抓取 {len(df)} 則新聞，分為 {n_clusters} 個群組！")

    # Display Results
    if not st.session_state.news_data.empty:
        df = st.session_state.news_data
        
        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("總新聞數", len(df))
        with col2:
            st.metric("提及股票的新聞數", df['has_stock'].sum())

        st.divider()

        # Group viewing
        # Get cluster counts to show in selectbox
        cluster_counts = df['cluster'].value_counts().sort_index()
        cluster_options = {f"群組 {i} ({count} 則)": i for i, count in cluster_counts.items()}
        
        selected_option = st.selectbox("選擇主題群組", list(cluster_options.keys()))
        selected_cluster_id = cluster_options[selected_option]
        
        # Filter data
        filtered_df = df[df['cluster'] == selected_cluster_id].copy()
        
        # Display as a clean table/list
        st.subheader(f"📌 {selected_option}")
        
        for idx, row in filtered_df.iterrows():
            with st.container():
                # Title with link
                st.markdown(f"**[{row['title']}]({row['url']})**")
                
                # Stocks pills
                if row['related_stocks']:
                    st.write("📈 關聯個股:", " ".join([f"`{s}`" for s in row['related_stocks']]))
                else:
                    st.caption("無相關股票")
                
                st.caption(f"發布時間: {row['published date']}")
                st.divider()

if __name__ == "__main__":
    main()
