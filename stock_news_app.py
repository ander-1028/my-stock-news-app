import streamlit as st
import pandas as pd
from gnews import GNews
import jieba
import twstock
from FinMind.data import DataLoader
import datetime
import math
import os
import yfinance as yf

# --- Configuration & Setup ---
st.set_page_config(page_title="台灣股市新聞分析與市場概況", layout="wide", page_icon="📈")

# Function to safely get secrets
def get_secret(key):
    try:
        # Truly safe access to st.secrets
        # We don't even check 'if key in st.secrets' as that might trigger it
        val = st.secrets.get(key)
        if val: return val
    except Exception:
        pass
    return os.environ.get(key)

# --- Classes ---

class StockMatcher:
    def __init__(self):
        self.company_to_code = {}
        self._initialize_stock_data()
        
    def _initialize_stock_data(self):
        """Initialize stock mapping from twstock and optimize jieba."""
        # Load all stocks and ETFs
        for code, info in twstock.codes.items():
            if info.type in ['股票', 'ETF']:
                self.company_to_code[info.name] = code
                # Add company names to jieba dictionary for better segmentation
                jieba.add_word(info.name)
        
    def extract_stocks(self, text):
        """
        Identify matches in text.
        Returns a list of tuples: (Stock Name, Stock Code)
        """
        matches = []
        # Optimization: use jieba cut to match words instead of substring search if possible,
        # but substring is safer for short names.
        # Given performance, we will stick to naive iteration for now but optimized by jieba add_word
        
        # Actually, iterating 2000+ companies for every headline might be slow.
        # Let's try to match against jieba segments.
        words = set(jieba.lcut(text))
        
        for name, code in self.company_to_code.items():
            # Check if full name is in text (more accurate)
            if name in text:
                 matches.append(f"{name}({code})")
        
        return list(set(matches))

    def is_stock_related(self, text):
        return len(self.extract_stocks(text)) > 0

class MarketDataFetcher:
    def __init__(self):
        self.api_token = get_secret("FINMIND_TOKEN")
        self.dl = DataLoader()
        if self.api_token:
            try:
                self.dl.login_by_token(api_token=self.api_token)
            except Exception as e:
                st.error(f"FinMind Login 失敗: {e}")
            
    def get_market_summary(self):
        """Get today's institutional investors data summary."""
        try:
            # FinMind data is usually updated end of day. 
            # If running early in the day, we might need yesterday's data.
            today = datetime.date.today()
            # Try to catch last 3 days to ensure we get data (holidays etc)
            start_date = (today - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            df = self.dl.taiwan_stock_institutional_investors(
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                return None, "查無近日法人數據 (可能為假日或尚未更新)"
                
            # Filter for latest date
            latest_date = df['date'].max()
            dashboard_df = df[df['date'] == latest_date]
            
            # Summarize by type (Foreign, Investment Trust, Dealer)
            summary = dashboard_df.groupby('name')[['buy', 'sell']].sum()
            summary['net'] = summary['buy'] - summary['sell']
            
            return summary, latest_date
            
        except Exception as e:
            return None, str(e)

class MarketQuoteFetcher:
    def __init__(self):
        self.symbols = {
            "^TWII": "台股大盤",
            "2330.TW": "台積電",
            "2454.TW": "聯發科",
            "AAPL": "蘋果 (AAPL)",
            "NVDA": "輝達 (NVDA)"
        }

    def fetch_quotes(self):
        data = []
        try:
            for symbol, name in self.symbols.items():
                ticker = yf.Ticker(symbol)
                # Use fast_info or history if info is slow
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_close
                    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
                    
                    data.append({
                        "股票名稱": name,
                        "代碼": symbol,
                        "最新價格": round(current_price, 2),
                        "漲跌金額": round(change, 2),
                        "漲跌幅 (%)": f"{pct_change:+.2f}%"
                    })
                else:
                    data.append({"股票名稱": name, "代碼": symbol, "最新價格": "無數據", "漲跌金額": "-", "漲跌幅 (%)": "-"})
            return pd.DataFrame(data)
        except Exception as e:
            st.error(f"行情抓取失敗: {e}")
            return pd.DataFrame()

# --- Main App Logic ---

def main():
    st.title("📈 台灣股市新聞分析與市場概況")
    
    # Initialize classes
    if 'matcher' not in st.session_state:
        st.session_state.matcher = StockMatcher()
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("🔍 搜尋與篩選")
        search_query = st.text_input("關鍵字搜尋 (例如: 台積電, 營收)")
        
        sort_order = st.selectbox(
            "排序方式",
            ["時間由新到舊", "時間由舊到新"]
        )
        
        st.markdown("---")
        st.header("🌙 市場概況")
        if st.button("今日市場概況回顧"):
            st.session_state.show_summary = True
            
        st.header("📈 即時行情")
        if st.button("🚀 顯示今日最新行情"):
            st.session_state.show_quotes = True

        st.markdown("---")
        st.info("資料來源: GNews, FinMind, Twstock, yfinance")

    # --- Market Quotes Section ---
    if st.session_state.get('show_quotes', False):
        st.subheader("🚀 今日最新熱門股票行情")
        with st.spinner("正在獲取最新即時數據..."):
            quote_fetcher = MarketQuoteFetcher()
            quotes_df = quote_fetcher.fetch_quotes()
            
            if not quotes_df.empty:
                # Top metrics for indices/big stocks
                m1, m2, m3 = st.columns(3)
                # Show Index, TSMC, NVDA as highlights
                def get_row(df, sym): return df[df['代碼'] == sym].iloc[0] if sym in df['代碼'].values else None
                
                idx_row = get_row(quotes_df, "^TWII")
                tsmc_row = get_row(quotes_df, "2330.TW")
                nvda_row = get_row(quotes_df, "NVDA")
                
                if idx_row is not None:
                    m1.metric(idx_row['股票名稱'], f"{idx_row['最新價格']:,}", idx_row['漲跌幅 (%)'])
                if tsmc_row is not None:
                    m2.metric(tsmc_row['股票名稱'], f"{tsmc_row['最新價格']:,}", tsmc_row['漲跌幅 (%)'])
                if nvda_row is not None:
                    m3.metric(nvda_row['股票名稱'], f"{nvda_row['最新價格']:,}", nvda_row['漲跌幅 (%)'])
                
                st.write("")
                st.dataframe(quotes_df, use_container_width=True, hide_index=True)
            
            if st.button("關閉行情"):
                st.session_state.show_quotes = False
        st.divider()

    # --- Market Summary Section ---
    if st.session_state.get('show_summary', False):
        st.subheader("🧐 每日市場概況回顧")
        token = get_secret("FINMIND_TOKEN")
        
        if not token:
            st.warning("⚠️ 未偵測到 FINMIND_TOKEN，無法獲取詳細法人數據。")
            st.info("請於 .streamlit/secrets.toml 中設定 FINMIND_TOKEN='您的密鑰'")
            if st.button("關閉概況 "):
                st.session_state.show_summary = False
        else:
            fetcher = MarketDataFetcher()
            summary_df, msg = fetcher.get_market_summary()
            
            if summary_df is not None:
                st.caption(f"資料日期: {msg}")
                
                # Display metrics
                cols = st.columns(len(summary_df.head(4)))
                for idx, (name, row) in enumerate(summary_df.iterrows()):
                    if idx >= 4: break
                    net = row['net']
                    
                    with cols[idx]:
                        st.metric(
                            label=name,
                            value=f"{int(net/1000000):,}M",
                            delta=f"{int(net/1000):,}K"
                        )
            else:
                st.warning(f"無法取得市場數據: {msg}")
            
            if st.button("關閉概況"):
                st.session_state.show_summary = False
        st.divider()

    # --- News Fetching & Processing ---
    
    # We store fetched news in session state to avoid refetching on every interaction
    if 'raw_news' not in st.session_state:
        st.session_state.raw_news = []
        
    # Auto-fetch logic or button? The spec implies "Search" triggers it, 
    # but initially we should verify if we have data.
    # Let's add a "Fetch" button effectively but also auto-fetch on load if empty?
    # User spec: "按下搜尋後...". So maybe a fetch button is better or just auto-update on input change.
    # To save API calls, let's use a explicit button or cache.
    # Given the flow, let's auto-fetch generic news if empty, and filter when 'search' changes.
    
    if not st.session_state.raw_news:
         with st.spinner("正在載入最新財經新聞..."):
            try:
                google_news = GNews(language='zh-Hant', country='TW', period='24h', max_results=50)
                # Retry different queries
                news = google_news.get_news('台灣 股市')
                if not news:
                    news = google_news.get_news('台股')
                if not news:
                    news = google_news.get_top_news()
                
                st.session_state.raw_news = news if news else []
            except Exception as e:
                st.error(f"抓取新聞失敗: {e}")
                st.session_state.raw_news = []

    # Apply Search & Filter
    all_news = st.session_state.raw_news
    
    # 1. Search Query
    if search_query:
        # If user types a query, maybe we should fetch NEW data from GNews for that query?
        # Specification says: "按下搜尋後，程式需先根據關鍵字過濾 gnews 結果"
        # Since we only fetched 100 items 'general', filtering locally might yield 0 results.
        # It's better to refetch if query exists.
        pass # Optimization: decide if refetch or local filter. 
        # For this implementation, let's refetch if query changes to ensure we get relevant news.
    
    # Simple Local Filter first for responsiveness if just sorting
    filtered_news = []
    
    # Logic: If search query is present, we might want to REFRETCH from GNews with that query 
    # because the generic '台灣 股市' list might not have specific keywords.
    # But to avoid complexity of state management, let's stick to local filtering first 
    # OR provide a "Refresh/Search" button. 
    # Let's add a "Search" button in sidebar to make it explicit if fetching new data.
    # User said: "按下搜尋後...". 
    
    # Let's refetch if we detect a change in intention, but standard UI just filters.
    # Standard GNews usage:
    # If search_query is provided, filter the existing list. 
    
    for item in all_news:
        if search_query and search_query not in item['title']:
            continue
        filtered_news.append(item)
        
    # 2. Smart Filtering (Stock Only)
    final_news = []
    matcher = st.session_state.matcher
    
    for item in filtered_news:
        # Some items might not have 'title' if GNews structure changes
        title = item.get('title', '')
        stocks = matcher.extract_stocks(title)
        if stocks:
            item['related_stocks'] = stocks
            final_news.append(item)
    
    # Debug info (only if dev mode or similar, but let's show success message with details)
    # st.write(f"抓取 {len(all_news)} 則，關鍵字過濾後 {len(filtered_news)} 則，個股識別後 {len(final_news)} 則")
            
    # 3. Sorting
    if sort_order == "時間由新到舊":
         # GNews usually returns newest first, but let's ensure
         # Format check: 'published date' is str like 'Mon, 20 Jan 2026 ...'
         # Parsing dates can be tricky. GNews returns standardized format.
         try:
            final_news.sort(key=lambda x: pd.to_datetime(x['published date']), reverse=True)
         except:
            pass # Keep default order if parse fails
    else:
         try:
            final_news.sort(key=lambda x: pd.to_datetime(x['published date']), reverse=False)
         except:
            pass

    # --- Display with Pagination ---
    total_items = len(final_news)
    items_per_page = 10
    total_pages = math.ceil(total_items / items_per_page)
    
    if total_items == 0:
        st.info("沒有找到符合條件的與「台灣上市櫃公司」相關的新聞。")
    else:
        st.success(f"找到 {total_items} 則相關新聞")
        
        # Pagination Control
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 1
            
        # Reset page if filter changes drastically (naive check: if page > total_pages)
        if st.session_state.page_number > total_pages:
            st.session_state.page_number = 1

        # Calculate slice
        start_idx = (st.session_state.page_number - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_items = final_news[start_idx:end_idx]
        
        # Display Items
        for item in page_items:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### [{item['title']}]({item['url']})")
                    st.caption(f"發布時間: {item['published date']}")
                with col2:
                    st.write(" ".join([f"`{s}`" for s in item['related_stocks']]))
                st.divider()

        # Pagination Buttons
        if total_pages > 1:
            st.write("---")
            # Centered columns for pagination
            cols = st.columns(total_pages + 2) # Just simple number buttons
            # Limit number of buttons if too many pages? For now assume < 10 pages reasonable.
            
            # Simple Prev/Next + Current Page indicator
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.session_state.page_number > 1:
                    if st.button("⬅️ 上一頁"):
                        st.session_state.page_number -= 1
                        st.rerun()
            with c2:
                st.markdown(f"<div style='text-align: center'> 第 {st.session_state.page_number} / {total_pages} 頁 </div>", unsafe_allow_html=True)
            with c3:
                if st.session_state.page_number < total_pages:
                    if st.button("下一頁 ➡️"):
                        st.session_state.page_number += 1
                        st.rerun()

if __name__ == "__main__":
    main()
