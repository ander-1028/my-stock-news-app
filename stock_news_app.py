import streamlit as st
import pandas as pd
from gnews import GNews
import jieba
import twstock
from FinMind.data import DataLoader
import datetime
import os
import re
import yfinance as yf
import math
import json

# --- Constants & Mapping ---
# Sources for financial news to ensure quality
FINANCIAL_SOURCES = "(工商時報 OR 經濟日報 OR 科技新報 OR 鉅亨網 OR 數位時代)"

DEFAULT_INDUSTRY_MAP = {
    "AI 與 伺服器": [
        ("台積電", "2330"), ("鴻海", "2317"), ("廣達", "2382"), ("緯創", "3231"), 
        ("技嘉", "2376"), ("勤誠", "2359"), ("川湖", "2059"), ("雙鴻", "3324"), ("奇鋐", "3017")
    ],
    "半導體": [
        ("台積電", "2330"), ("聯電", "2303"), ("聯發科", "2454"), ("日月光投控", "3711"), 
        ("世芯-KY", "3661"), ("創意", "3443"), ("京元電子", "2449")
    ],
    "軍工與航太": [
        ("漢翔", "2634"), ("龍德造船", "6753"), ("雷虎", "8033"), ("全訊", "5222"), 
        ("中信造船", "2644"), ("寶一", "8222"), ("亞航", "2630")
    ],
    "低軌衛星": [
        ("昇達科", "3491"), ("啟碁", "6285"), ("金像電", "2368"), ("耀華", "2367"), 
        ("華通", "2313"), ("台揚", "2314"), ("貿聯-KY", "3665")
    ],
    "監控清單": []
}

MAP_FILE = "my_industry_map.json"

def load_industry_map():
    """Load industry map from JSON if exists, otherwise return default."""
    if os.path.exists(MAP_FILE):
        try:
            with open(MAP_FILE, "r", encoding="utf-8") as f:
                saved_map = json.load(f)
                # Convert list [name, code] back to tuple (name, code)
                return {k: [tuple(v) for v in val] for k, val in saved_map.items()}
        except Exception as e:
            st.error(f"預讀清單發生錯誤: {e}")
    return DEFAULT_INDUSTRY_MAP.copy()

def save_industry_map(industry_map):
    """Save industry map to JSON."""
    try:
        with open(MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(industry_map, f, ensure_ascii=False, indent=4)
    except Exception as e:
        st.error(f"儲存清單發生錯誤: {e}")

# --- Configuration & Setup ---
st.set_page_config(page_title="台灣股市新聞分析與市場概況", layout="wide", page_icon="📈")

# Function to safely get secrets
def get_secret(key):
    try:
        val = st.secrets.get(key)
        if val: return val
    except Exception:
        pass
    return os.environ.get(key)

# --- Cached Fetching Functions ---

@st.cache_data(ttl=3600)
def fetch_institutional_data_cached(api_token, start_date, end_date):
    """Cached function to fetch FinMind data."""
    dl = DataLoader()
    if api_token:
        try:
            dl.login_by_token(api_token=api_token)
            df = dl.taiwan_stock_institutional_investors(
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_market_quotes_cached(symbols_tuple):
    """Cached function to fetch yfinance data."""
    data = []
    for symbol in symbols_tuple:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_close
                pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
                data.append({
                    "代碼": symbol,
                    "最新價格": round(current_price, 2),
                    "漲跌金額": round(change, 2),
                    "漲跌幅 (%)": f"{pct_change:+.2f}%"
                })
        except Exception:
            continue
    return pd.DataFrame(data)

@st.cache_data(ttl=600)
def fetch_news_data_cached(query=None):
    """Cached function to fetch GNews data with maximum transparency."""
    try:
        # Use explicit TW and zh-Hant settings
        google_news = GNews(language='zh-Hant', country='TW', period='24h', max_results=50)
        
        if query:
            # Strictly use user input without modification
            news = google_news.get_news(query)
        else:
            # Default fallback news
            news = google_news.get_news('台灣 股市')
            
        return news if news else []
    except Exception as e:
        st.error(f"GNews 抓取發生錯誤: {e}")
        return []

@st.cache_data(ttl=600)
def get_stock_price_cached(code):
    """Quickly fetch stock price for concept stocks."""
    try:
        symbol = f"{code}.TW"
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if not hist.empty:
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else curr
            change_pct = ((curr - prev) / prev) * 100 if prev != 0 else 0
            return curr, change_pct
    except:
        pass
    return None, None

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
            
    def get_market_summary(self):
        """Get today's institutional investors data summary using cached fetch."""
        try:
            today = datetime.date.today()
            start_date = (today - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
            end_date = today.strftime('%Y-%m-%d')
            
            df = fetch_institutional_data_cached(self.api_token, start_date, end_date)
            
            if df.empty:
                return None, "查無近日法人數據 (或 API 錯誤)"
                
            latest_date = df['date'].max()
            dashboard_df = df[df['date'] == latest_date]
            summary = dashboard_df.groupby('name')[['buy', 'sell']].sum()
            summary['net'] = summary['buy'] - summary['sell']
            
            return summary, latest_date
        except Exception as e:
            return None, str(e)

def get_industry_info(text, industry_map):
    """Detect industry from text and return associated stocks."""
    keywords = {
        "AI 與 伺服器": ["伺服器", "AI", "Server", "輝達", "NVIDIA", "運算"],
        "半導體": ["半導體", "晶圓", "台積電", "IC", "晶片", "Semiconductor"],
        "軍工與航太": ["軍工", "航太", "飛機", "國防", "造船", "雷虎"],
        "低軌衛星": ["低軌衛星", "低軌", "衛星", "SpaceX", "Starlink"]
    }
    for industry, keys in keywords.items():
        if any(k.lower() in text.lower() for k in keys):
            return industry, industry_map.get(industry, [])
    return None, None

def detect_supply_chain_action(text):
    """Detect supply chain actions in text."""
    actions = ["打入", "切入", "供應", "獲訂單", "供應鏈", "進軍", "合作"]
    return any(a in text for a in actions)

class MarketQuoteFetcher:
    def __init__(self):
        self.symbols_map = {
            "^TWII": "台股大盤",
            "2330.TW": "台積電",
            "AAPL": "蘋果 (AAPL)",
            "NVDA": "輝達 (NVDA)"
        }

    def fetch_quotes(self):
        try:
            symbols_tuple = tuple(self.symbols_map.keys())
            df = fetch_market_quotes_cached(symbols_tuple)
            
            if not df.empty:
                # Add human readable names back to the cached dataframe
                df['股票名稱'] = df['代碼'].map(self.symbols_map)
                # Reorder columns
                cols = ["股票名稱", "代碼", "最新價格", "漲跌金額", "漲跌幅 (%)"]
                return df[cols]
            return df
        except Exception as e:
            st.error(f"行情處理失敗: {e}")
            return pd.DataFrame()

# --- Main App Logic ---

def intelligent_extract_company(title):
    """
    Intelligently extract company name and code from title.
    Rules:
    1. Text inside brackets (e.g. "台積電").
    2. Nouns before action verbs like "打入", "供貨", "供應".
    """
    # 1. Brackets check
    bracket_match = re.search(r'[(（]([^)）0-9]{2,6})[)）]', title)
    if bracket_match:
        return bracket_match.group(1), ""
    
    # 2. Action verb check
    for verb in ["打入", "供貨", "供應", "供應鏈", "切入"]:
        if verb in title:
            # Simple take first 2-4 chars before verb as company name
            idx = title.find(verb)
            if idx > 2:
                name = title[max(0, idx-3):idx].strip()
                # Clean up punctuation
                name = re.sub(r'[^\w\s]', '', name)
                return name, ""
                
    return "", ""

def main():
    st.title("📈 台灣股市新聞分析與市場概況")
    
    # Initialize modes & session state
    if 'mode' not in st.session_state:
        st.session_state.mode = 'search' # Default mode
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'search_input' not in st.session_state:
        st.session_state.search_input = ""
    
    if 'matcher' not in st.session_state:
        st.session_state.matcher = StockMatcher()
    
    if 'industry_map' not in st.session_state:
        st.session_state.industry_map = load_industry_map()

    # Callbacks for cleaner state management
    def set_industry_mode(ind):
        st.session_state.mode = "industry"
        st.session_state.selected_industry = ind
        st.session_state.search_input = "" # Clear search
        
    def set_search_mode():
        st.session_state.mode = "search"

    # --- Sidebar ---
    with st.sidebar:
        st.header("🔍 搜尋與篩選")
        # Ensure key="search_input" and logic for mode
        search_query = st.text_input("關鍵字搜尋 (例如: 台積電, 營收)", key="search_input")
        if search_query:
            st.session_state.mode = "search"
        
        sort_order = st.selectbox(
            "排序方式",
            ["時間由新到舊", "時間由舊到新"]
        )
        
        if st.button("🚀 顯示今日最新概況", on_click=set_search_mode):
            st.session_state.show_quotes = True

        st.markdown("---")
        st.header("🚀 產業即時趨勢")
        
        # Display as buttons as requested for "Industry Real-time News"
        cols = st.columns(2)
        industries = list(st.session_state.industry_map.keys())
        for i, ind in enumerate(industries):
            with cols[i % 2]:
                st.button(ind, key=f"ind_btn_{i}", on_click=set_industry_mode, args=(ind,))

        if st.session_state.mode == "industry":
            st.button("⬅️ 返回搜尋模式", on_click=set_search_mode)
            
        st.markdown("---")
        st.header("🌙 法人數據")
        if st.button("今日三大法人買賣超"):
            st.session_state.show_summary = True

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

    # --- Industry Analysis Mode ---
    if st.session_state.mode == "industry":
        industry = st.session_state.get('selected_industry', industries[0] if industries else "AI 與 伺服器")
        st.subheader(f"🚀 {industry} - 產業趨勢分析")
        
        # Display Concept Stocks List
        concept_stocks = st.session_state.industry_map.get(industry, [])
        st.info(f"💡 **{industry} 概念股清單：** " + "、".join([f"{name}({code})" for name, code in concept_stocks]))
        
        # Quick price check columns
        price_cols = st.columns(min(len(concept_stocks), 5))
        for i, (name, code) in enumerate(concept_stocks[:5]): # Show first 5 in metrics
            with price_cols[i % 5]:
                price, pct = get_stock_price_cached(code)
                if price:
                    st.metric(name, f"{price:.1f}", f"{pct:+.2f}%")
        
        # Fetch industry specific news
        with st.spinner(f"正在抓取 {industry} 相關高品質財經新聞..."):
            industry_news = fetch_news_data_cached(industry)
            
            if industry_news:
                for item in industry_news[:5]: # Show top 5
                    with st.expander(f"📌 {item.get('title', '無標題')}"):
                        st.write(item.get('description', '內容載入中...'))
                        st.markdown(f"[閱讀全文]({item.get('url')})")
                
                # Full list with clickable price check
                st.write("**🔍 即時股價連動 (點擊個股查詢):**")
                p_cols = st.columns(min(len(concept_stocks), 4))
                for i, (name, code) in enumerate(concept_stocks):
                    with p_cols[i % 4]:
                        if st.button(f"💵 {name} ({code})", key=f"btn_{code}"):
                            p, c = get_stock_price_cached(code)
                            if p:
                                st.write(f"💰 {p:.1f} ({c:+.2f}%)")
                            else:
                                st.write("查無股價")
            else:
                st.warning("目前無相關重大產業新聞。")
        
        if st.button("結束產業分析"):
            st.session_state.mode = "search"
            st.rerun()
        st.divider()
    
    elif st.session_state.mode == "search":
        # --- News Fetching & Processing (Search Mode) ---
        
        # Trigger fetch if first run or query changed
        should_fetch = False
        if 'raw_news' not in st.session_state or not st.session_state.raw_news:
            should_fetch = True
        if search_query != st.session_state.last_query:
            should_fetch = True
            st.session_state.last_query = search_query

        if should_fetch:
            with st.spinner("正在搜尋新聞..."):
                # Strictly direct query
                st.session_state.raw_news = fetch_news_data_cached(search_query if search_query else None)
                st.session_state.page_number = 1 

        all_news = st.session_state.raw_news
        
        # Debug section disabled per user request
        if not all_news:
            st.info("請嘗試更換關鍵字，目前查無資料")
            return 

        # No strict filtering by stock ID - show all but identify stocks for labeling
        final_news = []
        matcher = st.session_state.matcher
        
        for item in all_news:
            title = item.get('title', '')
            # Only label, don't filter
            item['related_stocks'] = matcher.extract_stocks(title)
            final_news.append(item)
                
        # --- Sorting ---
        if sort_order == "時間由新到舊":
             try:
                final_news.sort(key=lambda x: pd.to_datetime(x.get('published date', '2000-01-01')), reverse=True)
             except:
                pass
        else:
             try:
                final_news.sort(key=lambda x: pd.to_datetime(x.get('published date', '2000-01-01')), reverse=False)
             except:
                pass

        # --- Display with Pagination ---
        total_items = len(final_news)
        items_per_page = 10
        total_pages = math.ceil(total_items / items_per_page)
        
        st.success(f"找到 {total_items} 則相關新聞")
            
        # Pagination Control
        if 'page_number' not in st.session_state:
            st.session_state.page_number = 1
                
        if st.session_state.page_number > total_pages:
            st.session_state.page_number = 1

        start_idx = (st.session_state.page_number - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_items = final_news[start_idx:end_idx]
        
        # Display Items
        for item in page_items:
            with st.container(border=True): # Use border for clarity
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"### [{item['title']}]({item['url']})")
                    st.caption(f"發布時間: {item.get('published date', '未知')}")
                with col2:
                    if item.get('related_stocks'):
                        st.write(" ".join([f"`{s}`" for s in item['related_stocks']]))
                    else:
                        st.info("無標註個股")
                
                # Industry smart detection
                ind_name, ind_stocks = get_industry_info(item['title'] + item.get('description', ''), st.session_state.industry_map)
                if ind_name:
                    st.info(f"✨ **{ind_name} 供應鏈追蹤：** " + "、".join([f"{n}({c})" for n, c in ind_stocks]))
                else:
                    st.caption("� 此公司尚未歸類，可手動加入關注清單或產業。")
                
                # --- Universal Supply Chain Update Tool ---
                with st.expander("🛠️ 展開更新工具"):
                    st.write("🚀 **發現潛在產業機會？** 您可以手動更新追蹤清單。")
                    
                    related_stocks = item.get('related_stocks', [])
                    target_name, target_code = "", ""
                    
                    if related_stocks:
                        target_full = related_stocks[0] 
                        match = re.match(r"(.*?)\((.*?)\)", target_full)
                        if match:
                            target_name, target_code = match.groups()
                    
                    if not target_name:
                        ext_name, ext_code = intelligent_extract_company(item['title'])
                        target_name = ext_name
                        target_code = ext_code

                    st.write("---")
                    st.write("確認公司資訊：")
                    fc1, fc2 = st.columns(2)
                    with fc1: target_name = st.text_input("公司名稱", value=target_name, key=f"n_in_{item['url']}")
                    with fc2: target_code = st.text_input("股票代碼", value=target_code, key=f"c_in_{item['url']}")

                    if target_name:
                        if st.button(f"➕ 加入我的監控清單", key=f"mon_btn_{item['url']}"):
                            if "監控清單" not in st.session_state.industry_map:
                                st.session_state.industry_map["監控清單"] = []
                            if (target_name, target_code) not in st.session_state.industry_map["監控清單"]:
                                st.session_state.industry_map["監控清單"].append((target_name, target_code))
                                save_industry_map(st.session_state.industry_map)
                                st.toast(f"📺 已將 {target_name} 加入監控清單", icon="📡")
                            st.rerun()

                        tab_ex, tab_new = st.tabs(["加入現有產業", "建立新主題"])
                        with tab_ex:
                            current_inds = [ind for ind in st.session_state.industry_map.keys() if ind != "監控清單"]
                            if current_inds:
                                target_ind = st.selectbox("選擇目標產業", current_inds, key=f"sel_ind_{item['url']}")
                                if st.button("確認加入產業", key=f"cf_add_{item['url']}"):
                                    if (target_name, target_code) not in st.session_state.industry_map[target_ind]:
                                        st.session_state.industry_map[target_ind].append((target_name, target_code))
                                        save_industry_map(st.session_state.industry_map)
                                        st.toast(f"✅ 已將 {target_name} 加入 {target_ind}！", icon="🚀")
                                    st.rerun()
                        with tab_new:
                            new_theme = st.text_input("新產業名稱", key=f"new_th_{item['url']}")
                            if st.button("建立並加入", key=f"cf_new_{item['url']}"):
                                if new_theme:
                                    if new_theme not in st.session_state.industry_map:
                                        st.session_state.industry_map[new_theme] = [(target_name, target_code)]
                                        save_industry_map(st.session_state.industry_map)
                                        st.toast(f"✨ 已建立新主題 {new_theme}", icon="🆕")
                                    st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True) # Spacer

        # --- Pagination UI at bottom ---
        if total_pages > 1:
            st.divider()
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                if st.session_state.page_number > 1:
                    if st.button("上頁"):
                        st.session_state.page_number -= 1
                        st.rerun()
            with c2:
                st.markdown(f"<div style='text-align: center'> 第 {st.session_state.page_number} / {total_pages} 頁 </div>", unsafe_allow_html=True)
            with c3:
                if st.session_state.page_number < total_pages:
                    if st.button("下頁"):
                        st.session_state.page_number += 1
                        st.rerun()

        # Move Summary Section here to show in search mode as an overlay if active
        if st.session_state.get('show_summary', False):
            st.markdown("---")
            st.subheader("🧐 每日市場概況回顧")
            token = get_secret("FINMIND_TOKEN")
            if not token:
                st.warning("⚠️ 未偵測到 FINMIND_TOKEN")
            else:
                fetcher = MarketDataFetcher()
                summary_df, msg = fetcher.get_market_summary()
                if summary_df is not None:
                    st.caption(f"資料日期: {msg}")
                    cols = st.columns(min(len(summary_df), 4))
                    for idx, (name, row) in enumerate(summary_df.iterrows()):
                        if idx >= 4: break
                        with cols[idx]:
                            st.metric(name, f"{int(row['net']/1000000):,}M", f"{int(row['net']/1000):,}K")
            if st.button("關閉概況"):
                st.session_state.show_summary = False
            st.divider()

if __name__ == "__main__":
    main()
