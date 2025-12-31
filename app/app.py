import streamlit as st
import time,os , base64, json, tqdm, boto3
from pathlib import Path


AWS_REGION = 'us-east-1'
AWS_BUCKET_VECTOR_NAME  = "coffee-products-tutorial"
AWS_INDEX_VECTOR_NAME ="idx-coffee-products"
shops = ["All",'starbucks', 'folgers','green mountain coffee roasters','nespresso','nescafé',
 'keurig','maxwell house','lavazza','tassimo','illy','dolce gusto','mccafé',"dunkin' donuts",'fresh roasted coffee']
 


class EmbeddingsGenerator:
    def __init__(self, 
                 MODEL_NAME:str='amazon.titan-embed-text-v2:0', 
                 AWS_REGION:str=''
                 ):
        self.MODEL_NAME = MODEL_NAME
        self.AWS_REGION = AWS_REGION

    def create_client(self):
        client = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.AWS_REGION,
            )
        return client
    
    def get_embeddings(self, text:str):
        client = self.create_client()

        response = client.invoke_model(
            modelId=self.MODEL_NAME,
            body=json.dumps({
                "inputText": text
            })
        )
        response_body = json.loads(response['body'].read())
        embeddings = response_body['embedding']
        return embeddings
    
    def generate_embeddings_batch(self, texts:list):
        embeddings_list = []
        for text in tqdm(texts):
            embeddings = self.get_embeddings(text)
            embeddings_list.append(embeddings)
        return embeddings_list


class S3VectorSearch:
    def __init__(self, aws_region, 
                 bucket_name, index_name):
        self.aws_region = aws_region
        self.bucket_name = bucket_name
        self.index_name = index_name
        
        # Initialize embeddings generator
        self.emb_generator = EmbeddingsGenerator(
            AWS_REGION=aws_region
        )
        
        # Initialize S3 Vectors client
        self.s3_client = boto3.client(
            service_name='s3vectors',
            region_name=aws_region
        )
    
    def get_embedding(self, text):
        """Get embedding for a given text"""
        return self.emb_generator.get_embeddings(text=text)
    
    def build_filter(self, min_price, max_price, min_rating, shop_name):
        """Build filter for S3 Vector search"""
        filters = []
        
        # Add price filter if exists
        if min_price is not None and min_price > 0:
            filters.append({"price": {"$gte": float(min_price)}})
        
        if max_price is not None and max_price < 100:
            filters.append({"price": {"$lte": float(max_price)}})
        
        # Add rating filter if exists
        if min_rating is not None and min_rating > 0:
            filters.append({"average": {"$gte": float(min_rating)}})

        # Add shop filter if exists (DON'T add if "All")
        if shop_name and shop_name.lower() not in ("all", "todos"):
            filters.append({"shop_name": {"$eq": shop_name.lower()}})

        # Return correct structure based on number of filters
        if not filters:
            return None
        elif len(filters) == 1:
            return filters[0]  
        else:
            return {"$and": filters}  
    
    def search(self, query_text, min_price=None, max_price=None, 
               min_rating=None, shop_name=None, top_k=3):
        """Perform complete search with text and filters"""
        # Get embedding
        query_embedding = self.get_embedding(query_text)
        
        # Build filter
        filter_data = self.build_filter(min_price, max_price, min_rating, shop_name)
        
        # Prepare base parameters
        query_params = {
            "vectorBucketName": self.bucket_name,
            "indexName": self.index_name,
            "queryVector": {"float32": query_embedding},
            "topK": top_k,
            "returnDistance": True,
            "returnMetadata": True
        }
        
        # Only add filter if exists
        if filter_data:
            query_params["filter"] = filter_data
        
        # Execute search
        query_result = self.s3_client.query_vectors(**query_params)
        return query_result['vectors']
    
    def format_results(self, raw_results):
        """Format raw S3 Vector results to app format"""
        formatted_results = []
        
        for item in raw_results:
            metadata = item.get('metadata', {})
            distance = item.get('distance', 0)
            similarity = round((1 - distance) * 100, 1)
            
            result = {
                "name": f"{metadata.get('shop_name', 'Unknown').title()} Coffee",
                "description": f"Coffee code: {item.get('key', 'N/A')}",
                "price": metadata.get('price', 0),
                "rating": metadata.get('average', 0),
                "reviews": metadata.get('rating_number', 0),
                "shop": metadata.get('shop_name', 'Unknown').title(),
                "similarity": similarity,
                "key": item.get('key', 'N/A')
            }
            formatted_results.append(result)
        
        return formatted_results



# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Coffee Finder ☕",
    page_icon="☕",
    layout="wide"
)

# ----------------------------
# Session state init
# ----------------------------
defaults = {
    "theme": "light",
    "messages": [],
    "user_name": None,
    "user_emoji": None,
    "min_price": 0,
    "max_price": 100,
    "min_rating": 0.0,
    "shop_name": "All",
    "show_results": False,
    "last_results": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# Helpers
# ----------------------------
def get_image_base64(path: Path):
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()

def get_colors():
    return {
        "bg": "#FFFFFF",
        "text": "#2C2F33",
        "secondary": "#F7F8FA",
        "accent": "#5865F2",
        "border": "#E3E5E8",
        "user_msg": "#5865F2",
        "assistant_msg": "#F2F3F5",
    }

colors = get_colors()

# ----------------------------
# CSS
# ----------------------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {colors['bg']};
}}

.welcome-box {{
    background: linear-gradient(135deg, {colors['accent']}, #7289DA);
    color: white;
    padding: 5px;
    border-radius: 20px;
    text-align: center;
    margin: 5px 0;
    box-shadow: 0 8px 16px rgba(88, 101, 242, 0.2);
}}

button[kind="primary"] {{
    background-color: {colors['accent']} !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    color: white !important;
}}

button[kind="secondary"] {{
    border-radius: 14px !important;
    font-size: 26px !important;
    padding: 12px !important;
}}

button[kind="secondary"]:hover {{
    border-color: {colors['accent']} !important;
    transform: scale(1.05);
}}

.preview-image {{
    display: block;
    margin: 0 auto;
    max-width: 320px;
    border-radius: 16px;
}}

.filter-info {{
    background-color: {colors['secondary']};
    border-left: 4px solid {colors['accent']};
    padding: 12px 16px;
    border-radius: 8px;
    margin: 10px 0;
}}

.result-card {{
    background-color: {colors['secondary']};
    border: 2px solid {colors['accent']};
    border-radius: 16px;
    padding: 20px;
    margin: 15px 0;
    transition: transform 0.2s;
}}

.result-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(88, 101, 242, 0.15);
}}

.coffee-spinner {{
    text-align: center;
    padding: 40px;
    background-color: {colors['secondary']};
    border-radius: 16px;
    margin: 20px 0;
}}

.coffee-walk {{
    font-size: 50px;
    animation: walk 3s linear infinite;
}}

@keyframes walk {{
    0% {{ transform: translateX(-50px); }}
    100% {{ transform: translateX(50px); }}
}}

/* Streamlit chat message styling */
.stChatMessage {{
    border-radius: 18px !important;
}}

/* Custom slider */
div[data-baseweb="slider"] > div > div > div {{
    background-color: {colors['accent']} !important;
}}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# SVG Icons
# ----------------------------
ICONS = {
    'search': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path></svg>',
    'trash': '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path></svg>',
    'dollar': '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>',
    'star': '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>',
    'shop': '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>',
}

# ----------------------------
# Header
# ----------------------------
col1, col2, col3 = st.columns([2, 8, 2])
with col2:
    img_path = Path("img/preview_app.png")
    img64 = get_image_base64(img_path)
    if img64:
        st.markdown(
            f'<img src="data:image/png;base64,{img64}" class="preview-image" width="50%"/>',
            unsafe_allow_html=True
        )
    else:
        st.markdown("<h1 style='text-align:center'>☕</h1>", unsafe_allow_html=True)

# ----------------------------
# ONBOARDING
# ----------------------------
if st.session_state.user_name is None:

    st.markdown("""
    <div class="welcome-box">
        <h3>☕ Welcome to Coffee Finder!</h3>
        <p>To get started, tell us a bit about yourself</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    name = st.text_input("**👤 Your name**", placeholder="E.g., María")

    st.markdown("**😊 Choose your avatar**")
    cols = st.columns(6)

    EMOJI_OPTIONS = ["🦄", "🐻‍❄️", "❤️", "⭐", "🌈", "🎨"]
    
    # Initialize selected emoji if not set
    if st.session_state.user_emoji is None:
        st.session_state.user_emoji = ""

    for idx, emoji in enumerate(EMOJI_OPTIONS):
        with cols[idx]:
            # Check if this emoji is selected
            is_selected = st.session_state.user_emoji == emoji
            
            if st.button(
                emoji,
                key=f"emoji_{idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.user_emoji = emoji
                st.rerun()  # Force immediate rerun to update UI

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Start", type="primary", use_container_width=True):
            if name and st.session_state.user_emoji:
                st.session_state.user_name = name
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "👋 I'm here to help you find the perfect coffee. Please select your search filters and describe what you're looking for!",
                    "avatar": "☕"
                })
                st.rerun()
            else:
                st.warning("Please enter your name and select an avatar")

    st.stop()

# ----------------------------
# SIDEBAR FILTERS
# ----------------------------
with st.sidebar:
    st.markdown(f"### 🔍 Search Filters")

    st.markdown(f"<div style='display: flex; align-items: center; gap: 8px; margin-bottom: 8px;'>{ICONS['dollar']} <b>Price Range</b></div>", unsafe_allow_html=True)
    price_range = st.slider(
        "Price ($)", 0, 100,
        (st.session_state.min_price, st.session_state.max_price),
        5,
        label_visibility="collapsed"
    )
    st.session_state.min_price = price_range[0]
    st.session_state.max_price = price_range[1]

    st.markdown(f"<div style='display: flex; align-items: center; gap: 8px; margin: 16px 0 8px 0;'>{ICONS['star']} <b>Minimum Rating</b></div>", unsafe_allow_html=True)
    st.session_state.min_rating = st.slider(
        "Rating", 0.0, 5.0, st.session_state.min_rating, 0.5,
        label_visibility="collapsed"
    )

    st.markdown(f"<div style='display: flex; align-items: center; gap: 8px; margin: 16px 0 8px 0;'>{ICONS['shop']} <b>Shop</b></div>", unsafe_allow_html=True)
    st.session_state.shop_name = st.selectbox(
        "Shop", shops,
        label_visibility="collapsed"
    )

# ----------------------------
# CHAT DISPLAY
# ----------------------------
st.markdown("### 💬 Chat")

# Display chat messages

for message in st.session_state.messages:
    avatar = message.get("avatar", "☕" if message["role"] == "assistant" else st.session_state.user_emoji)
    
    with st.chat_message(message["role"], avatar=avatar):
        content = message["content"]
        
        # If it's the first assistant message, add active filters
        if message["role"] == "assistant" and "I'm here to help" in content:
            st.write(content)
            st.markdown(f"""
            <div class="filter-info">
            📊 <b>Active Filters:</b>
            💰 Price: ${st.session_state.min_price} - ${st.session_state.max_price}     \t    
            ⭐ Rating: ≥ {st.session_state.min_rating}; 
            🏪 Shop: {st.session_state.shop_name}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show the message text
            st.write(content)
            
            # IF there are results to show
            if "results" in message and message["results"]:
                for i, r in enumerate(message["results"], 1):
                    st.markdown(f"""
                    <div class='result-card'>
                        <h4 style='color: {colors['accent']};'>#{i} {r['name']}</h4>
                        <p style='margin: 10px 0;'>{r['description']}</p>
                        <div style='display: flex; justify-content: space-between; margin-top: 12px; flex-wrap: wrap; gap: 10px;'>
                            <span><b>💰</b> ${r['price']}</span>
                            <span><b>⭐</b> {r['rating']}/5.0 ({r['reviews']} reviews)</span>
                            <span><b>🏪</b> {r['shop']}</span>
                        </div>
                        <div style='margin-top: 12px; text-align: right;'>
                            <span style='color: {colors['accent']}; font-weight: 600;'>🎯 Similarity: {r['similarity']}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Closing message
                st.write("Can I help you with another search?")
# INPUT
# ----------------------------
st.markdown("---")
user_input = st.chat_input(
    "Describe the coffee you're looking for..."
)


# ----------------------------
# SEARCH LOGIC
# ----------------------------
if user_input:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "avatar": st.session_state.user_emoji
    })

    # Show spinner
    with st.spinner(""):
        st.markdown(f"""
        <div class="coffee-spinner">
            <div class="coffee-walk">☕</div>
            <h3 style='color: {colors['text']};'>Your coffee is being prepared...</h3>
            <p style='opacity: 0.7; color: {colors['text']};'>Searching for the best options</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(2)
    
    # Initialize search class
    search_engine = S3VectorSearch(
        aws_region=AWS_REGION,
        bucket_name=AWS_BUCKET_VECTOR_NAME,
        index_name=AWS_INDEX_VECTOR_NAME
    )
    
    # Perform search
    raw_results = search_engine.search(
        query_text=user_input,
        min_price=st.session_state.min_price,
        max_price=st.session_state.max_price,
        min_rating=st.session_state.min_rating,
        shop_name=st.session_state.shop_name
    )
    
    # Format results
    filtered = search_engine.format_results(raw_results)
    
    # Store last results
    st.session_state.last_results = filtered
    
    if filtered:
        response_text = f"I found {len(filtered)} perfect options for you!"
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "avatar": "☕",
            "results": filtered  
        })
        
        st.session_state.show_results = True
    else:
        response = "I didn't find results with these filters. Try adjusting them. Can I help you with another search?"
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "avatar": "☕"
        })
        st.session_state.show_results = False

    
    st.rerun()

