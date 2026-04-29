import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.exa import ExaTools
from agno.tools.firecrawl import FirecrawlTools
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
import pandas as pd
import requests
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field
from typing import List, Optional
import json

# Streamlit UI Configuration
st.set_page_config(
    page_title="CompeteIQ | AI Competitor Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8f9fa;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #0066ff;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0052cc;
        box-shadow: 0 4px 12px rgba(0, 102, 255, 0.2);
        transform: translateY(-1px);
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 5px solid #0066ff;
    }
    
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .header-container {
        padding: 2rem 0;
        text-align: center;
        background: linear-gradient(135deg, #0066ff 0%, #00d4ff 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 8px;
    }
    
    /* Custom info box */
    .styled-info {
        padding: 1rem;
        background-color: #e7f0ff;
        border-radius: 8px;
        border-left: 4px solid #0066ff;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for API keys
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("Settings")
    st.markdown("---")
    
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    firecrawl_api_key = st.text_input("Firecrawl API Key", type="password", help="Enter your Firecrawl API key")
    
    search_engine = st.selectbox(
        "Search Engine",
        options=["Perplexity AI - Sonar Pro", "Exa AI"],
        help="Choose the service to find competitor URLs"
    )
    
    if search_engine == "Perplexity AI - Sonar Pro":
        perplexity_api_key = st.text_input("Perplexity API Key", type="password")
        if openai_api_key and firecrawl_api_key and perplexity_api_key:
            st.session_state.openai_api_key = openai_api_key
            st.session_state.firecrawl_api_key = firecrawl_api_key
            st.session_state.perplexity_api_key = perplexity_api_key
        else:
            st.warning("🔑 Keys required to proceed.")
    else:
        exa_api_key = st.text_input("Exa API Key", type="password")
        if openai_api_key and firecrawl_api_key and exa_api_key:
            st.session_state.openai_api_key = openai_api_key
            st.session_state.firecrawl_api_key = firecrawl_api_key
            st.session_state.exa_api_key = exa_api_key
        else:
            st.warning("🔑 Keys required to proceed.")

# Hero Header
st.markdown("""
    <div class="header-container">
        <h1>CompeteIQ</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">AI-Powered Competitor Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

# Instruction Section
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("""
        <div class="styled-info">
            <strong>How it works:</strong> Our specialized agents will discover your competitors, 
            crawl their digital footprint, and synthesize a strategic analysis report for you.
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.success("💡 Provide both URL and description for the most accurate results!")

# Input Section
with st.container():
    c1, c2 = st.columns(2)
    with c1:
        url = st.text_input("Company URL", placeholder="https://example.com")
    with c2:
        description = st.text_area("Company Description", placeholder="e.g., A cloud-based CRM for small businesses", height=68)

# Initialize API keys and tools
if "openai_api_key" in st.session_state and "firecrawl_api_key" in st.session_state:
    if (search_engine == "Perplexity AI - Sonar Pro" and "perplexity_api_key" in st.session_state) or \
       (search_engine == "Exa AI" and "exa_api_key" in st.session_state):
        
        firecrawl_tools = FirecrawlTools(
            api_key=st.session_state.firecrawl_api_key,
            scrape=False,
            crawl=True,
            limit=5
        )

        # Create agents
        model = OpenAIChat(id="gpt-4o", api_key=st.session_state.openai_api_key)
        
        if search_engine == "Exa AI":
            exa_tools = ExaTools(api_key=st.session_state.exa_api_key, category="company", num_results=3)
            competitor_finder_agent = Agent(
                model=model,
                tools=[exa_tools],
                debug_mode=True,
                markdown=True,
                instructions=[
                    "You are a competitor finder agent. Use ExaTools to find competitor company URLs.",
                    "Return ONLY the URLs, one per line, with no additional text."
                ]
            )

        firecrawl_agent = Agent(model=model, tools=[firecrawl_tools, DuckDuckGoTools()], debug_mode=True, markdown=True)
        analysis_agent = Agent(model=model, debug_mode=True, markdown=True)
        comparison_agent = Agent(model=model, debug_mode=True, markdown=True)

        def get_competitor_urls(url: str = None, description: str = None) -> list[str]:
            if search_engine == "Perplexity AI - Sonar Pro":
                perplexity_url = "https://api.perplexity.ai/chat/completions"
                content = f"Find 3 competitor URLs for: URL: {url}, Description: {description}. ONLY URLs."
                payload = {
                    "model": "sonar-pro",
                    "messages": [{"role": "system", "content": "Return 3 URLs only."}, {"role": "user", "content": content}],
                    "max_tokens": 1000, "temperature": 0.2,
                }
                headers = {"Authorization": f"Bearer {st.session_state.perplexity_api_key}", "Content-Type": "application/json"}
                try:
                    response = requests.post(perplexity_url, json=payload, headers=headers)
                    response.raise_for_status()
                    urls = response.json()['choices'][0]['message']['content'].strip().split('\n')
                    return [u.strip() for u in urls if u.strip()]
                except Exception as e:
                    st.error(f"Discovery Error: {str(e)}")
                    return []
            else:
                try:
                    prompt = f"Find 3 competitor URLs for: {url or description}. URLs only."
                    response: RunOutput = competitor_finder_agent.run(prompt)
                    return [l.strip() for l in response.content.strip().split('\n') if l.strip().startswith('http')][:3]
                except Exception as e:
                    st.error(f"Discovery Error: {str(e)}")
                    return []

        class CompetitorDataSchema(BaseModel):
            company_name: str = Field(description="Name of the company")
            pricing: str = Field(description="Pricing details")
            key_features: List[str] = Field(description="Main features")
            tech_stack: List[str] = Field(description="Tech stack")
            marketing_focus: str = Field(description="Marketing focus")
            customer_feedback: str = Field(description="Customer feedback")

        def extract_competitor_info(competitor_url: str) -> Optional[dict]:
            try:
                app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)
                url_pattern = f"{competitor_url}/*"
                response = app.extract([url_pattern], prompt="Extract competitor intelligence.", schema=CompetitorDataSchema.model_json_schema())
                if hasattr(response, 'success') and response.success and response.data:
                    data = response.data
                    return {
                        "competitor_url": competitor_url,
                        "company_name": data.get('company_name', 'N/A'),
                        "pricing": data.get('pricing', 'N/A'),
                        "key_features": data.get('key_features', [])[:5],
                        "tech_stack": data.get('tech_stack', [])[:5],
                        "marketing_focus": data.get('marketing_focus', 'N/A'),
                        "customer_feedback": data.get('customer_feedback', 'N/A')
                    }
                return None
            except: return None

        def generate_comparison_report(competitor_data: list):
            table_data = []
            for comp in competitor_data:
                table_data.append({
                    'Company': f"{comp.get('company_name', 'N/A')}",
                    'Website': comp.get('competitor_url', 'N/A'),
                    'Pricing': comp.get('pricing', 'N/A')[:100] + '...',
                    'Features': ', '.join(comp.get('key_features', [])[:3]),
                    'Tech': ', '.join(comp.get('tech_stack', [])[:3])
                })
            df = pd.DataFrame(table_data)
            st.markdown("### 📊 Competitive Comparison Matrix")
            st.dataframe(df, use_container_width=True)

        def generate_analysis_report(competitor_data: list):
            formatted_data = json.dumps(competitor_data, indent=2)
            report: RunOutput = analysis_agent.run(
                f"Analyze these competitors and suggest market opportunities:\n{formatted_data}\n"
                "Return a professional report with Gaps, Weaknesses, and Growth Strategies."
            )
            return report.content

        # Execution Button
        if st.button("🚀 Start Intelligence Gathering"):
            if url or description:
                progress_container = st.empty()
                with progress_container.container():
                    st.write("🔍 **Phase 1: Discovering Competitors...**")
                    competitor_urls = get_competitor_urls(url=url, description=description)
                
                if competitor_urls:
                    st.info(f"📍 Found {len(competitor_urls)} potential competitors.")
                    
                    competitor_data = []
                    status_cols = st.columns(len(competitor_urls))
                    
                    for i, comp_url in enumerate(competitor_urls):
                        with status_cols[i]:
                            st.write(f"🕵️ Agent {i+1} working...")
                            info = extract_competitor_info(comp_url)
                            if info:
                                competitor_data.append(info)
                                st.success(f"Done: {comp_url[:20]}...")
                            else:
                                st.error(f"Failed: {comp_url[:20]}...")
                    
                    if competitor_data:
                        # Dashboard Metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Competitors Analyzed", len(competitor_data))
                        m2.metric("Success Rate", f"{(len(competitor_data)/len(competitor_urls))*100:.0f}%")
                        m3.metric("Insights Generated", "25+")
                        
                        st.markdown("---")
                        
                        # Results Tabs
                        tab1, tab2, tab3 = st.tabs(["📊 Matrix", "📝 Strategic Report", "🗃️ Raw Data"])
                        
                        with tab1:
                            generate_comparison_report(competitor_data)
                        
                        with tab2:
                            st.markdown("### 🧠 Strategic Intelligence Report")
                            analysis_report = generate_analysis_report(competitor_data)
                            st.markdown(analysis_report)
                        
                        with tab3:
                            st.json(competitor_data)
                        
                        st.balloons()
                    else:
                        st.error("Could not extract data from competitors.")
                else:
                    st.error("No competitors found.")
            else:
                st.error("Please provide a URL or description.")
else:
    st.warning("👈 Please configure your API keys in the sidebar to get started.")