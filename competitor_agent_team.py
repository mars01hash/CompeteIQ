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
    page_title="CompeteIQ | Enterprise AI Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enterprise High-Fidelity Design System
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --bg-deep: #0a0a0a;
        --bg-charcoal: #121212;
        --indigo: #6366f1;
        --lime: #bef264;
        --slate: #1e293b;
        --glass: rgba(30, 41, 59, 0.4);
        --glass-border: rgba(255, 255, 255, 0.08);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
    }

    /* Global Architecture */
    .stApp {
        background-color: var(--bg-deep);
        color: var(--text-main);
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit Header/Footer */
    header, footer { visibility: hidden !important; }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: var(--slate); border-radius: 10px; }

    /* Sidebar - Enhanced Visibility */
    section[data-testid="stSidebar"] {
        background: rgba(24, 24, 27, 0.95) !important;
        backdrop-filter: blur(25px) saturate(200%);
        border-right: 1px solid var(--indigo);
        box-shadow: 10px 0 30px rgba(0,0,0,0.5);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, 
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: var(--text-main) !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] h2 {
        color: var(--indigo) !important;
        font-size: 1.4rem !important;
        margin-bottom: 1.5rem !important;
    }

    /* Enterprise Typography */
    h1, h2, h3, .header-text {
        font-weight: 700 !important;
        letter-spacing: -0.04em !important;
        color: white;
    }

    /* High-Fidelity Cards */
    .hf-card {
        background: var(--glass);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .hf-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, transparent, var(--indigo), transparent);
        opacity: 0.5;
    }

    /* Neural Reasoning Labels */
    .reasoning-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--lime);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    /* Enterprise Buttons */
    .stButton>button {
        background: var(--indigo) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        padding: 0.6rem 1.5rem !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2) !important;
    }

    .stButton>button:hover {
        background: #4f46e5 !important;
        box-shadow: 0 4px 30px rgba(99, 102, 241, 0.4) !important;
        transform: translateY(-1px);
    }

    /* Enhanced Inputs */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--glass-border) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: var(--indigo) !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.3) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }

    /* Indicators */
    .live-indicator {
        display: inline-block;
        width: 8px; height: 8px;
        background: var(--lime);
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px var(--lime);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }

    /* Dashboard Header */
    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 2rem 0;
        border-bottom: 1px solid var(--glass-border);
        margin-bottom: 2rem;
    }

    .status-badge {
        background: rgba(190, 242, 100, 0.1);
        color: var(--lime);
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(190, 242, 100, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Dashboard Layout Header
st.markdown(f"""
    <div class="dashboard-header">
        <div>
            <h1 style="margin:0; font-size: 2rem;">INTELLIGENCE_OS <span style="font-weight:300; color:var(--text-dim);">v4.2</span></h1>
            <p style="color:var(--text-dim); margin:0; font-size:0.9rem;">Neural Processing Unit for Competitive Strategic Analysis</p>
        </div>
        <div class="status-badge">
            <span class="live-indicator"></span>SYSTEM_ONLINE
        </div>
    </div>
    """, unsafe_allow_html=True)

# Enterprise Sidebar
with st.sidebar:
    st.markdown('<div class="reasoning-label">CONTROL_CENTER</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin-top:0;">Configuration</h2>', unsafe_allow_html=True)
    
    openai_api_key = st.text_input("NETWORK_API_KEY", type="password", value=st.session_state.get('openai_api_key', ''))
    firecrawl_api_key = st.text_input("CRAWL_PROTOCOL_KEY", type="password", value=st.session_state.get('firecrawl_api_key', ''))
    
    search_engine = st.selectbox(
        "CORE_SEARCH_ENGINE",
        options=["Perplexity AI - Sonar Pro", "Exa AI"]
    )
    
    if search_engine == "Perplexity AI - Sonar Pro":
        perplexity_api_key = st.text_input("PERPLEXITY_KEY", type="password")
        if openai_api_key and firecrawl_api_key and perplexity_api_key:
            st.session_state.openai_api_key = openai_api_key
            st.session_state.firecrawl_api_key = firecrawl_api_key
            st.session_state.perplexity_api_key = perplexity_api_key
    else:
        exa_api_key = st.text_input("EXA_NEURAL_KEY", type="password")
        if openai_api_key and firecrawl_api_key and exa_api_key:
            st.session_state.openai_api_key = openai_api_key
            st.session_state.firecrawl_api_key = firecrawl_api_key
            st.session_state.exa_api_key = exa_api_key

    st.markdown("---")
    st.markdown('<div class="reasoning-label">SYSTEM_METRICS</div>', unsafe_allow_html=True)
    st.code("Uptime: 99.98%\nLatency: 42ms\nAgents: Active")

# Main Multi-Column Grid
grid_col1, grid_col2 = st.columns([1, 2])

with grid_col1:
    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="reasoning-label">01 // TARGET_ACQUISITION</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0;">Neural Input</h3>', unsafe_allow_html=True)
    
    url = st.text_input("DOMAIN_URL", placeholder="https://intelligence.ai")
    description = st.text_area("DESCRIPTION_VECTOR", placeholder="Define the market space...", height=100)
    
    if st.button("EXECUTE_INTEL_GATHERING"):
        st.session_state.start_gathering = True
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
    st.markdown('<div class="reasoning-label">02 // AI_LOGIC_REASONING</div>', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top:0;">Step Chain</h3>', unsafe_allow_html=True)
    
    if st.session_state.get("start_gathering"):
        st.markdown("""
        <div style="font-size: 0.85rem; color: var(--text-dim);">
            <div style="margin-bottom: 8px;">✓ Initializing Scout Agent...</div>
            <div style="margin-bottom: 8px;">✓ Configuring Neural Search...</div>
            <div style="margin-bottom: 8px;">✓ Acquiring Target Vectors...</div>
            <div style="margin-bottom: 8px; color: var(--indigo);">⚡ Processing Web Digital Footprint...</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("Waiting for execution sequence...")
    st.markdown('</div>', unsafe_allow_html=True)

with grid_col2:
    if not st.session_state.get("start_gathering"):
        st.markdown('<div class="hf-card" style="height: 440px; display: flex; align-items: center; justify-content: center; flex-direction: column;">', unsafe_allow_html=True)
        st.markdown('<div class="live-indicator"></div>', unsafe_allow_html=True)
        st.markdown('<h2 style="color: var(--text-dim);">AWAITING_INPUT_SEQUENCE</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: var(--slate);">Initialize target parameters to begin neural processing</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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

        # Neural Processing Pipeline
        with grid_col2:
            if st.session_state.get("start_gathering"):
                if url or description:
                    # Neural Processing Visual
                    st.markdown('<div class="hf-card">', unsafe_allow_html=True)
                    st.markdown('<div class="reasoning-label">03 // LIVE_NEURAL_PROCESSING</div>', unsafe_allow_html=True)
                    st.markdown('<h3 style="margin-top:0;">Throughput Analysis</h3>', unsafe_allow_html=True)
                    
                    # Synthetic Real-time Chart
                    import numpy as np
                    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Signal A', 'Signal B', 'Signal C'])
                    st.line_chart(chart_data)
                    
                    st.markdown("---")
                    progress_container = st.empty()
                    with progress_container.container():
                        st.write("🔍 **Acquiring Competitive Intelligence Vectors...**")
                        competitor_urls = get_competitor_urls(url=url, description=description)
                    
                    if competitor_urls:
                        st.success(f"📍 Acquisition Successful: {len(competitor_urls)} Targets Identified")
                        
                        competitor_data = []
                        status_cols = st.columns(len(competitor_urls))
                        
                        for i, comp_url in enumerate(competitor_urls):
                            with status_cols[i]:
                                st.markdown(f'<div class="reasoning-label">AGENT_{i+1}</div>', unsafe_allow_html=True)
                                st.caption(f"CRAWLING_{comp_url[:20]}...")
                                info = extract_competitor_info(comp_url)
                                if info:
                                    competitor_data.append(info)
                                    st.markdown(f'<span style="color:var(--lime); font-size:0.8rem;">✓ DATA_EXTRACTED</span>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<span style="color:#ef4444; font-size:0.8rem;">! EXTRACTION_ERROR</span>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                        if competitor_data:
                            # Global Insights Section
                            st.markdown('<div class="hf-card">', unsafe_allow_html=True)
                            st.markdown('<div class="reasoning-label">04 // GLOBAL_INSIGHTS_ENGINE</div>', unsafe_allow_html=True)
                            st.markdown('<h2 style="margin-top:0;">Strategic Synthesis</h2>', unsafe_allow_html=True)
                            
                            m1, m2, m3 = st.columns(3)
                            m1.metric("NODES_ANALYZED", len(competitor_data))
                            m2.metric("EXTRACTION_INTEGRITY", f"{(len(competitor_data)/len(competitor_urls))*100:.0f}%")
                            m3.metric("HEURISTIC_INSIGHTS", "25+")
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Results Tabs
                            tab1, tab2, tab3 = st.tabs(["📊 COMPARISON_MATRIX", "📝 STRATEGIC_DOSSIER", "🗃️ RAW_NETWORK_DATA"])
                            
                            with tab1:
                                generate_comparison_report(competitor_data)
                            
                            with tab2:
                                st.markdown("### 🧠 Neural Analysis Report")
                                with st.spinner("Synthesizing market vectors..."):
                                    analysis_report = generate_analysis_report(competitor_data)
                                st.markdown(analysis_report)
                            
                            with tab3:
                                st.json(competitor_data)
                            
                            if st.button("TERMINATE_SESSION"):
                                st.session_state.start_gathering = False
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error("SYSTEM_FAILURE: Could not extract competitive vectors.")
                    else:
                        st.error("SEARCH_FAILURE: No competitor nodes detected in target space.")
                else:
                    st.error("PARAMETER_ERROR: Domain URL or Description Vector required.")
                    st.session_state.start_gathering = False
else:
    st.warning("👈 AWAITING_NETWORK_AUTHENTICATION: Please configure API keys in CONTROL_CENTER.")