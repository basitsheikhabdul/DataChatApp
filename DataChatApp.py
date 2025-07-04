import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import time
from typing import Optional

# Load environment variables
load_dotenv()

class DataChatApp:
    def __init__(self):
        self.setup_page_config()
        self.setup_custom_css()

    def setup_page_config(self):
        st.set_page_config(
            page_title="DataChat Pro",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def setup_custom_css(self):
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .main-header h1, .main-header p { color: white !important; margin: 0.5rem 0; }
        .chat-message {
            padding: 1.5rem; border-radius: 15px; margin: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1); border: 1px solid rgba(0,0,0,0.05);
        }
        .user-message {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white !important; margin-left: 15%; border: none;
        }
        .bot-message {
            background: #f8f9fa; color: #212529 !important;
            border-left: 4px solid #667eea; margin-right: 15%;
        }
        .metric-card {
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            padding: 1.5rem; border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center; margin: 1rem 0;
            border: 1px solid rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        .metric-card:hover { transform: translateY(-2px); }
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important; border: none !important;
            border-radius: 25px !important;
            padding: 0.5rem 2rem !important;
            transition: all 0.3s ease !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
            background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%) !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        st.markdown("""
        <div class="main-header">
            <h1>🤖 DataChat Pro</h1>
            <p>Intelligent conversations with your data using AI</p>
        </div>
        """, unsafe_allow_html=True)

    @st.cache_data
    def load_data(_self, uploaded_file) -> Optional[pd.DataFrame]:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file)
            else:
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None

    def display_data_overview(self, df: pd.DataFrame):
        st.subheader("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""<div class="metric-card">
                <h3>📝 Rows</h3><h2>{len(df):,}</h2></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
                <h3>📋 Columns</h3><h2>{len(df.columns)}</h2></div>""", unsafe_allow_html=True)
        with col3:
            missing = df.isnull().sum().sum()
            st.markdown(f"""<div class="metric-card">
                <h3>❓ Missing</h3><h2>{missing:,}</h2></div>""", unsafe_allow_html=True)
        with col4:
            numeric_cols = df.select_dtypes(include=['number']).shape[1]
            st.markdown(f"""<div class="metric-card">
                <h3>🔢 Numeric</h3><h2>{numeric_cols}</h2></div>""", unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📋 Preview", "📈 Statistics", "🔍 Data Types"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True, height=400)
        with tab2:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                st.dataframe(numeric_df.describe(), use_container_width=True)
            else:
                st.info("No numeric columns found.")
        with tab3:
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)

    def setup_sidebar(self, df: pd.DataFrame):
        st.sidebar.header("⚙️ Settings")
        model_options = {
            "GPT-3.5 Turbo (Fast & Cheap)": "gpt-3.5-turbo",
            "GPT-4 (More Accurate)": "gpt-4",
            "GPT-4 Turbo": "gpt-4-turbo-preview"
        }
        selected_model = st.sidebar.selectbox("🧠 Select AI Model", list(model_options.keys()), index=0)
        temperature = st.sidebar.slider("🌡️ Response Creativity", 0.0, 1.0, 0.1, 0.1)
        st.sidebar.header("💡 Quick Insights")
        if st.sidebar.button("📊 Generate Summary"):
            self.generate_quick_summary(df)
        if st.sidebar.button("📈 Show Correlations"):
            self.show_correlations(df)
        st.sidebar.header("❓ Sample Questions")
        for i, q in enumerate([
            "What are the key statistics of this dataset?",
            "Show me the distribution of the main columns",
            "Are there any missing values I should know about?",
            "What correlations exist between numeric columns?",
            "Can you identify any outliers in the data?"
        ]):
            if st.sidebar.button(f"Q{i+1}: {q[:30]}...", key=f"sample_{i}"):
                st.session_state.sample_question = q
        return model_options[selected_model], temperature

    def generate_quick_summary(self, df):
        summary = f"""
        **📊 Dataset Summary:**
        - Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
        - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        - Numeric Columns: {len(df.select_dtypes(include='number').columns)}
        - Text Columns: {len(df.select_dtypes(include='object').columns)}
        - Missing Values: {df.isnull().sum().sum():,}
        """
        st.sidebar.success(summary)

    def show_correlations(self, df):
        numeric = df.select_dtypes(include=['number'])
        if numeric.shape[1] > 1:
            fig = px.imshow(numeric.corr(), text_auto=True, title="Correlation Matrix")
            st.sidebar.plotly_chart(fig, use_container_width=True)
        else:
            st.sidebar.info("Need at least 2 numeric columns.")

    def create_chat_agent(self, df, model, temperature):
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        if not openai_api_key:
            st.error("🔑 Missing OPENAI_API_KEY.")
            return None
        try:
            llm = ChatOpenAI(temperature=temperature, model_name=model, openai_api_key=openai_api_key)
            return create_pandas_dataframe_agent(
                llm, df, verbose=True,
                agent_type="openai-tools",
                handle_parsing_errors=True,
                return_intermediate_steps=False,
                allow_dangerous_code=True
            )
        except Exception as e:
            st.error(f"❌ Error creating AI agent: {str(e)}")
            return None

    def display_chat_message(self, message, is_user=False):
        if is_user:
            st.markdown(f"""<div class="chat-message user-message"><strong>You:</strong> {message}</div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="chat-message bot-message"><strong>🤖 AI:</strong><br>{message}</div>""",
                        unsafe_allow_html=True)

    def run(self):
        self.render_header()
        st.subheader("📁 Upload Your Data")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if uploaded_file:
            df = self.load_data(uploaded_file)
            if df is not None:
                st.success(f"✅ Loaded **{uploaded_file.name}**")
                model, temperature = self.setup_sidebar(df)
                self.display_data_overview(df)
                agent = self.create_chat_agent(df, model, temperature)

                if agent:
                    st.subheader("💬 Chat with Your Data")
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = []

                    user_query = ""
                    is_sample = False
                    if "sample_question" in st.session_state:
                        user_query = st.session_state.sample_question
                        del st.session_state.sample_question
                        is_sample = True
                    else:
                        user_query = st.text_input("Ask a question:", placeholder="e.g. What's the average of column X?")

                    col1, col2 = st.columns([1, 4])
                    with col1:
                        submit_button = st.button("🚀 Ask", type="primary")
                    with col2:
                        if st.button("🗑️ Clear Chat"):
                            st.session_state.chat_history = []
                            st.rerun()

                    if (submit_button or is_sample) and user_query:
                        st.session_state.chat_history.append(("user", user_query))
                        with st.spinner("🤔 Thinking..."):
                            try:
                                start = time.time()
                                response = agent.run(user_query)
                                end = time.time()
                                st.session_state.chat_history.append(("assistant", response))
                                st.caption(f"⏱️ Response time: {end - start:.2f} sec")
                            except Exception as e:
                                error_msg = f"Error: {e}"
                                st.session_state.chat_history.append(("assistant", error_msg))

                    if st.session_state.chat_history:
                        st.subheader("💭 Conversation History")
                        for role, msg in st.session_state.chat_history:
                            self.display_chat_message(msg, is_user=(role == "user"))
                else:
                    st.error("❌ AI agent initialization failed.")
        else:
            st.info("Upload a CSV or Excel file to begin.")

if __name__ == "__main__":
    app = DataChatApp()
    app.run()
