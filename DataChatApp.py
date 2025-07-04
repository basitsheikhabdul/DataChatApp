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
import io

# Load environment variables
load_dotenv()

class DataChatApp:
    def __init__(self):
        self.setup_page_config()
        self.setup_custom_css()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="DataChat Pro",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def setup_custom_css(self):
        """Apply custom CSS for modern styling"""
        st.markdown("""
        <style>
        /* Main header styling */
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .main-header h1, .main-header p {
            color: white !important;
            margin: 0.5rem 0;
        }
        
        
        /* Chat message styling */
        .chat-message {
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white !important;
            margin-left: 15%;
            border: none;
        }
        
        .user-message strong {
            color: white !important;
        }
        
        .bot-message {
            background: #f8f9fa;
            color: #212529 !important;
            border-left: 4px solid #667eea;
            margin-right: 15%;
        }
        
        .bot-message strong {
            color: #495057 !important;
        }
        
        /* Metric card styling */
        .metric-card {
            background: linear-gradient(145deg, #ffffff, #f8f9fa);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            margin: 1rem 0;
            border: 1px solid rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
        }
        
        .metric-card h2, .metric-card h3 {
            color: #212529 !important;
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
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
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Fix text visibility issues */
        .stMarkdown p, .stMarkdown div, .stText {
            color: #212529 !important;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f8f9fa;
            border-radius: 10px;
            color: #495057;
            padding: 0.5rem 1rem;
            border: 1px solid #dee2e6;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }
        
        /* Success/Error message styling */
        .stSuccess {
            background-color: #d4edda !important;
            border: 1px solid #c3e6cb !important;
            color: #155724 !important;
        }
        
        .stError {
            background-color: #f8d7da !important;
            border: 1px solid #f5c6cb !important;
            color: #721c24 !important;
        }
        
        .stInfo {
            background-color: #d1ecf1 !important;
            border: 1px solid #bee5eb !important;
            color: #0c5460 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 DataChat Pro</h1>
            <p>Intelligent conversations with your data using AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    @st.cache_data
    def load_data(_self, uploaded_file) -> Optional[pd.DataFrame]:
        """Load and cache data from uploaded file"""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    
    def display_data_overview(self, df: pd.DataFrame):
        """Display comprehensive data overview"""
        st.subheader("📊 Data Overview")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">📝 Rows</h3>
                <h2 style="margin: 0.5rem 0;">{:,}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">📋 Columns</h3>
                <h2 style="margin: 0.5rem 0;">{}</h2>
            </div>
            """.format(len(df.columns)), unsafe_allow_html=True)
        
        with col3:
            missing_values = df.isnull().sum().sum()
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">❓ Missing</h3>
                <h2 style="margin: 0.5rem 0;">{:,}</h2>
            </div>
            """.format(missing_values), unsafe_allow_html=True)
        
        with col4:
            numeric_cols = len(df.select_dtypes(include=['number']).columns)
            st.markdown("""
            <div class="metric-card">
                <h3 style="color: #667eea; margin: 0;">🔢 Numeric</h3>
                <h2 style="margin: 0.5rem 0;">{}</h2>
            </div>
            """.format(numeric_cols), unsafe_allow_html=True)
        
        # Data preview tabs
        tab1, tab2, tab3 = st.tabs(["📋 Preview", "📈 Statistics", "🔍 Data Types"])
        
        with tab1:
            st.dataframe(
                df.head(10),
                use_container_width=True,
                height=400
            )
        
        with tab2:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                st.dataframe(
                    numeric_df.describe(),
                    use_container_width=True
                )
            else:
                st.info("No numeric columns found for statistical summary.")
        
        with tab3:
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
    
    def setup_sidebar(self, df: pd.DataFrame):
        """Setup sidebar with additional options"""
        st.sidebar.header("⚙️ Settings")
        
        # Model selection
        model_options = {
            "GPT-3.5 Turbo (Fast & Cheap)": "gpt-3.5-turbo",
            "GPT-4 (More Accurate)": "gpt-4",
            "GPT-4 Turbo": "gpt-4-turbo-preview"
        }
        
        selected_model = st.sidebar.selectbox(
            "🧠 Select AI Model",
            options=list(model_options.keys()),
            index=0
        )
        
        # Temperature setting
        temperature = st.sidebar.slider(
            "🌡️ Response Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="Higher values make responses more creative but less focused"
        )
        
        # Quick insights
        st.sidebar.header("💡 Quick Insights")
        
        if st.sidebar.button("📊 Generate Summary"):
            self.generate_quick_summary(df)
        
        if st.sidebar.button("📈 Show Correlations"):
            self.show_correlations(df)
        
        # Sample questions
        st.sidebar.header("❓ Sample Questions")
        sample_questions = [
            "What are the key statistics of this dataset?",
            "Show me the distribution of the main columns",
            "Are there any missing values I should know about?",
            "What correlations exist between numeric columns?",
            "Can you identify any outliers in the data?"
        ]
        
        for i, question in enumerate(sample_questions):
            if st.sidebar.button(f"Q{i+1}: {question[:30]}...", key=f"sample_{i}"):
                st.session_state.sample_question = question
                st.rerun()  # Force rerun to process the sample question immediately
        
        return model_options[selected_model], temperature
    
    def generate_quick_summary(self, df: pd.DataFrame):
        """Generate quick data summary"""
        summary = f"""
        **📊 Dataset Summary:**
        - **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns
        - **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB
        - **Numeric Columns:** {len(df.select_dtypes(include=['number']).columns)}
        - **Text Columns:** {len(df.select_dtypes(include=['object']).columns)}
        - **Missing Values:** {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/df.size*100:.1f}%)
        """
        
        st.sidebar.success(summary)
    
    def show_correlations(self, df: pd.DataFrame):
        """Show correlation matrix for numeric columns"""
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdBu"
            )
            
            st.sidebar.plotly_chart(fig, use_container_width=True)
        else:
            st.sidebar.info("Need at least 2 numeric columns for correlations")
    
    def create_chat_agent(self, df: pd.DataFrame, model: str, temperature: float):
        """Create and configure the pandas dataframe agent"""
        openai_api_key = st.secrets["OPENAI_API_KEY"]
        
        if not openai_api_key:
            st.error("🔑 OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")
            return None
        
        try:
            llm = ChatOpenAI(
                temperature=temperature,
                model_name=model,
                openai_api_key=openai_api_key
            )
            
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=True,
                agent_type="openai-tools",
                handle_parsing_errors=True,
                return_intermediate_steps=False,
                allow_dangerous_code=True
            )
            
            return agent
            
        except Exception as e:
            st.error(f"❌ Error creating AI agent: {str(e)}")
            return None
    
    def display_chat_message(self, message: str, is_user: bool = False):
        """Display chat message with styling"""
        if is_user:
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 AI Assistant:</strong><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
    
    def run(self):
        """Main application logic"""
        self.render_header()
        
        # File upload section
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📁 Upload Your Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx"],
            help="Upload your data file to start chatting with it using AI"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            # Load data
            df = self.load_data(uploaded_file)
            
            if df is not None:
                st.success(f"✅ Successfully loaded **{uploaded_file.name}**!")
                
                # Setup sidebar and get configuration
                model, temperature = self.setup_sidebar(df)
                
                # Display data overview
                self.display_data_overview(df)
                
                # Create chat agent
                agent = self.create_chat_agent(df, model, temperature)
                
                if agent:
                    st.subheader("💬 Chat with Your Data")
                    
                    # Initialize chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    # Handle user input - check for sample question first
                    user_query = ""
                    if 'sample_question' in st.session_state:
                        user_query = st.session_state.sample_question
                        # Clear the sample question after using it
                        del st.session_state.sample_question
                        # Display the sample question in the text input
                        st.text_input(
                            "Ask a question about your data:",
                            value=user_query,
                            placeholder="e.g., What's the average of column X? Show me trends in the data...",
                            key="user_input_display",
                            disabled=True
                        )
                    else:
                        user_query = st.text_input(
                            "Ask a question about your data:",
                            placeholder="e.g., What's the average of column X? Show me trends in the data...",
                            key="user_input"
                        )
                    
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        submit_button = st.button("🚀 Ask", type="primary", use_container_width=True)
                    with col2:
                        if st.button("🗑️ Clear Chat", use_container_width=True):
                            st.session_state.chat_history = []
                            st.rerun()
                    
                    # Process the query if there's input
                    if user_query and (submit_button or 'sample_question' in st.session_state):
                        # Add user message to history
                        st.session_state.chat_history.append(("user", user_query))
                        
                        with st.spinner("🤔 AI is thinking..."):
                            try:
                                start_time = time.time()
                                response = agent.run(user_query)
                                end_time = time.time()
                                
                                # Add AI response to history
                                st.session_state.chat_history.append(("assistant", response))
                                
                                # Show response time
                                st.caption(f"⏱️ Response time: {end_time - start_time:.2f} seconds")
                                
                            except Exception as e:
                                error_msg = f"Sorry, I encountered an error: {str(e)}"
                                st.session_state.chat_history.append(("assistant", error_msg))
                        
                        # Clear the input after processing
                        if 'user_input' in st.session_state:
                            del st.session_state.user_input
                        st.rerun()
                    
                    # Display chat history
                    if st.session_state.chat_history:
                        st.subheader("💭 Conversation History")
                        for role, message in st.session_state.chat_history:
                            self.display_chat_message(message, is_user=(role == "user"))
                
                else:
                    st.error("❌ Could not initialize AI agent. Please check your API key and try again.")
        
        else:
            # Welcome message
            st.info("""
            👋 **Welcome to DataChat Pro!**
            
            Upload your CSV or Excel file to start having intelligent conversations with your data.
            
            **What you can do:**
            - Ask questions about your data in natural language
            - Get statistical insights and summaries
            - Create visualizations and charts
            - Identify patterns and trends
            - Find outliers and anomalies
            
            **Example questions:**
            - "What's the average sales by region?"
            - "Show me the top 10 customers by revenue"
            - "Are there any missing values in my dataset?"
            - "Create a chart showing monthly trends"
            """)

if __name__ == "__main__":
    app = DataChatApp()
    app.run()
