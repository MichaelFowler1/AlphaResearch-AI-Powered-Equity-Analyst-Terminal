import os
import uuid
import logging
import streamlit as st
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import yfinance as yf
from ddgs import DDGS
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crewai import Agent, Task, Crew, Process, LLM  
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai.tools import tool
from dotenv import load_dotenv

# --- 1. SYSTEM CONFIGURATION & TELEMETRY ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Configuration Error: GOOGLE_API_KEY is missing from the environment variables.")
    st.stop()

# Initialize core LLM engines
agent_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.1
)

chat_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=GOOGLE_API_KEY,
    temperature=0.6 
)

# --- 2. PERSISTENT STORAGE INITIALIZATION ---
@st.cache_resource
def get_databases():
    """Initializes local ChromaDB collections for session history and document RAG."""
    chroma_client = chromadb.PersistentClient(path="./local_rag_storage")
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=GOOGLE_API_KEY, 
        model_name="models/gemini-embedding-001"
    )
    history_col = chroma_client.get_or_create_collection(name="chat_history_v1", embedding_function=google_ef)
    doc_col = chroma_client.get_or_create_collection(name="document_knowledge_base_v1", embedding_function=google_ef)
    return history_col, doc_col

history_collection, doc_collection = get_databases()

# --- 3. ANALYTICAL TOOLSET ---
@tool("Web Search Module")
def web_search_tool(query: str) -> str:
    """Performs comprehensive web searches to extract market data and news catalysts."""
    logging.info(f"Web search initiated: {query}")
    results = DDGS().text(query, max_results=6)
    return str(results)

@tool("Market Data Module")
def stock_price_tool(ticker: str) -> str:
    """Retrieves real-time market pricing and volatility metrics for a specific ticker."""
    logging.info(f"Market data request: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info.get('last_price', 'Unavailable')
        return f"Asset: {ticker} | Current Price: ${price:.2f}"
    except Exception as e:
        return f"Data retrieval failed for {ticker}. Error: {e}"

@tool("Internal Knowledge Search")
def pdf_search_tool(query: str) -> str:
    """Queries the local vector database for context from uploaded research documentation."""
    logging.info("Vector database query executed.")
    try:
        if doc_collection.count() == 0:
            return "Local knowledge base contains no documents."
        
        results = doc_collection.query(query_texts=[query], n_results=3)
        if results['documents'] and results['documents'][0]:
            return "Relevant context from internal documents:\n" + "\n---\n".join(results['documents'][0])
        return "No relevant context found in internal documents."
    except Exception as e:
        return f"Database query error: {str(e)}"

# --- 4. INTERFACE ARCHITECTURE ---
st.set_page_config(page_title="AlphaResearch Terminal", layout="wide")

with st.sidebar:
    st.header("Configuration")
    
    system_persona = st.text_area(
        "Agent Persona / Constraints:",
        value=(
            "You are a Senior Quantitative Equity Analyst and Portfolio Manager. Your objective is to identify "
            "alpha-generating opportunities and construct risk-optimized portfolios. "
            "OPERATIONAL SCOPE: You are authorized to utilize the full spectrum of quantitative strategies. "
            "This includes spot equity, options, statistical arbitrage, hedging, and the tactical use of leverage. "
            "Provide detailed, mathematical, and objective breakdowns of your reasoning, focusing on risk-adjusted "
            "returns, volatility metrics, and structural market inefficiencies."
        ),
        height=280
    )
    
    st.divider()
    st.header("Knowledge Base Management")
    uploaded_file = st.file_uploader("Upload Analysis Document (PDF)", type="pdf")
    
    if uploaded_file:
        file_hash = hash(uploaded_file.getvalue())
        if st.session_state.get('pdf_hash') != file_hash:
            st.session_state.pdf_hash = file_hash
            
            with st.spinner("Processing document..."):
                reader = PdfReader(uploaded_file)
                full_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_text(full_text)
                
                old_ids = doc_collection.get()['ids']
                if old_ids:
                    doc_collection.delete(ids=old_ids)
                    
                new_ids = [str(uuid.uuid4()) for _ in chunks]
                doc_collection.add(documents=chunks, ids=new_ids)
            
            st.success(f"Successfully ingested {len(chunks)} document clusters.")

    st.divider()
    st.header("System Maintenance")

    if 'confirm_purge' not in st.session_state:
        st.session_state.confirm_purge = False

    if not st.session_state.confirm_purge:
        if st.button("Purge Local Databases"):
            st.session_state.confirm_purge = True
            st.rerun()

    if st.session_state.confirm_purge:
        st.warning("Confirmation Required: This action will permanently delete all session history and vector memory.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm Purge", type="primary"):
                st.session_state.messages = []
                if 'pdf_hash' in st.session_state:
                    del st.session_state['pdf_hash']
                
                log_ids = history_collection.get()['ids']
                if log_ids:
                    history_collection.delete(ids=log_ids)
                    
                doc_ids = doc_collection.get()['ids']
                if doc_ids:
                    doc_collection.delete(ids=doc_ids)
                
                st.session_state.confirm_purge = False
                st.success("Storage cleared.")
                
        with col2:
            if st.button("Cancel"):
                st.session_state.confirm_purge = False
                st.rerun()

# --- 5. EXECUTION ENGINE ---
st.title("AlphaResearch: Quantitative Terminal")
st.caption("Mode: Unconstrained Quantitative Analysis & Strategy Deployment")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Input research query or strategy parameters..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Historical Context Retrieval
    memory_query = history_collection.query(query_texts=[prompt], n_results=3)
    chat_context = "\n".join(memory_query['documents'][0]) if memory_query['documents'] else "No prior logs."

    # Knowledge Base Retrieval
    doc_context = ""
    if doc_collection.count() > 0:
        doc_query = doc_collection.query(query_texts=[prompt], n_results=2)
        doc_context = "\n".join(doc_query['documents'][0]) if doc_query['documents'] else ""

    # Workflow Trigger Logic
    action_keywords = ["research", "analyze", "price", "ticker", "invest", "calculate", "model", "recommend", "strategy", "alpha", "hedge"]
    is_action = any(keyword in prompt.lower() for keyword in action_keywords)

    with st.chat_message("assistant"):
        if is_action:
            with st.status("Executing Quantitative Workflow...", expanded=True) as status:
                
                def stream_agent_log(step_output):
                    st.markdown(f"**Execution Log:**\n```text\n{step_output}\n```")
                
                research_agent = Agent(
                    role='Senior Quantitative Analyst',
                    goal='Identify high-conviction trading targets, statistical anomalies, and alpha opportunities across all asset classes.',
                    backstory='You are a high-level quantitative researcher. You leverage data, volatility analysis, and market mechanics to find inefficiencies. You are authorized to evaluate equities and derivatives.',
                    tools=[stock_price_tool, web_search_tool, pdf_search_tool],
                    llm=agent_llm,
                    verbose=True,
                    step_callback=stream_agent_log
                )
                
                portfolio_manager = Agent(
                    role='Quantitative Portfolio Director',
                    goal='Review identified opportunities and formulate a risk-optimized allocation strategy.',
                    backstory='You are a disciplined director of portfolio strategy. You optimize capital efficiency using modern portfolio theory, advanced hedging, and strategic leverage.',
                    llm=agent_llm,
                    verbose=True,
                    step_callback=stream_agent_log
                )

                task_research = Task(
                    description=f"Request: {prompt}. Analyze market data to identify 3 specific opportunities. Include pricing, volatility context, and a quantitative thesis. History: {chat_context}",
                    expected_output="A technical brief on 3 opportunities, including asset types, current pricing, and the underlying mathematical or fundamental thesis.",
                    agent=research_agent
                )
                
                task_allocation = Task(
                    description="Review the analyst brief. Select the primary objective and formulate a strategy. Output 'STRATEGY ALLOCATION: [ASSET/INSTRUMENT]' followed by a technical rationale detailing risk parameters and hedging requirements.",
                    expected_output="A formal strategy recommendation with an explicit allocation target and comprehensive risk-adjusted rationale.",
                    agent=portfolio_manager
                )

                crew = Crew(
                    agents=[research_agent, portfolio_manager], 
                    tasks=[task_research, task_allocation], 
                    process=Process.sequential,
                    max_rpm=10 
                )
                
                response_content = str(crew.kickoff())
                status.update(label="Analysis Complete", state="complete", expanded=False)
            
        else:
            # General query processing
            context_block = f"\n\nINTERNAL CONTEXT:\n{doc_context}" if doc_context else ""
            full_prompt = (
                f"{system_persona}\n"
                f"{context_block}\n\n"
                f"SESSION HISTORY: {chat_context}\n\n"
                f"USER INPUT: {prompt}"
            )
            response = chat_llm.invoke(full_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)

        st.markdown(response_content)

    # Historical Persistence
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    history_collection.add(
        documents=[f"Input: {prompt}", f"Output: {response_content}"],
        ids=[str(uuid.uuid4()), str(uuid.uuid4())]
    )