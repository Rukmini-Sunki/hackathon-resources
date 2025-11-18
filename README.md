import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import io
import re
from typing import List, Dict, Tuple
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for StudyMate application"""
    CHUNK_SIZE = 500  # Characters per chunk
    CHUNK_OVERLAP = 100  # Overlap between chunks
    TOP_K_RESULTS = 3  # Number of chunks to retrieve
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
    
    # IBM Watsonx Configuration
    WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
    WATSONX_PROJECT_ID = "your_project_id_here"
    WATSONX_API_KEY = "your_api_key_here"
    MODEL_ID = "mistralai/mixtral-8x7b-instruct-v01"

# ============================================================================
# PDF PROCESSING
# ============================================================================

class PDFProcessor:
    """Handles PDF text extraction and preprocessing"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file using PyMuPDF"""
        try:
            pdf_bytes = pdf_file.read()
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            full_text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                full_text += page.get_text()
            
            pdf_document.close()
            return full_text
        except Exception as e:
            st.error(f"Error extracting text from {pdf_file.name}: {str(e)}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\(\)\"\']+', '', text)
        return text.strip()
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = Config.CHUNK_SIZE, 
                   overlap: int = Config.CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.5:  # Only break if reasonable
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if len(c) > 50]  # Filter very short chunks

# ============================================================================
# VECTOR STORE & RETRIEVAL
# ============================================================================

class VectorStore:
    """Manages embeddings and semantic search using FAISS"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.metadata = []
    
    def create_embeddings(self, chunks: List[str], source_names: List[str]):
        """Create embeddings for text chunks and build FAISS index"""
        self.chunks = chunks
        self.metadata = [{"source": name} for name in source_names]
        
        with st.spinner("Creating embeddings..."):
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
            embeddings = np.array(embeddings).astype('float32')
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
            self.index.add(embeddings)
        
        return len(chunks)
    
    def search(self, query: str, top_k: int = Config.TOP_K_RESULTS) -> List[Dict]:
        """Search for most relevant chunks using semantic similarity"""
        if self.index is None:
            return []
        
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "score": float(distance),
                    "source": self.metadata[idx]["source"]
                })
        
        return results

# ============================================================================
# LLM INTEGRATION
# ============================================================================

class WatsonxLLM:
    """IBM Watsonx AI integration for answer generation"""
    
    def __init__(self):
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize IBM Watsonx model"""
        try:
            # Generation parameters
            parameters = {
                GenParams.DECODING_METHOD: "greedy",
                GenParams.MAX_NEW_TOKENS: 500,
                GenParams.MIN_NEW_TOKENS: 50,
                GenParams.TEMPERATURE: 0.7,
                GenParams.TOP_K: 50,
                GenParams.TOP_P: 0.9
            }
            
            self.model = Model(
                model_id=Config.MODEL_ID,
                params=parameters,
                credentials={
                    "url": Config.WATSONX_URL,
                    "apikey": Config.WATSONX_API_KEY
                },
                project_id=Config.WATSONX_PROJECT_ID
            )
        except Exception as e:
            st.error(f"Failed to initialize Watsonx model: {str(e)}")
            st.info("Please ensure your IBM Watsonx credentials are correctly configured.")
    
    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate answer using retrieved context"""
        if not self.model:
            return "⚠️ LLM not initialized. Please check your IBM Watsonx credentials."
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}" 
            for chunk in context_chunks
        ])
        
        # Create prompt
        prompt = f"""You are an academic assistant helping students understand their study materials. Based on the provided context, answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: Provide a clear, accurate answer based solely on the context provided. If the context doesn't contain enough information, say so. Include relevant details and cite the source when appropriate."""

        try:
            response = self.model.generate_text(prompt=prompt)
            return response
        except Exception as e:
            return f"⚠️ Error generating answer: {str(e)}"

# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    if 'llm' not in st.session_state:
        st.session_state.llm = WatsonxLLM()
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    # Page configuration
    st.set_page_config(
        page_title="StudyMate - AI Academic Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📚 StudyMate - AI Academic Assistant")
    st.markdown("""
    Upload your study materials (PDFs) and ask questions in natural language. 
    StudyMate will provide accurate, context-based answers from your documents.
    """)
    
    # Sidebar for document upload and configuration
    with st.sidebar:
        st.header("📁 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files containing your study materials"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
            
            if st.button("🔄 Process Documents", type="primary"):
                process_documents(uploaded_files)
        
        st.divider()
        
        # Configuration options
        with st.expander("⚙️ Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 300, 1000, Config.CHUNK_SIZE)
            top_k = st.slider("Results to Retrieve", 1, 10, Config.TOP_K_RESULTS)
            Config.CHUNK_SIZE = chunk_size
            Config.TOP_K_RESULTS = top_k
        
        st.divider()
        
        # Statistics
        if st.session_state.documents_processed:
            st.subheader("📊 Statistics")
            st.metric("Documents", len(uploaded_files) if uploaded_files else 0)
            st.metric("Text Chunks", len(st.session_state.vector_store.chunks))
        
        st.divider()
        
        # API Configuration
        with st.expander("🔑 IBM Watsonx Setup"):
            st.markdown("""
            **Configure your credentials in the code:**
            1. Set `WATSONX_API_KEY`
            2. Set `WATSONX_PROJECT_ID`
            3. Verify `WATSONX_URL`
            
            [Get IBM Watsonx Credentials →](https://www.ibm.com/watsonx)
            """)
    
    # Main content area
    if not st.session_state.documents_processed:
        st.info("👆 Upload and process PDF documents to get started!")
        
        # Example questions
        st.subheader("💡 Example Questions You Can Ask:")
        st.markdown("""
        - What are the main concepts covered in chapter 3?
        - Explain the theory of relativity based on my notes
        - Summarize the key findings from the research paper
        - What are the differences between X and Y?
        - Define [specific term] from the textbook
        """)
    else:
        # Question input
        st.subheader("💬 Ask a Question")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the main themes in Chapter 2?",
                label_visibility="collapsed"
            )
        with col2:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        # Process question
        if ask_button and question:
            process_question(question)
        
        # Display chat history
        if st.session_state.chat_history:
            st.divider()
            st.subheader("📜 Conversation History")
            
            for idx, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question'][:80]}...", expanded=(idx == 0)):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    
                    if chat.get('sources'):
                        st.markdown("**📚 Sources:**")
                        for source in chat['sources']:
                            st.caption(f"- {source['source']} (Relevance: {source['score']:.2%})")

def process_documents(uploaded_files):
    """Process uploaded PDF documents"""
    processor = PDFProcessor()
    all_chunks = []
    all_sources = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_file in enumerate(uploaded_files):
        status_text.text(f"Processing {pdf_file.name}...")
        
        # Extract text
        text = processor.extract_text_from_pdf(pdf_file)
        
        if text:
            # Clean and chunk text
            cleaned_text = processor.clean_text(text)
            chunks = processor.chunk_text(cleaned_text)
            
            all_chunks.extend(chunks)
            all_sources.extend([pdf_file.name] * len(chunks))
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    # Create embeddings
    if all_chunks:
        num_chunks = st.session_state.vector_store.create_embeddings(all_chunks, all_sources)
        st.session_state.documents_processed = True
        
        status_text.empty()
        progress_bar.empty()
        st.success(f"✅ Successfully processed {len(uploaded_files)} documents into {num_chunks} chunks!")
    else:
        status_text.empty()
        progress_bar.empty()
        st.error("No text could be extracted from the uploaded documents.")

def process_question(question: str):
    """Process user question and generate answer"""
    with st.spinner("Searching documents and generating answer..."):
        # Retrieve relevant chunks
        results = st.session_state.vector_store.search(question, Config.TOP_K_RESULTS)
        
        if not results:
            st.warning("No relevant information found in the documents.")
            return
        
        # Generate answer
        answer = st.session_state.llm.generate_answer(question, results)
        
        # Store in chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": results
        })
        
        # Display answer
        st.markdown("### 💡 Answer")
        st.markdown(answer)
        
        # Display sources
        st.markdown("### 📚 Retrieved Sources")
        for idx, result in enumerate(results, 1):
            with st.expander(f"Source {idx}: {result['source']} (Relevance: {result['score']:.2%})"):
                st.text(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])

if __name__ == "__main__":
    main()
