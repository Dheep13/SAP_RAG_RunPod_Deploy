#!/usr/bin/env python3
"""
LangChain + RunPod + Supabase RAG Integration
Complete RAG system using LangChain components for better maintainability and extensibility
"""

import runpod
import os
import json
import logging
import time
from typing import List, Dict, Optional, Any

# LangChain imports (updated for v0.2+)
try:
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Fallback for older versions
    from langchain.vectorstores import SupabaseVectorStore
    from langchain.embeddings import HuggingFaceEmbeddings

from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Supabase
from supabase import create_client, Client

# Transformers for local model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RunPodCodeLlamaLLM(LLM):
    """Custom LangChain LLM wrapper for RunPod CodeLlama"""

    model_path: str = "codellama/CodeLlama-13b-Instruct-hf"
    tokenizer: Any = None
    model: Any = None
    pipeline: Any = None

    def __init__(self, model_path: str = "codellama/CodeLlama-13b-Instruct-hf"):
        super().__init__()
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load CodeLlama model locally on RunPod with memory optimization"""
        try:
            logger.info(f"Loading CodeLlama model: {self.model_path}")

            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"GPU Memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            # Load tokenizer first (lightweight)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir="/cache"
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with aggressive memory optimizations
            logger.info("Loading model with memory optimizations...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True,  # 8-bit quantization
                low_cpu_mem_usage=True,  # Reduce CPU memory usage
                max_memory={0: "20GB"},  # Reserve memory for other components
                cache_dir="/cache"
            )

            # Create pipeline with memory-efficient settings
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto",
                max_length=2048,  # Limit max length to save memory
                batch_size=1  # Process one at a time
            )

            if torch.cuda.is_available():
                logger.info(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            logger.info("✅ CodeLlama model loaded successfully for LangChain")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        return "runpod_codellama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate response using CodeLlama"""
        try:
            response = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generated_text = response[0]['generated_text'].strip()
            
            # Clean up the response
            if "Answer:" in generated_text:
                generated_text = generated_text.split("Answer:")[-1].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return "I apologize, but I encountered an error generating a response."

class LangChainSAPRAG:
    """Complete SAP RAG system using LangChain components"""
    
    def __init__(self):
        """Initialize LangChain RAG system with memory optimization"""

        # Environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        self.model_path = os.getenv("MODEL_PATH", "codellama/CodeLlama-13b-Instruct-hf")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

        # Initialize components sequentially to manage memory
        logger.info("🚀 Starting sequential component initialization...")
        self._init_components_sequentially()

        logger.info("✅ LangChain SAP RAG system initialized successfully")

    def _init_components_sequentially(self):
        """Initialize components one by one with memory management"""
        try:
            # Step 1: Initialize vector store (lightweight)
            logger.info("📊 Step 1/3: Initializing Supabase vector store...")
            self._init_supabase_vectorstore()

            # Step 2: Clear cache and load LLM (heavy)
            logger.info("🧠 Step 2/3: Loading LLM (this may take a few minutes)...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"GPU Memory before LLM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            self._init_llm()

            # Step 3: Initialize RAG chain
            logger.info("🔗 Step 3/3: Setting up RAG chain...")
            self._init_rag_chain()

            if torch.cuda.is_available():
                logger.info(f"GPU Memory after initialization: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"❌ GPU OOM during initialization: {e}")
            logger.info("🔄 Attempting CPU fallback mode...")
            self._fallback_to_cpu_mode()
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise
    
    def _init_supabase_vectorstore(self):
        """Initialize Supabase vector store with LangChain"""
        try:
            # Supabase client
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            
            # Test connection
            test_response = self.supabase_client.table('langchain_documents').select('id').limit(1).execute()
            logger.info(f"✅ Supabase connected - Found {len(test_response.data)} test documents")
            
            # Embeddings model - always use CPU to save GPU memory
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'},
                cache_folder="/cache"
            )

            logger.info("✅ Supabase vector store initialized")

        except Exception as e:
            logger.error(f"❌ Supabase vector store failed: {e}")
            raise

    def _fallback_to_cpu_mode(self):
        """Fallback to CPU-only mode if GPU OOM occurs"""
        try:
            logger.warning("🔄 Initializing CPU fallback mode...")

            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Initialize a smaller model or CPU-only mode
            self.llm = RunPodCodeLlamaLLM("microsoft/DialoGPT-medium")  # Smaller fallback model

            # Re-initialize embeddings on CPU (already done)
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller embedding model
                model_kwargs={'device': 'cpu'},
                cache_folder="/cache"
            )

            # Initialize RAG chain with CPU components
            self._init_rag_chain()

            logger.info("✅ CPU fallback mode initialized successfully")

        except Exception as e:
            logger.error(f"❌ CPU fallback failed: {e}")
            raise RuntimeError("Both GPU and CPU initialization failed")
    
    def _init_llm(self):
        """Initialize LLM (CodeLlama)"""
        try:
            self.llm = RunPodCodeLlamaLLM(self.model_path)
            logger.info("✅ LLM initialized")
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            raise
    
    def _init_rag_chain(self):
        """Initialize RAG chain with custom SAP prompt"""
        
        # Custom prompt template for SAP Integration Suite
        sap_prompt_template = """You are an expert SAP Integration Suite developer with deep knowledge of Cloud Integration (CPI), API Management, and security best practices.

Use the following SAP documentation to answer the question accurately and provide practical guidance.

SAP Documentation Context:
{context}

Question: {question}

Instructions:
- Base your answer on the provided SAP documentation
- For iFlow development: Include Groovy scripts, message mapping, and adapter configurations
- For security: Include authentication, authorization, and encryption details
- For compliance: Include audit logging and data protection measures
- Provide step-by-step procedures when applicable
- Include code examples if relevant
- If information is incomplete, clearly state what's missing
- Use specific SAP terminology and feature names

Answer:"""
        
        self.prompt = PromptTemplate(
            template=sap_prompt_template,
            input_variables=["context", "question"]
        )
        
        logger.info("✅ RAG chain initialized with SAP-specific prompt")
    
    def search_sap_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search SAP documentation using text-based search"""
        try:
            # Build search conditions for comprehensive matching
            search_conditions = []
            search_conditions.append(f'content.ilike.%{query}%')
            search_conditions.append(f'metadata->>title.ilike.%{query}%')
            
            # Split query into words for better matching
            words = query.lower().split()
            for word in words:
                if len(word) > 2:  # Skip very short words
                    search_conditions.append(f'content.ilike.%{word}%')
            
            # Execute search
            or_condition = f"({','.join(search_conditions)})"
            response = self.supabase_client.table('langchain_documents').select('*').or_(
                or_condition
            ).limit(k * 2).execute()  # Get more to rank
            
            if not response.data:
                return []
            
            # Calculate relevance scores
            results = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for doc in response.data:
                content = doc.get('content', '').lower()
                metadata = doc.get('metadata', {})
                title = metadata.get('title', '').lower()
                
                # Calculate relevance score
                score = 0
                
                # Exact query match (highest score)
                if query_lower in title:
                    score += 10
                if query_lower in content:
                    score += 5
                
                # Word matches
                title_words = set(title.split())
                content_words = set(content.split())
                
                # Title word matches (high value)
                title_matches = len(query_words.intersection(title_words))
                score += title_matches * 3
                
                # Content word matches
                content_matches = len(query_words.intersection(content_words))
                score += content_matches * 1
                
                if score > 0:
                    results.append({
                        'id': doc.get('id'),
                        'content': doc.get('content', ''),
                        'metadata': metadata,
                        'relevance_score': score,
                        'similarity': score / 10.0  # Normalize for compatibility
                    })
            
            # Sort by relevance and return top results
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"❌ Error searching documentation: {e}")
            return []
    
    def get_security_compliance_context(self, query: str) -> str:
        """Get security and compliance context for the query"""
        
        security_keywords = ['security', 'authentication', 'authorization', 'encryption', 'certificate']
        compliance_keywords = ['audit', 'compliance', 'data protection', 'retention']
        
        context_parts = []
        
        # Check if query needs security context
        if any(keyword in query.lower() for keyword in security_keywords):
            security_docs = self.search_sap_documents(f"security {query}", k=2)
            for doc in security_docs:
                context_parts.append(f"**Security Context:**\n{doc['content'][:300]}...")
        
        # Check if query needs compliance context
        if any(keyword in query.lower() for keyword in compliance_keywords):
            compliance_docs = self.search_sap_documents(f"compliance {query}", k=2)
            for doc in compliance_docs:
                context_parts.append(f"**Compliance Context:**\n{doc['content'][:300]}...")
        
        return "\n\n".join(context_parts[:2])  # Limit to avoid token overflow
    
    def answer_question(self, question: str) -> Dict:
        """Answer SAP question using enhanced RAG"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing question: {question[:100]}...")
            
            # Step 1: Search relevant documents
            documents = self.search_sap_documents(question, k=5)
            
            if not documents:
                return {
                    "answer": "I couldn't find relevant SAP documentation for your question.",
                    "sources": [],
                    "documents_found": 0,
                    "status": "no_docs_found"
                }
            
            # Step 2: Build enhanced context
            context_parts = []
            
            # Add security/compliance context if relevant
            security_compliance_context = self.get_security_compliance_context(question)
            if security_compliance_context:
                context_parts.append(security_compliance_context)
            
            # Add main documentation context
            for i, doc in enumerate(documents[:3], 1):  # Limit to top 3 docs
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'SAP Documentation')
                source_file = metadata.get('source_file', 'Unknown')
                doc_type = metadata.get('doc_type', 'documentation')
                relevance = doc.get('relevance_score', 0)
                
                doc_context = f"""
## Document {i}: {title}
**Source:** {source_file} ({doc_type})
**Relevance:** {relevance}

**Content:**
{doc.get('content', '')[:800]}...

---
"""
                context_parts.append(doc_context)
            
            context = "\n".join(context_parts)
            
            # Step 3: Generate response using LangChain prompt
            prompt_text = self.prompt.format(context=context, question=question)
            answer = self.llm(prompt_text)
            
            processing_time = time.time() - start_time
            
            # Step 4: Format response
            response = {
                "answer": answer,
                "question": question,
                "documents_found": len(documents),
                "processing_time": f"{processing_time:.2f}s",
                "confidence": documents[0].get('relevance_score', 0) / 10.0 if documents else 0.0,
                "sources": [
                    {
                        "title": doc.get('metadata', {}).get('title', 'SAP Documentation'),
                        "source_file": doc.get('metadata', {}).get('source_file', 'Unknown'),
                        "doc_type": doc.get('metadata', {}).get('doc_type', 'documentation'),
                        "relevance": doc.get('relevance_score', 0),
                        "preview": doc.get('content', '')[:150] + "..."
                    }
                    for doc in documents[:3]
                ],
                "status": "success"
            }
            
            logger.info(f"✅ Question processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return {
                "answer": f"I encountered an error processing your SAP question: {str(e)}",
                "question": question,
                "error": str(e),
                "status": "error"
            }

# Global RAG system
rag_system = None

def init_langchain_rag():
    """Initialize LangChain RAG system once"""
    global rag_system
    
    try:
        if rag_system is None:
            logger.info("🚀 Initializing LangChain SAP RAG Handler...")
            rag_system = LangChainSAPRAG()
        return rag_system
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        return None

def handler(event):
    """
    RunPod serverless handler using LangChain
    
    Expected input format:
    {
        "input": {
            "query": "How do I create a secure CPI iFlow with audit logging?",
            "max_tokens": 512,
            "component": "CPI"  # Optional component filter
        }
    }
    """
    
    try:
        # Initialize RAG system
        rag = init_langchain_rag()
        if rag is None:
            return {
                "error": "Failed to initialize RAG system",
                "status": "initialization_failed"
            }
        
        # Extract input
        input_data = event.get("input", {})
        question = input_data.get("query", input_data.get("question", ""))
        component = input_data.get("component", None)  # Optional component filter
        
        if not question.strip():
            return {
                "error": "No question provided",
                "status": "invalid_input"
            }
        
        logger.info(f"Processing SAP question: {question[:50]}...")
        
        # Answer question
        result = rag.answer_question(question)
        
        # Add component filter info if used
        if component:
            result["component_filter"] = component
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "status": "handler_error"
        }

# Health check
def health_check():
    """Health check for the LangChain system"""
    global rag_system
    
    return {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "supabase_url": os.getenv("SUPABASE_URL", "Not set")[:50] + "...",
        "model_path": os.getenv("MODEL_PATH", "Default"),
        "langchain_version": "integrated"
    }

# Main execution
if __name__ == "__main__":
    # Test locally
    test_event = {
        "input": {
            "query": "How do I create a secure CPI iFlow with proper authentication and audit logging?",
            "component": "CPI"
        }
    }
    
    print("🧪 Testing LangChain SAP RAG Handler locally...")
    response = handler(test_event)
    print(json.dumps(response, indent=2))
else:
    # Production: Start RunPod serverless
    runpod.serverless.start({"handler": handler})
