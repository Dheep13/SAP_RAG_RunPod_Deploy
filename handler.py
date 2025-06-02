#!/usr/bin/env python3
"""
Complete Fixed SAP RAG Handler
Compatible with LangChain ingestion pipeline and vector search
"""

import runpod
import os
import json
import logging
import time
import shutil
import subprocess
from typing import List, Dict, Optional, Any

# LangChain imports with fallback compatibility
try:
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.language_models.llms import LLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import SupabaseVectorStore
        from langchain.llms.base import LLM
        from langchain.prompts import PromptTemplate
        from langchain.schema import Document
    except ImportError:
        from langchain.vectorstores import SupabaseVectorStore
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.llms.base import LLM
        from langchain.prompts import PromptTemplate
        from langchain.schema import Document

# Supabase
from supabase import create_client, Client

# Transformers for CodeLlama
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig,
    GenerationConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_storage_space():
    """Enhanced storage space checking with better error handling"""
    storage_info = {}
    
    try:
        # Check main filesystem
        total, used, free = shutil.disk_usage("/")
        storage_info["/"] = {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage_percent": round((used / total) * 100, 1)
        }

        # Check runpod volume if it exists
        volume_paths = ["/runpod-volume", "/workspace", "/tmp"]
        for path in volume_paths:
            if os.path.exists(path):
                try:
                    total, used, free = shutil.disk_usage(path)
                    storage_info[path] = {
                        "total_gb": round(total / (1024**3), 2),
                        "used_gb": round(used / (1024**3), 2),
                        "free_gb": round(free / (1024**3), 2),
                        "usage_percent": round((used / total) * 100, 1)
                    }
                except:
                    storage_info[path] = {"status": "ACCESS_DENIED"}
            else:
                storage_info[path] = {"status": "NOT_MOUNTED"}

        # GPU memory info if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_info = torch.cuda.mem_get_info(i)
                storage_info[f"gpu_{i}"] = {
                    "free_gb": round(memory_info[0] / (1024**3), 2),
                    "total_gb": round(memory_info[1] / (1024**3), 2),
                    "used_gb": round((memory_info[1] - memory_info[0]) / (1024**3), 2)
                }

        logger.info(f"💾 Storage Info: {json.dumps(storage_info, indent=2)}")
        return storage_info

    except Exception as e:
        logger.error(f"❌ Error checking storage: {e}")
        return {"error": str(e)}

class OptimizedCodeLlamaLLM(LLM):
    """Optimized CodeLlama LLM for RunPod with better memory management"""

    model_path: str = "codellama/CodeLlama-13b-Instruct-hf"
    tokenizer: Any = None
    model: Any = None
    pipeline: Any = None
    generation_config: Any = None
    max_memory_gb: int = 20
    cache_dir: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model_path: str = "codellama/CodeLlama-13b-Instruct-hf", max_memory_gb: int = 20):
        super().__init__()
        self.model_path = model_path
        self.max_memory_gb = max_memory_gb
        self.cache_dir = None
        self._setup_cache_dirs()
        self._load_model()
    
    def _setup_cache_dirs(self):
        """Setup cache directories for model storage"""
        cache_dirs = ["/runpod-volume", "/workspace", "/tmp"]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir) and os.access(cache_dir, os.W_OK):
                self.cache_dir = os.path.join(cache_dir, "model_cache")
                os.makedirs(self.cache_dir, exist_ok=True)
                logger.info(f"📁 Using cache directory: {self.cache_dir}")
                break
        
        if not self.cache_dir:
            self.cache_dir = "./model_cache"
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.warning("⚠️ Using local cache directory")

    def _load_model(self):
        """Load CodeLlama model with optimized settings"""
        try:
            logger.info(f"🚀 Loading CodeLlama model: {self.model_path}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"🎮 GPU Memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            # Load tokenizer first
            logger.info("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                use_fast=True
            )

            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configure quantization for memory efficiency
            logger.info("⚙️ Configuring model quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True
            )

            # Calculate max memory per GPU
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                max_memory_per_gpu = f"{self.max_memory_gb}GB"
                max_memory = {i: max_memory_per_gpu for i in range(num_gpus)}
                max_memory["cpu"] = "8GB"
            else:
                max_memory = None

            # Load model with optimizations
            logger.info("🧠 Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
                cache_dir=self.cache_dir
            )

            # Setup generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

            # Log memory usage
            if torch.cuda.is_available():
                logger.info(f"🎮 GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

            logger.info("✅ CodeLlama model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "optimized_codellama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate response with better error handling and formatting"""
        try:
            # Format prompt for CodeLlama
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
                padding=False
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with optimized settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Clean up response
            response = response.strip()
            
            # Remove common artifacts
            if response.startswith("Answer:"):
                response = response[7:].strip()
            
            # Stop at stop sequences
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0].strip()

            return response

        except torch.cuda.OutOfMemoryError:
            logger.error("❌ GPU OOM during generation")
            torch.cuda.empty_cache()
            return "I apologize, but I'm experiencing memory constraints. Please try a shorter question."
        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return f"I encountered an error: {str(e)}"

class EnhancedSAPRAG:
    """Enhanced SAP RAG system compatible with LangChain ingestion pipeline"""
    
    def __init__(self):
        """Initialize with robust error handling"""
        
        # Check storage and memory
        logger.info("💾 Checking system resources...")
        self.storage_info = check_storage_space()
        
        # Environment variables with validation
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY") or 
            os.getenv("SUPABASE_ANON_KEY")
        )
        self.model_path = os.getenv("MODEL_PATH", "codellama/CodeLlama-13b-Instruct-hf")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) must be set"
            )

        # Initialize components with error handling
        self._init_components()
        logger.info("✅ Enhanced SAP RAG system initialized")

    def _init_components(self):
        """Initialize all components with proper error handling"""
        try:
            # 1. Initialize Supabase (lightweight)
            logger.info("📊 Initializing Supabase connection...")
            self._init_supabase()

            # 2. Initialize embeddings (FIXED - Match ingestion pipeline)
            logger.info("📄 Initializing embeddings...")
            self._init_embeddings()

            # 3. Initialize LLM (heavy - do this last)
            logger.info("🧠 Initializing CodeLlama LLM...")
            self._init_llm()

            # 4. Setup RAG prompt
            logger.info("🔗 Setting up RAG chain...")
            self._init_prompt()

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _init_supabase(self):
        """Initialize and test Supabase connection"""
        try:
            self.supabase_client = create_client(self.supabase_url, self.supabase_key)
            
            # Test connection
            test_response = self.supabase_client.table('langchain_documents').select('id').limit(1).execute()
            logger.info(f"✅ Supabase connected - Documents available: {len(test_response.data) > 0}")
            
        except Exception as e:
            logger.error(f"❌ Supabase initialization failed: {e}")
            raise

    def _init_embeddings(self):
        """Initialize embeddings to MATCH ingestion pipeline"""
        try:
            # FIXED: Use same model and settings as ingestion pipeline
            model_name = "sentence-transformers/all-mpnet-base-v2"  # Match ingestion
            
            cache_dir = getattr(self, 'cache_dir', None) or "/tmp/embeddings_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Always use CPU for embeddings
                encode_kwargs={'normalize_embeddings': True},  # CRITICAL: Match ingestion
                cache_folder=cache_dir
            )
            
            logger.info(f"✅ Embeddings initialized: {model_name} (768 dimensions)")
            
        except Exception as e:
            logger.error(f"❌ Embeddings initialization failed: {e}")
            raise

    def _init_llm(self):
        """Initialize CodeLlama with memory management"""
        try:
            # Calculate available GPU memory
            available_memory = 20  # Default
            if torch.cuda.is_available():
                memory_info = torch.cuda.mem_get_info(0)
                available_memory = int(memory_info[0] / (1024**3)) - 2  # Reserve 2GB
                
            self.llm = OptimizedCodeLlamaLLM(
                model_path=self.model_path,
                max_memory_gb=available_memory
            )
            
            logger.info("✅ CodeLlama LLM initialized")
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            raise

    def _init_prompt(self):
        """Initialize SAP-specific prompt template"""
        
        self.sap_prompt_template = """You are an expert SAP Integration Suite consultant with deep knowledge of:
- SAP Cloud Platform Integration (CPI)
- SAP API Management
- SAP Process Orchestration
- Security and compliance best practices

Based on the SAP documentation provided below, answer the user's question with practical, actionable guidance.

SAP Documentation:
{context}

User Question: {question}

Instructions:
1. Use ONLY information from the provided SAP documentation
2. For integration flows: Include Groovy script examples and adapter configurations
3. For security: Specify authentication methods, certificates, and encryption
4. For compliance: Include audit logging and data protection measures
5. Provide step-by-step procedures when applicable
6. Use exact SAP terminology and feature names
7. If information is incomplete, clearly state what's missing

Answer:"""
        
        self.prompt = PromptTemplate(
            template=self.sap_prompt_template,
            input_variables=["context", "question"]
        )
        
        logger.info("✅ SAP prompt template configured")

    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Fixed vector search with fallback strategy"""
        try:
            logger.info(f"🔍 Performing vector search for: '{query}'")
            
            # Create SupabaseVectorStore instance
            vector_store = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name="langchain_documents",
                query_name="match_documents"
            )
            
            # TRY METHOD 1: similarity_search_with_score() (ideal but sometimes fails)
            try:
                docs_with_scores = vector_store.similarity_search_with_score(query, k=k)
                
                if docs_with_scores:
                    results = []
                    for doc, score in docs_with_scores:
                        results.append({
                            'id': doc.metadata.get('id', ''),
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'relevance_score': float(score * 20),  # Scale score
                            'similarity': float(1 - score)  # Convert distance to similarity
                        })
                    
                    logger.info(f"✅ similarity_search_with_score found {len(results)} documents")
                    return results
                
            except Exception as score_error:
                logger.warning(f"⚠️ similarity_search_with_score failed: {score_error}")
            
            # FALLBACK METHOD 2: similarity_search() (works perfectly)
            try:
                docs = vector_store.similarity_search(query, k=k)
                
                if docs:
                    results = []
                    for i, doc in enumerate(docs):
                        # Generate synthetic relevance score based on position
                        relevance_score = max(20 - (i * 3), 5)  # Decreasing score by position
                        
                        results.append({
                            'id': doc.metadata.get('id', ''),
                            'content': doc.page_content,
                            'metadata': doc.metadata,
                            'relevance_score': float(relevance_score),
                            'similarity': float(1.0 - (i * 0.1))  # Synthetic similarity
                        })
                    
                    logger.info(f"✅ similarity_search found {len(results)} documents")
                    return results
                
            except Exception as search_error:
                logger.warning(f"⚠️ similarity_search failed: {search_error}")
            
            # FALLBACK METHOD 3: Direct RPC call (manual vector search)
            try:
                query_embedding = self.embeddings.embed_query(query)
                
                response = self.supabase_client.rpc(
                    'match_documents',
                    {
                        'query_embedding': query_embedding,
                        'match_threshold': 0.7,
                        'match_count': k
                    }
                ).execute()
                
                if response.data:
                    results = []
                    for doc in response.data:
                        results.append({
                            'id': doc.get('id'),
                            'content': doc.get('content', ''),
                            'metadata': doc.get('metadata', {}),
                            'relevance_score': float(doc.get('similarity', 0) * 20),
                            'similarity': float(doc.get('similarity', 0))
                        })
                    
                    logger.info(f"✅ Direct RPC search found {len(results)} documents")
                    return results
                    
            except Exception as rpc_error:
                logger.warning(f"⚠️ Direct RPC search failed: {rpc_error}")
            
            # FALLBACK METHOD 4: Text-based search (last resort)
            try:
                # Clean query for text search
                clean_query = query.replace('[', '').replace(']', '').strip()
                
                response = self.supabase_client.table('langchain_documents').select('*').or_(
                    f'content.ilike.%{clean_query}%,metadata->>title.ilike.%{clean_query}%'
                ).limit(k).execute()
                
                if response.data:
                    results = []
                    for i, doc in enumerate(response.data):
                        results.append({
                            'id': doc.get('id'),
                            'content': doc.get('content', ''),
                            'metadata': doc.get('metadata', {}),
                            'relevance_score': float(15 - i),  # Basic scoring
                            'similarity': float(0.8 - (i * 0.1))
                        })
                    
                    logger.info(f"✅ Text search found {len(results)} documents")
                    return results
                    
            except Exception as text_error:
                logger.error(f"❌ Text search failed: {text_error}")
            
            # If all methods fail
            logger.warning("❌ All search methods failed, returning empty results")
            return []
                
        except Exception as e:
            logger.error(f"❌ Vector search system failed: {e}")
            return []

    def answer_question(self, question: str) -> Dict:
        """Generate answer with enhanced context and error handling"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing: {question[:50]}...")
            
            # Search documents using vector similarity
            documents = self.search_documents(question, k=5)
            
            if not documents:
                return {
                    "answer": "I couldn't find relevant SAP documentation for your question. Please try rephrasing with specific SAP terms like 'CPI', 'iFlow', or 'API Management'.",
                    "sources": [],
                    "documents_found": 0,
                    "confidence": 0.0,
                    "status": "no_docs_found",
                    "processing_time": f"{time.time() - start_time:.2f}s"
                }

            # Build context from top documents (LangChain format)
            context_parts = []
            for i, doc in enumerate(documents[:3], 1):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'SAP Documentation')
                
                context_part = f"""
Document {i}: {title}
Source: {metadata.get('source_file', 'Unknown')}
SAP Component: {metadata.get('sap_component', 'Unknown')}
Relevance: {doc.get('relevance_score', 0)}

Content:
{doc.get('content', '')[:1000]}

---
"""
                context_parts.append(context_part)
            
            context = "\n".join(context_parts)
            
            # Generate answer using CodeLlama
            prompt_text = self.prompt.format(context=context, question=question)
            answer = self.llm._call(prompt_text)
            
            processing_time = time.time() - start_time
            
            # Format response with proper metadata extraction
            response = {
                "answer": answer,
                "question": question,
                "documents_found": len(documents),
                "confidence": min(documents[0].get('relevance_score', 0) / 15.0, 1.0),
                "processing_time": f"{processing_time:.2f}s",
                "sources": [
                    {
                        "title": doc.get('metadata', {}).get('title', 'SAP Documentation'),
                        "source_file": doc.get('metadata', {}).get('source_file', 'Unknown'),
                        "sap_component": doc.get('metadata', {}).get('sap_component', 'Unknown'),
                        "document_type": doc.get('metadata', {}).get('document_type', 'documentation'),
                        "relevance": doc.get('relevance_score', 0),
                        "preview": doc.get('content', '')[:150] + "..."
                    }
                    for doc in documents[:3]
                ],
                "status": "success"
            }
            
            logger.info(f"✅ Processed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing question: {e}")
            return {
                "answer": f"I encountered an error processing your SAP question: {str(e)}",
                "question": question,
                "error": str(e),
                "status": "error",
                "processing_time": f"{time.time() - start_time:.2f}s"
            }

# Global RAG system
rag_system = None

def init_rag_system():
    """Initialize RAG system with better error handling"""
    global rag_system
    
    try:
        if rag_system is None:
            logger.info("🚀 Initializing Enhanced SAP RAG System...")
            rag_system = EnhancedSAPRAG()
            logger.info("✅ RAG system ready")
        return rag_system
    except Exception as e:
        logger.error(f"❌ RAG system initialization failed: {e}")
        return None

def handler(event):
    """
    RunPod serverless handler for SAP RAG system
    
    Input format:
    {
        "input": {
            "query": "How do I create a secure CPI iFlow?",
            "component": "CPI",  # Optional filter
            "max_tokens": 512    # Optional
        }
    }
    """
    
    try:
        # Initialize RAG system
        rag = init_rag_system()
        if rag is None:
            return {
                "error": "Failed to initialize RAG system",
                "status": "initialization_failed"
            }
        
        # Extract and validate input
        input_data = event.get("input", {})
        question = input_data.get("query", input_data.get("question", "")).strip()
        component = input_data.get("component")
        
        if not question:
            return {
                "error": "No question provided. Please include 'query' in your input.",
                "status": "invalid_input"
            }
        
        # Add component filter to question if specified
        if component:
            question = f"[{component}] {question}"
        
        logger.info(f"🔍 Processing SAP question: {question[:100]}...")
        
        # Process question
        result = rag.answer_question(question)
        
        # Add metadata
        if component:
            result["component_filter"] = component
        
        result["model"] = "CodeLlama-13B-Instruct"
        result["system"] = "SAP RAG"
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": f"Handler error: {str(e)}",
            "status": "handler_error"
        }

def health_check():
    """Comprehensive health check"""
    global rag_system
    
    storage_info = check_storage_space()
    
    health = {
        "status": "healthy",
        "timestamp": time.time(),
        "rag_initialized": rag_system is not None,
        "gpu_available": torch.cuda.is_available(),
        "model_path": os.getenv("MODEL_PATH", "codellama/CodeLlama-13b-Instruct-hf"),
        "storage": storage_info
    }
    
    if torch.cuda.is_available():
        health["gpu_info"] = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0)
        }
    
    return health

# Main execution
if __name__ == "__main__":
    # Check if running in RunPod environment
    if os.getenv('RUNPOD_ENDPOINT_ID'):
        # Production: Start RunPod serverless
        logger.info("🚀 Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        # Local testing
        test_event = {
            "input": {
                "query": "How do I create a secure CPI iFlow with proper error handling?",
                "component": "CPI"
            }
        }
        
        print("🧪 Testing Enhanced SAP RAG Handler locally...")
        print("=" * 60)
        
        response = handler(test_event)
        print(json.dumps(response, indent=2))
        
        print("\n" + "=" * 60)
        print("🏥 Health Check:")
        health = health_check()
        print(json.dumps(health, indent=2))