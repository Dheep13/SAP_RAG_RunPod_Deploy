#!/usr/bin/env python3
"""
Optimized LangChain + RunPod + Supabase SAP RAG Integration
Enhanced for CodeLlama 13B with better memory management and error handling
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
        from langchain.vectorstores import SupabaseVectorStore
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
    cache_dir: Optional[str] = None  # Fix: Add this field

    class Config:
        arbitrary_types_allowed = True  # Allow complex types like model objects

    def __init__(self, model_path: str = "codellama/CodeLlama-13b-Instruct-hf", max_memory_gb: int = 20):
        super().__init__()
        self.model_path = model_path
        self.max_memory_gb = max_memory_gb
        self.cache_dir = None  # Initialize here
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
    """Enhanced SAP RAG system with better error handling and performance"""
    
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

            # 2. Initialize embeddings (CPU-based to save GPU memory)
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
        """Initialize embeddings model on CPU"""
        try:
            # Use faster, smaller model for embeddings
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Faster than all-mpnet-base-v2
            
            cache_dir = getattr(self, 'cache_dir', None) or "/tmp/embeddings_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            # In your handler.py, update this:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",  # Match ingestion
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},  # Match normalization
                cache_folder=cache_dir
            )
            
            logger.info(f"✅ Embeddings initialized: {model_name}")
            
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
        """Enhanced document search with better ranking"""
        try:
            # Multi-strategy search
            search_terms = []
            
            # Add original query
            search_terms.append(query)
            
            # Add SAP-specific variations
            sap_keywords = ['CPI', 'iFlow', 'API Management', 'Integration Suite']
            for keyword in sap_keywords:
                if keyword.lower() in query.lower():
                    search_terms.append(f"{keyword} {query}")
            
            # Build search conditions
            search_conditions = []
            for term in search_terms[:3]:  # Limit to avoid too complex queries
                search_conditions.extend([
                    f'content.ilike.%{term}%',
                    f'metadata->>title.ilike.%{term}%'
                ])
            
            # Execute search
            or_condition = f"({','.join(search_conditions)})"
            response = self.supabase_client.table('langchain_documents').select('*').or_(
                or_condition
            ).limit(k * 3).execute()  # Get more to rank better
            
            if not response.data:
                # Fallback: search individual words
                words = [w for w in query.split() if len(w) > 2]
                if words:
                    fallback_conditions = [f'content.ilike.%{word}%' for word in words[:3]]
                    or_condition = f"({','.join(fallback_conditions)})"
                    response = self.supabase_client.table('langchain_documents').select('*').or_(
                        or_condition
                    ).limit(k).execute()
            
            if not response.data:
                return []

            # Enhanced relevance scoring
            results = []
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            for doc in response.data:
                content = doc.get('content', '').lower()
                metadata = doc.get('metadata', {})
                title = metadata.get('title', '').lower()
                source_file = metadata.get('source_file', '').lower()
                
                score = 0
                
                # Exact phrase matches (highest priority)
                if query_lower in title:
                    score += 15
                if query_lower in content:
                    score += 10
                
                # SAP-specific terms boost
                sap_terms = ['cpi', 'iflow', 'api management', 'integration suite', 'groovy']
                for term in sap_terms:
                    if term in query_lower and term in content:
                        score += 5
                
                # Word matches
                title_words = set(title.split())
                content_words = set(content.split())
                
                title_matches = len(query_words.intersection(title_words))
                content_matches = len(query_words.intersection(content_words))
                
                score += title_matches * 4
                score += content_matches * 2
                
                # Document type boost
                if 'tutorial' in source_file or 'guide' in source_file:
                    score += 3
                
                if score > 0:
                    results.append({
                        'id': doc.get('id'),
                        'content': doc.get('content', ''),
                        'metadata': metadata,
                        'relevance_score': score,
                        'similarity': min(score / 20.0, 1.0)  # Normalize
                    })
            
            # Sort and return top results
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"❌ Document search failed: {e}")
            return []

    def answer_question(self, question: str) -> Dict:
        """Generate answer with enhanced context and error handling"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing: {question[:50]}...")
            
            # Search documents
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

            # Build context from top documents
            context_parts = []
            for i, doc in enumerate(documents[:3], 1):
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'SAP Documentation')
                
                context_part = f"""
Document {i}: {title}
Source: {metadata.get('source_file', 'Unknown')}
Relevance: {doc.get('relevance_score', 0)}

Content:
{doc.get('content', '')[:1000]}

---
"""
                context_parts.append(context_part)
            
            context = "\n".join(context_parts)
            
            # Generate answer
            prompt_text = self.prompt.format(context=context, question=question)
            answer = self.llm._call(prompt_text)
            
            processing_time = time.time() - start_time
            
            # Format response
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

# Always start serverless in production
if __name__ == "__main__":
    logger.info("🚀 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})