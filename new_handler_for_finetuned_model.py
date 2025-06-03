#!/usr/bin/env python3
"""
Enhanced SAP RAG Handler with Fine-tuned CodeLlama Support
Compatible with LangChain ingestion pipeline and vector search
Supports LoRA fine-tuned models from Hugging Face
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

# Transformers and PEFT for fine-tuned models
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline,
    BitsAndBytesConfig,
    GenerationConfig
)

# PEFT for LoRA models
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("⚠️ PEFT not available - fine-tuned model loading may fail")

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

        logger.info(f"🔍 Processing enhanced SAP question: {enhanced_question[:100]}...")
        
        # Process question with enhanced model
        result = rag.answer_question(enhanced_question)
        
        # Add metadata
        if component:
            result["component_filter"] = component
        if include_code:
            result["code_emphasis"] = True
        
        result["model"] = "SAP-CodeLlama-13B-Fine-tuned"
        result["system"] = "Enhanced SAP RAG"
        result["version"] = "2.0"
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": f"Handler error: {str(e)}",
            "status": "handler_error",
            "model": "SAP-CodeLlama-13B-Fine-tuned",
            "system": "Enhanced SAP RAG"
        }

def health_check():
    """Comprehensive health check for enhanced system"""
    global rag_system
    
    storage_info = check_storage_space()
    
    health = {
        "status": "healthy",
        "timestamp": time.time(),
        "rag_initialized": rag_system is not None,
        "gpu_available": torch.cuda.is_available(),
        "peft_available": PEFT_AVAILABLE,
        "model_path": os.getenv("MODEL_PATH", "DheepLearning/sap-codellama-13b-is"),
        "base_model": "codellama/CodeLlama-13b-Instruct-hf",
        "fine_tuned": True,
        "storage": storage_info,
        "system_version": "Enhanced SAP RAG v2.0"
    }
    
    if torch.cuda.is_available():
        health["gpu_info"] = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB"
        }
    
    if rag_system:
        health["model_info"] = {
            "type": rag_system.llm._llm_type,
            "is_fine_tuned": rag_system.llm.is_fine_tuned,
            "model_path": rag_system.llm.model_path,
            "cache_dir": rag_system.llm.cache_dir
        }
    
    return health

def test_sap_questions():
    """Test the enhanced system with SAP-specific questions"""
    test_questions = [
        {
            "query": "How do I create a secure CPI iFlow with OAuth authentication?",
            "component": "CPI",
            "include_code": True
        },
        {
            "query": "What's the best way to handle errors in SAP Integration Suite?",
            "component": "CPI"
        },
        {
            "query": "How do I transform JSON to XML in a CPI integration flow?",
            "include_code": True
        },
        {
            "query": "Create a Python function that processes JSON customer data for SAP integration",
            "include_code": True
        }
    ]
    
    results = []
    for question in test_questions:
        try:
            result = handler({"input": question})
            results.append({
                "question": question["query"],
                "status": result.get("status", "unknown"),
                "confidence": result.get("confidence", 0),
                "processing_time": result.get("processing_time", "unknown"),
                "model_used": result.get("model_used", "unknown"),
                "answer_length": len(result.get("answer", "")),
                "sources_found": result.get("documents_found", 0)
            })
        except Exception as e:
            results.append({
                "question": question["query"],
                "status": "error",
                "error": str(e)
            })
    
    return results

# Main execution
if __name__ == "__main__":
    # Check if running in RunPod environment
    if os.getenv('RUNPOD_ENDPOINT_ID'):
        # Production: Start RunPod serverless
        logger.info("🚀 Starting Enhanced SAP RAG Handler with Fine-tuned Model...")
        runpod.serverless.start({"handler": handler})
    else:
        # Local testing
        print("🧪 Testing Enhanced SAP RAG Handler with Fine-tuned Model...")
        print("=" * 70)
        
        # Test individual question
        test_event = {
            "input": {
                "query": "Create a Python function that processes JSON customer data for SAP integration with proper error handling",
                "component": "CPI",
                "include_code": True
            }
        }
        
        print("📝 Processing test question...")
        response = handler(test_event)
        print(json.dumps(response, indent=2))
        
        print("\n" + "=" * 70)
        print("🏥 Health Check:")
        health = health_check()
        print(json.dumps(health, indent=2))
        
        print("\n" + "=" * 70)
        print("🎯 SAP Question Testing Suite:")
        test_results = test_sap_questions()
        for i, result in enumerate(test_results, 1):
            print(f"\n{i}. {result['question'][:60]}...")
            print(f"   Status: {result['status']}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Time: {result.get('processing_time', 'N/A')}")
            print(f"   Sources: {result.get('sources_found', 0)}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
        
        print("\n" + "=" * 70)
        print("✅ Enhanced SAP RAG Handler testing complete!")
        print("🎯 Your fine-tuned model is ready for SAP Integration Suite questions!")
        
        # Installation instructions
        print("\n📋 Deployment Instructions:")
        print("1. Set environment variables:")
        print("   export MODEL_PATH='DheepLearning/sap-codellama-13b-is'")
        print("   export SUPABASE_URL='your_supabase_url'")
        print("   export SUPABASE_SERVICE_ROLE_KEY='your_key'")
        print("2. Install requirements:")
        print("   pip install runpod transformers torch peft supabase langchain")
        print("3. Deploy to RunPod serverless")
        print("4. Test with SAP Integration Suite questions!").info(f"💾 Storage Info: {json.dumps(storage_info, indent=2)}")
        return storage_info

    except Exception as e:
        logger.error(f"❌ Error checking storage: {e}")
        return {"error": str(e)}

class EnhancedSAPCodeLlamaLLM(LLM):
    """Enhanced CodeLlama LLM with fine-tuned model support for SAP Integration Suite"""

    model_path: str = "DheepLearning/sap-codellama-13b-is"  # Your fine-tuned model
    base_model_path: str = "codellama/CodeLlama-13b-Instruct-hf"
    tokenizer: Any = None
    model: Any = None
    pipeline: Any = None
    generation_config: Any = None
    max_memory_gb: int = 20
    cache_dir: Optional[str] = None
    is_fine_tuned: bool = True

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model_path: str = None, max_memory_gb: int = 20):
        super().__init__()
        
        # Use environment variable or provided path or default
        self.model_path = (
            model_path or 
            os.getenv("MODEL_PATH", "DheepLearning/sap-codellama-13b-is")
        )
        
        self.max_memory_gb = max_memory_gb
        self.cache_dir = None
        
        # Determine if this is a fine-tuned model
        self.is_fine_tuned = self._check_if_fine_tuned()
        
        self._setup_cache_dirs()
        self._load_model()
    
    def _check_if_fine_tuned(self) -> bool:
        """Check if the model is a fine-tuned model"""
        # Simple heuristic: if it's not the base model, assume it's fine-tuned
        return self.model_path != self.base_model_path
    
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
        """Load CodeLlama model with fine-tuned adapter support"""
        try:
            logger.info(f"🚀 Loading SAP CodeLlama model: {self.model_path}")
            logger.info(f"📋 Fine-tuned model: {self.is_fine_tuned}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"🎮 GPU Memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

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

            if self.is_fine_tuned and PEFT_AVAILABLE:
                # Load fine-tuned model with PEFT
                self._load_fine_tuned_model(quantization_config, max_memory)
            else:
                # Load base model
                self._load_base_model(quantization_config, max_memory)

            # Setup generation config optimized for SAP code generation
            self.generation_config = GenerationConfig(
                max_new_tokens=1024,  # Increased for code generation
                temperature=0.2,      # Lower for more deterministic code
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

            logger.info("✅ SAP CodeLlama model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise

    def _load_fine_tuned_model(self, quantization_config, max_memory):
        """Load fine-tuned model with LoRA adapters"""
        try:
            logger.info("🔧 Loading fine-tuned SAP model with LoRA adapters...")
            
            # Load tokenizer from fine-tuned model (has vocabulary adjustments)
            logger.info("📝 Loading fine-tuned tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                cache_dir=self.cache_dir,
                use_fast=True
            )

            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load base model first
            logger.info(f"🧠 Loading base model: {self.base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
                cache_dir=self.cache_dir
            )

            # Resize embeddings to match fine-tuned tokenizer if needed
            current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
            target_vocab_size = len(self.tokenizer)
            
            if current_vocab_size != target_vocab_size:
                logger.info(f"🔧 Resizing embeddings: {current_vocab_size} → {target_vocab_size}")
                base_model.resize_token_embeddings(target_vocab_size)

            # Load fine-tuned adapter
            logger.info(f"🎯 Loading SAP fine-tuned adapter: {self.model_path}")
            self.model = PeftModel.from_pretrained(
                base_model, 
                self.model_path,
                is_trainable=False,
                ignore_mismatched_sizes=True
            )

            logger.info("✅ Fine-tuned SAP model loaded with LoRA adapters")

        except Exception as e:
            logger.error(f"❌ Fine-tuned model loading failed: {e}")
            logger.info("🔄 Falling back to base model...")
            self._load_base_model(quantization_config, max_memory)

    def _load_base_model(self, quantization_config, max_memory):
        """Load base CodeLlama model as fallback"""
        logger.info(f"🧠 Loading base model: {self.base_model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            cache_dir=self.cache_dir
        )

    @property
    def _llm_type(self) -> str:
        return f"sap_codellama_{'fine_tuned' if self.is_fine_tuned else 'base'}"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Generate SAP-optimized response with better formatting"""
        try:
            # Format prompt for CodeLlama with SAP context
            formatted_prompt = f"<s>[INST] You are an expert SAP Integration Suite developer. {prompt} [/INST]"
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,  # Reduced from 4096 for better performance
                padding=False
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with SAP-optimized settings
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

            # Clean up response for SAP context
            response = response.strip()
            
            # Remove common artifacts
            for prefix in ["Answer:", "Response:", "Output:"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            
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
    """Enhanced SAP RAG system with fine-tuned CodeLlama"""
    
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
        self.model_path = os.getenv("MODEL_PATH", "DheepLearning/sap-codellama-13b-is")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) must be set"
            )

        # Initialize components with error handling
        self._init_components()
        logger.info("✅ Enhanced SAP RAG system with fine-tuned model initialized")

    def _init_components(self):
        """Initialize all components with proper error handling"""
        try:
            # 1. Initialize Supabase (lightweight)
            logger.info("📊 Initializing Supabase connection...")
            self._init_supabase()

            # 2. Initialize embeddings (FIXED - Match ingestion pipeline)
            logger.info("📄 Initializing embeddings...")
            self._init_embeddings()

            # 3. Initialize fine-tuned LLM (heavy - do this last)
            logger.info("🧠 Initializing fine-tuned SAP CodeLlama...")
            self._init_llm()

            # 4. Setup enhanced SAP prompt
            logger.info("🔗 Setting up enhanced SAP RAG chain...")
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
            # Use same model and settings as ingestion pipeline
            model_name = "sentence-transformers/all-mpnet-base-v2"
            
            cache_dir = getattr(self, 'cache_dir', None) or "/tmp/embeddings_cache"
            os.makedirs(cache_dir, exist_ok=True)
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=cache_dir
            )
            
            logger.info(f"✅ Embeddings initialized: {model_name} (768 dimensions)")
            
        except Exception as e:
            logger.error(f"❌ Embeddings initialization failed: {e}")
            raise

    def _init_llm(self):
        """Initialize fine-tuned CodeLlama with memory management"""
        try:
            # Calculate available GPU memory
            available_memory = 20  # Default
            if torch.cuda.is_available():
                memory_info = torch.cuda.mem_get_info(0)
                available_memory = int(memory_info[0] / (1024**3)) - 2  # Reserve 2GB
                
            self.llm = EnhancedSAPCodeLlamaLLM(
                model_path=self.model_path,
                max_memory_gb=available_memory
            )
            
            logger.info("✅ Fine-tuned SAP CodeLlama LLM initialized")
            
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            raise

    def _init_prompt(self):
        """Initialize enhanced SAP-specific prompt template"""
        
        self.sap_prompt_template = """You are an expert SAP Integration Suite consultant with deep knowledge of:
- SAP Cloud Platform Integration (CPI)
- SAP API Management
- SAP Process Orchestration
- Security and compliance best practices
- Modern integration patterns and best practices

You have been fine-tuned on SAP Integration Suite documentation and code examples.

Based on the SAP documentation provided below, answer the user's question with practical, actionable guidance.

SAP Documentation Context:
{context}

User Question: {question}

Instructions:
1. Use ONLY information from the provided SAP documentation
2. For integration flows: Include detailed Groovy script examples and adapter configurations
3. For security: Specify authentication methods, certificates, and encryption details
4. For compliance: Include audit logging and data protection measures
5. Provide step-by-step procedures with exact parameter names
6. Use precise SAP terminology and feature names
7. Include code examples when applicable (Python, Groovy, JSON)
8. If information is incomplete, clearly state what's missing and suggest resources
9. Focus on practical, implementable solutions
10. Consider error handling and best practices

Generate a comprehensive, technical response:"""
        
        self.prompt = PromptTemplate(
            template=self.sap_prompt_template,
            input_variables=["context", "question"]
        )
        
        logger.info("✅ Enhanced SAP prompt template configured")

    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Enhanced vector search with SAP-specific optimizations"""
        try:
            logger.info(f"🔍 Performing enhanced vector search for: '{query}'")
            
            # Create SupabaseVectorStore instance
            vector_store = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name="langchain_documents",
                query_name="match_documents"
            )
            
            # Enhance query with SAP context if not already present
            sap_keywords = ['CPI', 'iFlow', 'Integration Suite', 'API Management', 'Groovy']
            if not any(keyword.lower() in query.lower() for keyword in sap_keywords):
                enhanced_query = f"SAP Integration Suite {query}"
            else:
                enhanced_query = query
            
            # Use vector similarity search with scores
            docs_with_scores = vector_store.similarity_search_with_score(enhanced_query, k=k)
            
            results = []
            for doc, score in docs_with_scores:
                # Convert LangChain document format with enhanced metadata
                results.append({
                    'id': doc.metadata.get('id', ''),
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'relevance_score': float(score * 20),
                    'similarity': float(1 - score),
                    'sap_relevance': self._calculate_sap_relevance(doc.page_content)
                })
            
            # Sort by combined relevance and SAP-specific score
            results.sort(key=lambda x: x['relevance_score'] + x['sap_relevance'], reverse=True)
            
            logger.info(f"✅ Enhanced search found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            return []

    def _calculate_sap_relevance(self, content: str) -> float:
        """Calculate SAP-specific relevance score"""
        sap_terms = {
            'integration flow': 3.0, 'iflow': 3.0, 'groovy': 2.5,
            'adapter': 2.0, 'security': 2.0, 'authentication': 2.0,
            'cpi': 2.5, 'api management': 2.0, 'json': 1.5,
            'xml': 1.5, 'rest': 1.5, 'soap': 1.5, 'oauth': 2.0,
            'certificate': 2.0, 'encryption': 2.0, 'error handling': 1.5
        }
        
        content_lower = content.lower()
        score = 0.0
        
        for term, weight in sap_terms.items():
            score += content_lower.count(term) * weight
        
        return min(score, 10.0)  # Cap at 10.0

    def answer_question(self, question: str) -> Dict:
        """Generate enhanced answer with fine-tuned model"""
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Processing SAP question: {question[:50]}...")
            
            # Enhanced document search
            documents = self.search_documents(question, k=6)  # Get more docs for better context
            
            if not documents:
                return {
                    "answer": "I couldn't find relevant SAP Integration Suite documentation for your question. Please try rephrasing with specific SAP terms like 'CPI', 'iFlow', 'API Management', or include the specific SAP component you're working with.",
                    "sources": [],
                    "documents_found": 0,
                    "confidence": 0.0,
                    "status": "no_docs_found",
                    "processing_time": f"{time.time() - start_time:.2f}s",
                    "model_used": self.llm._llm_type
                }

            # Build enhanced context from top documents
            context_parts = []
            for i, doc in enumerate(documents[:4], 1):  # Use top 4 documents
                metadata = doc.get('metadata', {})
                title = metadata.get('title', 'SAP Documentation')
                
                context_part = f"""
Document {i}: {title}
Source: {metadata.get('source_file', 'Unknown')}
SAP Component: {metadata.get('sap_component', 'Unknown')}
Relevance: {doc.get('relevance_score', 0):.1f}
SAP Relevance: {doc.get('sap_relevance', 0):.1f}

Content:
{doc.get('content', '')[:1200]}

---
"""
                context_parts.append(context_part)
            
            context = "\n".join(context_parts)
            
            # Generate answer using fine-tuned CodeLlama
            prompt_text = self.prompt.format(context=context, question=question)
            answer = self.llm._call(prompt_text)
            
            processing_time = time.time() - start_time
            
            # Calculate enhanced confidence score
            base_confidence = min(documents[0].get('relevance_score', 0) / 15.0, 1.0)
            sap_confidence = min(documents[0].get('sap_relevance', 0) / 10.0, 1.0)
            combined_confidence = (base_confidence + sap_confidence) / 2
            
            # Format enhanced response
            response = {
                "answer": answer,
                "question": question,
                "documents_found": len(documents),
                "confidence": combined_confidence,
                "processing_time": f"{processing_time:.2f}s",
                "model_used": self.llm._llm_type,
                "fine_tuned": self.llm.is_fine_tuned,
                "sources": [
                    {
                        "title": doc.get('metadata', {}).get('title', 'SAP Documentation'),
                        "source_file": doc.get('metadata', {}).get('source_file', 'Unknown'),
                        "sap_component": doc.get('metadata', {}).get('sap_component', 'Unknown'),
                        "document_type": doc.get('metadata', {}).get('document_type', 'documentation'),
                        "relevance": doc.get('relevance_score', 0),
                        "sap_relevance": doc.get('sap_relevance', 0),
                        "preview": doc.get('content', '')[:200] + "..."
                    }
                    for doc in documents[:4]
                ],
                "status": "success"
            }
            
            logger.info(f"✅ Enhanced processing completed in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing SAP question: {e}")
            return {
                "answer": f"I encountered an error processing your SAP question: {str(e)}",
                "question": question,
                "error": str(e),
                "status": "error",
                "processing_time": f"{time.time() - start_time:.2f}s",
                "model_used": getattr(self, 'llm', {}).get('_llm_type', 'unknown')
            }

# Global RAG system
rag_system = None

def init_rag_system():
    """Initialize enhanced RAG system with fine-tuned model"""
    global rag_system
    
    try:
        if rag_system is None:
            logger.info("🚀 Initializing Enhanced SAP RAG System with Fine-tuned Model...")
            rag_system = EnhancedSAPRAG()
            logger.info("✅ Enhanced RAG system ready")
        return rag_system
    except Exception as e:
        logger.error(f"❌ RAG system initialization failed: {e}")
        return None

def handler(event):
    """
    Enhanced RunPod serverless handler for SAP RAG system with fine-tuned model
    
    Input format:
    {
        "input": {
            "query": "How do I create a secure CPI iFlow with error handling?",
            "component": "CPI",           # Optional filter
            "max_tokens": 1024,           # Optional
            "temperature": 0.2,           # Optional
            "include_code": true          # Optional - emphasize code examples
        }
    }
    """
    
    try:
        # Initialize enhanced RAG system
        rag = init_rag_system()
        if rag is None:
            return {
                "error": "Failed to initialize enhanced RAG system",
                "status": "initialization_failed"
            }
        
        # Extract and validate input
        input_data = event.get("input", {})
        question = input_data.get("query", input_data.get("question", "")).strip()
        component = input_data.get("component")
        include_code = input_data.get("include_code", False)
        
        if not question:
            return {
                "error": "No question provided. Please include 'query' in your input.",
                "status": "invalid_input",
                "example": {
                    "input": {
                        "query": "How do I create a secure CPI iFlow?",
                        "component": "CPI",
                        "include_code": True
                    }
                }
            }
        
        # Enhance question based on parameters
        enhanced_question = question
        if component:
            enhanced_question = f"[{component}] {question}"
        if include_code:
            enhanced_question += " Please include detailed code examples and configurations."
        
        logger.info(f"🔍 Processing enhanced SAP question: {enhanced_question[:100]}...")
        
        # Process question with enhanced model
        result = rag.answer_question(enhanced_question)
        
        # Add metadata
        if component:
            result["component_filter"] = component
        if include_code:
            result["code_emphasis"] = True
        
        result["model"] = "SAP-CodeLlama-13B-Fine-tuned"
        result["system"] = "Enhanced SAP RAG"
        result["version"] = "2.0"
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": f"Handler error: {str(e)}",
            "status": "handler_error",
            "model": "SAP-CodeLlama-13B-Fine-tuned",
            "system": "Enhanced SAP RAG"
        }

def health_check():
    """Comprehensive health check for enhanced system"""
    global rag_system
    
    storage_info = check_storage_space()
    
    health = {
        "status": "healthy",
        "timestamp": time.time(),
        "rag_initialized": rag_system is not None,
        "gpu_available": torch.cuda.is_available(),
        "peft_available": PEFT_AVAILABLE,
        "model_path": os.getenv("MODEL_PATH", "DheepLearning/sap-codellama-13b-is"),
        "base_model": "codellama/CodeLlama-13b-Instruct-hf",
        "fine_tuned": True,
        "storage": storage_info,
        "system_version": "Enhanced SAP RAG v2.0"
    }
    
    if torch.cuda.is_available():
        health["gpu_info"] = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB",
            "memory_reserved": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB"
        }
    
    if rag_system:
        health["model_info"] = {
            "type": rag_system.llm._llm_type,
            "is_fine_tuned": rag_system.llm.is_fine_tuned,
            "model_path": rag_system.llm.model_path,
            "cache_dir": rag_system.llm.cache_dir
        }
    
    return health

def test_sap_questions():
    """Test the enhanced system with SAP-specific questions"""
    test_questions = [
        {
            "query": "How do I create a secure CPI iFlow with OAuth authentication?",
            "component": "CPI",
            "include_code": True
        },
        {
            "query": "What's the best way to handle errors in SAP Integration Suite?",
            "component": "CPI"
        },
        {
            "query": "How do I transform JSON to XML in a CPI integration flow?",
            "include_code": True
        },
        {
            "query": "Create a Python function that processes JSON customer data for SAP integration",
            "include_code": True
        }
    ]
    
    results = []
    for question in test_questions:
        try:
            result = handler({"input": question})
            results.append({
                "question": question["query"],
                "status": result.get("status", "unknown"),
                "confidence": result.get("confidence", 0),
                "processing_time": result.get("processing_time", "unknown"),
                "model_used": result.get("model_used", "unknown"),
                "answer_length": len(result.get("answer", "")),
                "sources_found": result.get("documents_found", 0)
            })
        except Exception as e:
            results.append({
                "question": question["query"],
                "status": "error",
                "error": str(e)
            })
    
    return results

# Main execution
if __name__ == "__main__":
    # Check if running in RunPod environment
    if os.getenv('RUNPOD_ENDPOINT_ID'):
        # Production: Start RunPod serverless
        logger.info("🚀 Starting Enhanced SAP RAG Handler with Fine-tuned Model...")
        runpod.serverless.start({"handler": handler})
    else:
        # Local testing
        print("🧪 Testing Enhanced SAP RAG Handler with Fine-tuned Model...")
        print("=" * 70)
        
        # Test individual question
        test_event = {
            "input": {
                "query": "Create a Python function that processes JSON customer data for SAP integration with proper error handling",
                "component": "CPI",
                "include_code": True
            }
        }
        
        print("📝 Processing test question...")
        response = handler(test_event)
        print(json.dumps(response, indent=2))
        
        print("\n" + "=" * 70)
        print("🏥 Health Check:")
        health = health_check()
        print(json.dumps(health, indent=2))
        
        print("\n" + "=" * 70)
        print("🎯 SAP Question Testing Suite:")
        test_results = test_sap_questions()
        for i, result in enumerate(test_results, 1):
            print(f"\n{i}. {result['question'][:60]}...")
            print(f"   Status: {result['status']}")
            print(f"   Confidence: {result.get('confidence', 0):.2f}")
            print(f"   Time: {result.get('processing_time', 'N/A')}")
            print(f"   Sources: {result.get('sources_found', 0)}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
        
        print("\n" + "=" * 70)
        print("✅ Enhanced SAP RAG Handler testing complete!")
        print("🎯 Your fine-tuned model is ready for SAP Integration Suite questions!")
        
        # Installation instructions
        print("\n📋 Deployment Instructions:")
        print("1. Set environment variables:")
        print("   export MODEL_PATH='DheepLearning/sap-codellama-13b-is'")
        print("   export SUPABASE_URL='your_supabase_url'")
        print("   export SUPABASE_SERVICE_ROLE_KEY='your_key'")
        print("2. Install requirements:")
        print("   pip install runpod transformers torch peft supabase langchain")
        print("3. Deploy to RunPod serverless")
        print("4. Test with SAP Integration Suite questions!")