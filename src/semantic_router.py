import os
import pickle
import faiss
import numpy as np
import torch
import litellm
import torch.nn.functional as F
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
import onnxruntime as ort
import redis
import time
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
import string

class SemanticRouter:
    def __init__(self, model_path="onnx_output", similarity_threshold=0.95):
        """
        Initializes the SemanticRouter, loading the ONNX model, tokenizer, and FAISS index.
        """
        print(f"Loading ONNX model from {model_path}...")
        
        try:
            # 1. Load the Tokenizer and ONNX Model
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.inter_op_num_threads = 2
            
            self.encoder = ORTModelForFeatureExtraction.from_pretrained(
                model_path,
                session_options=sess_options,
                provider="CPUExecutionProvider" 
            )
        except Exception as e:
            print(f"Failed to load ONNX model from {model_path}. Please check the path and dependencies: {e}")
            raise
        
        # MiniLM-L6-v2 always outputs 384-dimensional vectors
        self.dimension = 384 

        self.similarity_threshold = similarity_threshold

        self.index_path = "faiss_cache.index"
        self.map_path = "cache_map.pkl"

        # 2. Initialize or Load FAISS and Cache Map
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.map_path):
                print("Loading existing FAISS index and cache map...")
                self.index = faiss.read_index(self.index_path)
                with open(self.map_path, "rb") as f:
                    self.cache_map = pickle.load(f)
                self.next_id = len(self.cache_map)
            else:
                print("Initializing new FAISS index...")
                self.index = faiss.IndexFlatIP(self.dimension)
                self.cache_map = {} 
                self.next_id = 0
        except Exception as e:
            print(f"Error initializing FAISS index or cache map: {e}")
            print("Falling back to new FAISS index and cache map.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.cache_map = {} 
            self.next_id = 0

        self.redis_conn = redis.from_url(
            url = os.getenv("REDIS_URI"),
            decode_responses=False,  
        )

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.trained_model = joblib.load("trained_model/intent_classification_model.pkl")
        self.label_encoder = joblib.load("trained_model/intent_label_encoder.pkl")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a normalized embedding using the ONNX model.
        It tokenizes the text, runs it through the ONNX model, applies mean pooling,
        and normalizes the resulting vector.
        """
        # 1. Tokenize the input text
        inputs = self.tokenizer(
            [text], 
            padding=True, 
            truncation=True, 
            return_tensors="np"
        )

        # 2. Run inference (ONNX runs natively with NumPy inputs)
        outputs = self.encoder(**inputs)

        # 3. Pure NumPy Mean Pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Expand mask and convert to float
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        
        # Multiply and sum
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        # Divide to pool
        pooled_embeddings = sum_embeddings / sum_mask

        # 4. Pure NumPy L2 Normalization
        norms = np.linalg.norm(pooled_embeddings, axis=1, keepdims=True)
        normalized_embeddings = pooled_embeddings / norms

        # 5. Return FAISS-ready contiguous array
        return np.ascontiguousarray(normalized_embeddings.astype(np.float32))

    def add_to_cache(self, prompt: str, response: str):
        """
        Saves a new prompt-response pair to the FAISS index and cache map.
        Updates the local files to persist the cache.
        """
        try:
            vector = self._get_embedding(prompt)
            
            # Add to FAISS index
            self.index.add(vector)
            
            # Map the internal FAISS ID to the response (simulate Redis)
            self.redis_conn.set(prompt, response)

            # Persist the updated index and cache map to disk
            faiss.write_index(self.index, self.index_path)
            with open(self.map_path, "wb") as f:
                pickle.dump(self.cache_map, f)
        except Exception as e:
            print(f"Failed to add to cache: {e}")

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        # Lowercase the text
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize (split into words)
        tokens = word_tokenize(text)

        # Remove stopwords and Lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words
        ]

        # Join the words back into a single string
        return " ".join(processed_tokens)

    def predict_intents_sklearn(self, new_texts):
        """
        Predicts the intent of new sentences using a Scikit-Learn model.
        """
        # 1. Preprocess the text (Use the same function from your training script!)
        cleaned_text = self.clean_text(new_texts)
        
        # 2. Generate Embeddings
        print("Generating embeddings for new text...")
        embeddings = self._get_embedding(cleaned_text)
        
        # 3. Make Predictions
        numerical_predictions = self.trained_model.predict(embeddings)
        
        # 4. Decode predictions back to original strings ("simple" or "complex")
        string_labels = self.label_encoder.inverse_transform(numerical_predictions)
        
        # Print results nicely
        print("\n--- Prediction Results ---", new_texts, string_labels)
        # for original_text, label in zip(new_texts, string_labels):
        #     print(f"[{label.upper()}] : {original_text}")
            
        return string_labels[0]

    def route_request(self, prompt: str) -> tuple[str, str, str]:
        """
        The main gateway logic for routing requests.
        Checks the semantic cache first. If a match is found, returns the cached response.
        Otherwise, routes to an appropriate LLM based on task complexity, and caches the result.
        Returns a tuple of (response_content, route_type, model_name).
        """
        model_name = ""
        try:
            start_time = time.time()
            vector = self._get_embedding(prompt)
            latency_ms = round((time.time() - start_time) * 1000, 2)
            print(f"[EMBEDDING TIME] {latency_ms}ms")

            # --- STEP 1: CHECK SEMANTIC CACHE ---
            if self.index.ntotal > 0:
                # Search for the 1 nearest neighbor (k=1)
                start_time = time.time()
                similarities, indices = self.index.search(vector, k=1)
                latency_ms = round((time.time() - start_time) * 1000, 2)
                print(f"[FAISS SEARCH TIME] {latency_ms}ms")
                
                best_score = similarities[0][0]

                if best_score >= self.similarity_threshold:
                    print(f"[CACHE HIT] Similarity: {best_score:.4f}")
                    # In production: fetch from Redis using `best_id`
                    return self.redis_conn.get(prompt), "cache hit", "none"
            
            print("[CACHE MISS] Routing to LLM...")

            # --- STEP 2: MODEL ROUTING (Fallback Logic) ---
            # You can use simple heuristics, keyword matching, or a secondary fast classifier here.
            intents = self.predict_intents_sklearn(prompt)

            if intents == "complex":
                start_time = time.time()
                response = self._call_qwen3_groq(prompt)
                latency_ms = round((time.time() - start_time) * 1000, 2)
                print(f"[QWEN-3-32B TIME] {latency_ms}ms")
                model_name = "Qwen-3-32B"
            else:
                start_time = time.time()
                response = self._call_llama3_groq(prompt)
                latency_ms = round((time.time() - start_time) * 1000, 2)
                print(f"[LLAMA-3-8B TIME] {latency_ms}ms")
                model_name = "Llama-3-8B"

            # --- STEP 3: UPDATE CACHE FOR NEXT TIME ---
            content = response.choices[0].message.content
            start_time = time.time()
            self.add_to_cache(prompt, content)
            latency_ms = round((time.time() - start_time) * 1000, 2)
            print(f"[CACHE UPDATE TIME] {latency_ms}ms")
            return content, "llm route", model_name

        except Exception as e:
            print(f"Error in route_request: {e}")
            return f"An error occurred while processing the request: {str(e)}", "error", "none"

    # --- Dummy LLM Callers for Example ---
    def _call_llama3_groq(self, prompt: str):
        """Calls the Llama-3 8B model on Groq for simple tasks."""
        print("-> Routing to Llama 3 8B (Groq) for simple task...")
        return litellm.completion(
            model="groq/llama-3.1-8b-instant", 
            messages=[{"role": "user", "content": prompt}]
        )

    def _call_qwen3_groq(self, prompt: str):
        """Calls the Qwen-3 32B model on Groq for complex reasoning tasks."""
        print("-> Routing to Qwen-3 for complex reasoning...")
        return litellm.completion(
            model="groq/qwen/qwen3-32b", 
            messages=[{"role": "user", "content": prompt}]
        )