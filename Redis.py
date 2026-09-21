import os
from groq import Groq
from redisvl.index import SearchIndex
from redisvl.extensions.cache.llm import SemanticCache

# 1. Initialize OpenAI (for creating meaning/embeddings)
# and Redis (running in RAM)
client = Groq(api_key=["GROQ_API_KEY"])

# 2. Setup the Semantic Cache in Redis
# distance_threshold=0.1 means the questions must be roughly 90% similar in meaning
try:
    llm_cache = SemanticCache(
        name="llm_semantic_cache",
        redis_url="redis://localhost:6379",
        distance_threshold=0.1
    )
except ConnectionError:
    print("❌ Redis Stack is not running. Start Redis Stack first.")
    raise

def ask_ai_bot(user_prompt: str) -> str:
    # --- STEP 1: Check Redis RAM Cache ---
    cached_response = llm_cache.check(prompt=user_prompt)
    
    if cached_response:
        print("⚡ [CACHE HIT] Found similar question in Redis RAM!")
        return cached_response[0]["response"]
        
    # --- STEP 4: Cache Miss -> Go to the expensive LLM/GPU ---
    print("🐢 [CACHE MISS] Querying the expensive LLM...")
    
    # Generate response from OpenAI/LLM
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": user_prompt}]
    )
    actual_answer = response.choices[0].message.content
    
    # Save the answer into Redis RAM for the next user
    llm_cache.store(prompt=user_prompt, response=actual_answer)
    
    return actual_answer

# --- Testing the AI Engineer Pipeline ---
# User 1 asks a question
print(ask_ai_bot("How do I fix a leaky kitchen pipe?"))

# User 2 asks the SAME question but using completely different words!
print(ask_ai_bot("What is the best way to repair a leaking water line in my kitchen?"))
