import streamlit as st
import os
import warnings
import logging
from typing import List, Tuple, Optional, Dict, Any
import random
import json
import base64
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import time
import pickle
import numpy as np
from dotenv import load_dotenv
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
import faiss
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep warnings quiet
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Enhanced API key management
def get_api_key():
    """
    Get API key from multiple sources with proper fallback
    Priority: Streamlit Secrets -> Environment Variable -> .env file
    """
    # Load environment variables from .env file (for local development)
    load_dotenv()
    
    api_key = None
    
    # 1. Try Streamlit Secrets first (for deployed apps)
    try:
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
            logger.info("✅ API key loaded from Streamlit Secrets")
    except Exception as e:
        logger.warning(f"Could not access Streamlit secrets: {e}")
    
    # 2. Fallback to environment variable
    if not api_key:
        api_key = os.getenv('GROQ_API_KEY')
        if api_key:
            logger.info("✅ API key loaded from environment variable")
    
    # 3. Check if key is valid
    if not api_key or api_key.strip() == "" or api_key == "your_api_key_here":
        return None
    
    return api_key.strip()

# Get the API key
groq_key = get_api_key()

class Config:
    """Centralized configuration management"""
    GROQ_API_KEY = groq_key
    DEFAULT_MODEL = "groq/llama-3.1-8b-instant"
    MEMORY_DIR = "./jumbo_memory"
    MAX_MEMORIES_PER_QUERY = 5
    MEMORY_CLEANUP_DAYS = 30
    MIN_RELEVANCE_THRESHOLD = 0.6
    MAX_MESSAGE_HISTORY = 50
    DEFAULT_ASSETS_DIR = "./assets"
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    CACHE_TTL = 300  # 5 minutes
    EMBEDDING_DIM = 384  # Dimension for sentence embeddings

MOOD_KEYWORDS = {
    "happy": ["happy", "good", "great", "wonderful", "amazing", "fantastic", "excellent", "joy", "joyful", "elated", "thrilled", "overjoyed", "blessed", "grateful", "content", "cheerful"],
    "excited": ["excited", "pumped", "energetic", "enthusiastic", "awesome", "brilliant", "stoked", "hyped", "fired up", "motivated"],
    "sad": ["sad", "upset", "depressed", "hurt", "pain", "down", "devastated", "heartbroken", "grief", "melancholy", "blue", "disappointed"],
    "anxious": ["anxious", "worried", "scared", "nervous", "stressed", "overwhelmed", "panicked", "stress", "stressful", "tense", "restless", "uneasy"],
    "angry": ["angry", "frustrated", "mad", "annoyed", "irritated", "furious", "pissed", "rage", "outraged", "livid", "resentful"],
    "lost": ["lost", "confused", "uncertain", "unsure", "stuck", "directionless", "bewildered", "perplexed", "aimless"],
    "work_stress": ["work", "job", "career", "office", "boss", "colleague", "deadline", "workload", "promotion", "interview", "meeting", "project", "role", "transition", "burnout"],
    "lonely": ["lonely", "alone", "isolated", "disconnected", "abandoned", "solitary", "empty"],
    "proud": ["proud", "accomplished", "achieved", "successful", "victorious", "triumphant"]
}

JUMBO_RESPONSES = {
    "happy": [
        "That sounds really wonderful! You deserve to feel this good.",
        "I can hear the happiness in your words. It's beautiful to see you shine.",
        "You sound so content right now. That warmth is contagious.",
        "What a lovely thing to share. You seem genuinely happy.",
        "Your joy is radiating through your words. This is beautiful to witness."
    ],
    "excited": [
        "Your excitement is absolutely infectious! You sound ready to take on the world.",
        "I can feel your energy through your words. That enthusiasm is amazing.",
        "You sound pumped up and ready for whatever's coming. That's awesome.",
        "Your passion is lighting up our conversation. Keep that fire burning!"
    ],
    "sad": [
        "It sounds like you're going through something really tough right now.",
        "You seem to be carrying some heavy feelings. That must be hard.",
        "I can hear the sadness in your words. You don't have to go through this alone.",
        "It sounds like things feel pretty overwhelming right now.",
        "Your pain is valid and real. I'm here to sit with you through this."
    ],
    "anxious": [
        "It sounds like your mind is racing with worry right now.",
        "You seem really unsettled about this. That anxiety must be exhausting.",
        "I can hear how much this is weighing on you. You don't have to face it alone.",
        "It sounds like you're feeling pretty overwhelmed. That's completely understandable.",
        "Your worries make complete sense given what you're facing."
    ],
    "angry": [
        "You sound really frustrated about this. That anger makes complete sense.",
        "It sounds like something really got under your skin. You have every right to feel upset.",
        "I can hear how irritated you are. Those feelings are totally valid.",
        "Your frustration is completely justified. Anyone would feel this way."
    ],
    "lost": [
        "You seem unsure about where things are heading, and that can be really unsettling.",
        "It sounds like you're in a confusing place right now. That uncertainty is tough.",
        "You seem to be searching for direction. It's okay not to have all the answers.",
        "Feeling lost is part of the human experience. You're not alone in this uncertainty."
    ],
    "work_stress": [
        "Work stress can be incredibly draining. You're handling a lot right now.",
        "Career transitions and workplace pressure are genuinely challenging.",
        "It sounds like your professional life is weighing heavily on you right now.",
        "Work-life balance is tough, and you're navigating some real challenges."
    ],
    "lonely": [
        "Loneliness can feel so heavy. You're reaching out, and that takes courage.",
        "Feeling disconnected is one of the hardest human experiences.",
        "You're not as alone as you feel right now. I'm here with you.",
        "Isolation is painful, but you're taking steps to connect by being here."
    ],
    "proud": [
        "You should absolutely celebrate this achievement! You've earned this feeling.",
        "Your sense of accomplishment comes through so clearly. Well done!",
        "It's wonderful to hear you recognizing your own success.",
        "You sound genuinely proud, and that pride is well-deserved."
    ],
    "neutral": [
        "I hear you. Thanks for sharing that with me.",
        "You're on my mind. What you're going through matters.",
        "I'm glad you felt comfortable sharing that.",
        "You don't have to carry this alone."
    ],
    "trust": [
        "I'm Jumbo, and you can absolutely trust me as your friend. This conversation is completely private between us.",
        "You can trust me, Jumbo, completely. No one else has access to what you share with me - it's just between friends.",
        "I'm Jumbo, your friend, and this space is totally safe. What you tell me stays between us, always.",
        "You can feel safe opening up to me - I'm Jumbo, and our conversations are private and secure."
    ]
}

def enhanced_mood_detection(text: str) -> Tuple[str, float]:
    """Enhanced mood detection using both keywords and sentiment analysis"""
    try:
        text_lower = text.lower()
        trust_keywords = ["trust", "safe", "safety", "secure", "private", "confidential", "remember", "friend"]
        if any(keyword in text_lower for keyword in trust_keywords):
            return "trust", 0.9
        mood_scores = {}
        for mood, keywords in MOOD_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                mood_scores[mood] = score / len(keywords)
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            if polarity > 0.3:
                sentiment_mood = "happy"
                sentiment_confidence = min(polarity + 0.5, 1.0)
            elif polarity < -0.3:
                sentiment_mood = "sad"
                sentiment_confidence = min(abs(polarity) + 0.5, 1.0)
            elif subjectivity > 0.7:
                sentiment_mood = "anxious"
                sentiment_confidence = subjectivity
            else:
                sentiment_mood = "neutral"
                sentiment_confidence = 0.5
            if mood_scores:
                top_keyword_mood = max(mood_scores.items(), key=lambda x: x[1])
                if top_keyword_mood[1] > sentiment_confidence:
                    return top_keyword_mood, top_keyword_mood[1]
                else:
                    return sentiment_mood, sentiment_confidence
            else:
                return sentiment_mood, sentiment_confidence
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            if mood_scores:
                top_mood = max(mood_scores.items(), key=lambda x: x[1])
                return top_mood, top_mood[1]
        return "neutral", 0.5
    except Exception as e:
        logger.error(f"Mood detection failed: {e}")
        return "neutral", 0.5

def extract_name_from_text(text: str) -> Optional[str]:
    """Enhanced name extraction with better patterns"""
    text_clean = text.lower().strip()
    patterns = [
        (r"my name is ([a-zA-Z]+)", 1),
        (r"i'm ([a-zA-Z]+)", 1),
        (r"i am ([a-zA-Z]+)", 1),
        (r"call me ([a-zA-Z]+)", 1),
        (r"name's ([a-zA-Z]+)", 1),
        (r"i go by ([a-zA-Z]+)", 1)
    ]
    import re
    for pattern, group in patterns:
        match = re.search(pattern, text_clean)
        if match:
            name = match.group(group).strip()
            if len(name) > 1 and name.isalpha():
                return name.title()
    return None

def make_llm(groq_api_key: str, model: str = "groq/llama-3.1-8b-instant") -> ChatGroq:
    if not groq_api_key:
        raise ValueError("No Groq API key found.")
    return ChatGroq(
        groq_api_key=groq_api_key.strip(),
        model=model,
        temperature=0.7,
        max_tokens=500,
        timeout=None
    )

def simple_text_embedding(text: str, dim: int = 384) -> np.ndarray:
    """
    Create a simple text embedding using basic text features
    This is a fallback when proper sentence embeddings aren't available
    """
    # Convert text to lowercase and split into words
    words = text.lower().split()
    
    # Create a hash-based embedding
    embedding = np.zeros(dim)
    
    for i, word in enumerate(words):
        # Use hash of word to create pseudo-random but consistent features
        word_hash = hash(word) % dim
        embedding[word_hash] += 1.0 / (i + 1)  # Weight by position
    
    # Add some basic text statistics
    embedding[0] = len(words)  # Word count
    embedding[1] = len(text)   # Character count
    embedding[2] = len(set(words))  # Unique word count
    
    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.astype('float32')

class EnhancedJumboMemory:
    """Enhanced memory system using FAISS for vector storage"""
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        self.memory_dir = Config.MEMORY_DIR
        self.embedding_dim = Config.EMBEDDING_DIM
        
        # Create memory directory
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # File paths for this user
        self.user_hash = hashlib.md5(user_id.encode()).hexdigest()[:8]
        self.index_path = os.path.join(self.memory_dir, f"faiss_index_{self.user_hash}.index")
        self.metadata_path = os.path.join(self.memory_dir, f"metadata_{self.user_hash}.pkl")
        self.user_info_path = os.path.join(self.memory_dir, f"user_info_{self.user_hash}.json")
        
        # Initialize FAISS index
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Created new FAISS index")
            
            # Load metadata
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                self.metadata = []
                
            # Load user info
            if os.path.exists(self.user_info_path):
                with open(self.user_info_path, 'r') as f:
                    self.user_info = json.load(f)
            else:
                self.user_info = {}
                
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            # Create fresh instances
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            self.user_info = {}

    def _save_to_disk(self):
        """Save index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            with open(self.user_info_path, 'w') as f:
                json.dump(self.user_info, f)
        except Exception as e:
            logger.error(f"Failed to save memory to disk: {e}")

    def store_conversation(self, user_message: str, jumbo_response: str, mood: str, 
                          confidence: float, user_name: str = None) -> bool:
        """Store conversation with enhanced metadata"""
        try:
            conversation_text = f"User: {user_message}\nJumbo: {jumbo_response}"
            
            # Create embedding
            embedding = simple_text_embedding(conversation_text, self.embedding_dim)
            
            # Create metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "mood": mood,
                "confidence": confidence,
                "type": "conversation",
                "message_length": len(user_message),
                "response_length": len(jumbo_response),
                "user_message": user_message,
                "jumbo_response": jumbo_response
            }
            if user_name:
                metadata["user_name"] = user_name
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1))
            self.metadata.append(metadata)
            
            # Save to disk
            self._save_to_disk()
            
            # Cleanup old memories
            self._cleanup_old_memories()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store conversation: {e}")
            return False

    def _cleanup_old_memories(self):
        """Clean up old memories to prevent database bloat"""
        try:
            cutoff_date = datetime.now() - timedelta(days=Config.MEMORY_CLEANUP_DAYS)
            cutoff_str = cutoff_date.isoformat()
            
            # Find indices to keep
            indices_to_keep = []
            new_metadata = []
            
            for i, meta in enumerate(self.metadata):
                if meta.get("timestamp", "") >= cutoff_str:
                    indices_to_keep.append(i)
                    new_metadata.append(meta)
            
            # If we need to remove some memories
            if len(indices_to_keep) < len(self.metadata):
                logger.info(f"Cleaning up {len(self.metadata) - len(indices_to_keep)} old memories")
                
                # Rebuild index with only recent memories
                if indices_to_keep:
                    # Get vectors for indices to keep
                    vectors = []
                    for i in indices_to_keep:
                        vector = self.index.reconstruct(i)
                        vectors.append(vector)
                    
                    # Create new index
                    new_index = faiss.IndexFlatL2(self.embedding_dim)
                    if vectors:
                        vectors_array = np.array(vectors).astype('float32')
                        new_index.add(vectors_array)
                    
                    self.index = new_index
                    self.metadata = new_metadata
                    self._save_to_disk()
                else:
                    # No memories to keep - reset everything
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                    self.metadata = []
                    self._save_to_disk()
                    
        except Exception as e:
            logger.warning(f"Memory cleanup failed: {e}")

    def store_user_info(self, info_type: str, info_value: str):
        """Store specific user information like name"""
        try:
            self.user_info[info_type] = {
                "value": info_value,
                "timestamp": datetime.now().isoformat()
            }
            self._save_to_disk()
        except Exception as e:
            logger.error(f"Failed to store user info: {e}")

    def get_user_name(self) -> Optional[str]:
        """Retrieve user's name if stored"""
        try:
            name_info = self.user_info.get("name")
            if name_info:
                return name_info["value"]
        except Exception as e:
            logger.error(f"Failed to get user name: {e}")
        return None

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """Retrieve relevant past conversations"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Create query embedding
            query_embedding = simple_text_embedding(query, self.embedding_dim)
            
            # Search for similar conversations
            k = min(limit, self.index.ntotal)
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            memories = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    memory_data = self.metadata[idx].copy()
                    memory_data["distance"] = float(distance)
                    memory_data["similarity"] = 1.0 / (1.0 + distance)  # Convert distance to similarity
                    memories.append(memory_data)
            
            # Sort by similarity (higher is better)
            memories.sort(key=lambda x: x["similarity"], reverse=True)
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []

    def clear_memory(self) -> bool:
        """Clear all stored memories and user info"""
        try:
            # Reset index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            self.user_info = {}
            
            # Remove files
            for path in [self.index_path, self.metadata_path, self.user_info_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info("Memory cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False

class EnhancedJumboCrew:
    """Enhanced emotional wellness chatbot with improved error handling and caching"""
    def __init__(self, groq_api_key: Optional[str] = None, user_id: str = "default_user"):
        self.user_id = user_id
        self.api_key = groq_api_key or Config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("No Groq API key found.")
        try:
            self.llm = make_llm(self.api_key)
            self.memory = EnhancedJumboMemory(user_id)
            self.listener, self.companion, self.summariser = self._create_agents()
            self._cached_crew = None
        except Exception as e:
            logger.error(f"Failed to initialize EnhancedJumboCrew: {e}")
            raise

    def _create_agents(self) -> Tuple[Agent, Agent, Agent]:
        """Create enhanced agents with better error handling"""
        try:
            listener_agent = Agent(
                role="Advanced Emotion Detector",
                goal="Identify emotional states, context, and support needs with high accuracy, considering conversation history and user patterns.",
                backstory="""You are an expert in emotional intelligence and human psychology. You can detect subtle emotional cues, understand context from past conversations, and identify what kind of support would be most helpful. You consider both explicit statements and implicit feelings, analyzing mood patterns over time.""",
                verbose=False,
                llm=self.llm
            )
            companion_agent = Agent(
                role="Jumbo the Wise Elephant Companion", 
                goal="Provide empathetic, personalized responses that make users feel heard, understood, and supported while maintaining conversation continuity.",
                backstory="""You are Jumbo, a wise and caring elephant with an excellent memory and deep emotional intelligence. You remember past conversations and use this knowledge to provide personalized, continuous support.

Your core principles:
- Always use "you" language to reflect the user's feelings
- Never impose your own feelings with "I feel" statements
- Remember and reference past conversations naturally
- Introduce yourself as "Jumbo" when discussing trust/safety
- Provide specific, relevant responses (not generic emotional validation)
- Match the emotional tone appropriately
- Use the user's name when you know it
- Be warm, conversational, and genuine
- Ask thoughtful follow-up questions
- Maintain user privacy and confidentiality

You have perfect memory of past conversations and can reference them to provide continuity and show that you truly care about the user's journey.""",
                verbose=False,
                llm=self.llm
            )
            summariser_agent = Agent(
                role="Response Quality Enhancer",
                goal="Ensure responses are perfectly crafted - empathetic, conversational, appropriately emotional, and end with thoughtful questions.",
                backstory="""You are responsible for ensuring every response meets the highest standards of emotional support. You refine responses to be natural, warm, and conversational while maintaining therapeutic value. You ensure proper "you" language, appropriate emotional matching, and meaningful follow-up questions.""",
                verbose=False,
                llm=self.llm
            )
            return listener_agent, companion_agent, summariser_agent
        except Exception as e:
            logger.error(f"Failed to create agents: {e}")
            raise

    def respond(self, user_message: str, max_retries: int = Config.MAX_RETRIES) -> Tuple[str, Dict]:
        """Generate response with enhanced error handling and retries"""
        response_metadata = {
            "timestamp": datetime.now().isoformat(),
            "processing_time": 0,
            "mood_detected": "neutral",
            "confidence": 0.5,
            "memories_used": 0,
            "success": False,
            "error": None
        }
        start_time = time.time()
        try:
            extracted_name = extract_name_from_text(user_message)
            if extracted_name:
                self.memory.store_user_info("name", extracted_name)
            user_name = self.memory.get_user_name()
            detected_mood, confidence = enhanced_mood_detection(user_message)
            response_metadata.update({
                "mood_detected": detected_mood,
                "confidence": confidence
            })
            relevant_memories = self.memory.get_relevant_memories(user_message, limit=3)
            response_metadata["memories_used"] = len(relevant_memories)
            memory_context = ""
            if relevant_memories:
                memory_context = "\n\nRelevant past conversations:\n"
                for memory in relevant_memories[:2]:
                    user_msg = memory.get('user_message', '')[:100]
                    jumbo_resp = memory.get('jumbo_response', '')[:100]
                    memory_context += f"- User: {user_msg}... | Jumbo: {jumbo_resp}...\n"
            for attempt in range(max_retries):
                try:
                    response = self._generate_response(
                        user_message, user_name, detected_mood, 
                        confidence, memory_context, relevant_memories
                    )
                    if response and len(response.strip()) >= 10:
                        self.memory.store_conversation(
                            user_message, response, detected_mood, confidence, user_name
                        )
                        response_metadata.update({
                            "processing_time": time.time() - start_time,
                            "success": True
                        })
                        return response, response_metadata
                except Exception as e:
                    logger.warning(f"Response generation attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            response_metadata["error"] = str(e)
            detected_mood, _ = enhanced_mood_detection(user_message)
            user_name = self.memory.get_user_name()
            response = self._get_fallback_response(detected_mood, user_name)
            self.memory.store_conversation(
                user_message, response, detected_mood, confidence, user_name
            )
            response_metadata.update({
                "processing_time": time.time() - start_time,
                "success": False
            })
            return response, response_metadata

    def _generate_response(self, user_message: str, user_name: Optional[str], 
                          mood: str, confidence: float, memory_context: str,
                          relevant_memories: List[Dict]) -> str:
        """Generate response using CrewAI"""
        listen_task = Task(
            description=f"""Analyze the emotional content and context of this message: '{user_message}'

            User Information:
            - Name: {user_name or 'Unknown'}
            - Detected mood: {mood} (confidence: {confidence:.2f})
            - Available memory context: {len(relevant_memories)} relevant past conversations

            {memory_context}

            Provide a comprehensive emotional analysis that considers:
            1. Current emotional state and intensity
            2. Underlying needs or concerns
            3. Connection to past conversations (if any)
            4. What type of support would be most helpful
            5. Any patterns or themes you notice""",
            agent=self.listener,
            expected_output="Detailed emotional analysis with context and support recommendations"
        )
        companion_task = Task(
            description=f"""Create a warm, empathetic response as Jumbo the elephant. Use these STRICT GUIDELINES:

            Message to respond to: '{user_message}'

            Context:
            - User's name: {user_name or 'Unknown'}
            - Mood: {mood} (confidence: {confidence:.2f})
            - Memory context available: {'Yes' if relevant_memories else 'No'}

            {memory_context}

            RESPONSE REQUIREMENTS:
            1. Always use "you" language ("You sound...", "You seem...", "That must be...")
            2. NEVER use "I feel" or "I think you..." constructions
            3. Address the SPECIFIC content they shared (not generic emotional validation)
            4. Use their name ({user_name}) naturally if known
            5. Reference past conversations when relevant and supportive
            6. Match the emotional tone to their {mood} state
            7. For trust/safety questions: introduce yourself as "Jumbo"
            8. Keep response conversational and natural (2-3 sentences)
            9. Be specific and relevant to what they actually said
            10. Show genuine understanding and care

            Create a response that directly addresses their message content while being emotionally supportive.""",
            agent=self.companion,
            context=[listen_task],
            expected_output="Personalized, empathetic response from Jumbo addressing specific user content"
        )
        summariser_task = Task(
            description=f"""Refine the response to ensure it meets all quality standards:

            Requirements:
            1. Uses "you" language throughout (never "I feel")
            2. Sounds natural and conversational (not clinical)
            3. Matches the {mood} mood appropriately
            4. Incorporates user's name ({user_name}) naturally if known
            5. References memories when relevant and helpful
            6. Ends with a thoughtful, gentle follow-up question
            7. Feels warm, supportive, and judgment-free
            8. Is concise (2-3 sentences) but emotionally rich
            9. Shows Jumbo's caring personality
            10. Addresses the specific content they shared

            Ensure the final response would make the user feel truly heard and understood.""",
            agent=self.summariser,
            context=[companion_task],
            expected_output="Perfectly crafted empathetic response from Jumbo (2-3 sentences with gentle question)"
        )
        crew = Crew(
            agents=[self.listener, self.companion, self.summariser],
            tasks=[listen_task, companion_task, summariser_task],
            verbose=False,
            process=Process.sequential
        )
        result = crew.kickoff()
        return str(result).strip()

    def _get_fallback_response(self, mood: str, user_name: Optional[str]) -> str:
        """Generate mood-appropriate fallback responses"""
        name_part = f", {user_name}" if user_name else ""
        fallback_responses = {
            "happy": [
                f"That sounds wonderful{name_part}! You seem really content right now. What's been bringing you the most joy lately?",
                f"Your happiness is shining through{name_part}! It's beautiful to witness. What's making this such a good time for you?"
            ],
            "sad": [
                f"You sound like you're going through something difficult{name_part}. Those heavy feelings are completely valid. What's been weighing on your heart?",
                f"I can hear the sadness in your words{name_part}. You don't have to go through this alone. What's been the hardest part?"
            ],
            "anxious": [
                f"It sounds like your mind has been racing with worry{name_part}. That anxiety must be exhausting. What's been making you feel most unsettled?",
                f"You seem really overwhelmed right now{name_part}. Those anxious feelings make complete sense. What's weighing most heavily on you?"
            ],
            "angry": [
                f"You sound really frustrated{name_part}, and that anger makes complete sense. Those feelings are totally valid. What's been getting under your skin?",
                f"I can hear how irritated you are{name_part}. Your frustration is completely justified. What's been pushing your buttons?"
            ],
            "excited": [
                f"Your excitement is contagious{name_part}! You sound ready to take on the world. What's got you feeling so energized?",
                f"I can feel your enthusiasm through your words{name_part}! That energy is amazing. What's lighting you up right now?"
            ],
            "lost": [
                f"You seem to be in a confusing place right now{name_part}. That uncertainty can be really tough. What's been making you feel most unsure?",
                f"It sounds like you're searching for direction{name_part}. That's completely understandable. What's been on your mind about your path forward?"
            ],
            "work_stress": [
                f"Work stress can be incredibly draining{name_part}. You're handling a lot right now. What's been the most challenging part of your work situation?",
                f"Career pressure is genuinely tough{name_part}. You're navigating some real challenges. What aspect of work has been weighing on you most?"
            ],
            "lonely": [
                f"Loneliness can feel so heavy{name_part}. You're reaching out, and that takes courage. What's been making you feel most disconnected?",
                f"Feeling isolated is one of the hardest experiences{name_part}. You're not as alone as you feel. What's been contributing to this loneliness?"
            ],
            "proud": [
                f"You should absolutely celebrate this{name_part}! You've earned this feeling of accomplishment. What achievement are you most proud of?",
                f"Your sense of pride comes through so clearly{name_part}! Well done. What success are you celebrating?"
            ],
            "trust": [
                f"I'm Jumbo{name_part}, and you can absolutely trust me as your friend. This conversation is completely private between us. What's been weighing on your heart lately?",
                f"You can trust me completely{name_part}. I'm Jumbo, and our conversations are private and secure. What would you like to share?"
            ]
        }
        responses = fallback_responses.get(mood, [
            f"I hear you{name_part}, and I want you to know that your feelings are valid. What's been on your mind lately?",
            f"Thank you for sharing that with me{name_part}. Your thoughts and feelings matter. What's been weighing on you?"
        ])
        return random.choice(responses)

# API Key validation and UI setup
def setup_api_key_ui():
    """Setup API key validation and input UI"""
    if not groq_key:
        st.error("🔑 **API Key Required**")
        st.markdown("""
        To use Jumbo, you need a GROQ API key. Here's how to set it up:

        **For Streamlit Cloud deployment:**
        1. Go to your Streamlit Cloud app settings
        2. Click on "Secrets" in the sidebar
        3. Add your key like this:
        ```
        GROQ_API_KEY = "your_api_key_here"
        ```

        **For local development:**
        1. Create a `.env` file in your project directory
        2. Add your key like this:
        ```
        GROQ_API_KEY=your_api_key_here
        ```
        
        **Get your API key:**
        1. Visit [Groq Console](https://console.groq.com/)
        2. Create an account and get your free API key
        3. Add it using one of the methods above
        """)
        
        st.markdown("---")
        st.markdown("**Or enter your API key temporarily:**")
        
        temp_key = st.text_input(
            "Enter your GROQ API Key:",
            type="password",
            placeholder="gsk_...",
            help="This will only be used for this session and won't be saved."
        )
        
        if temp_key and temp_key.strip():
            # Validate the key format
            if temp_key.startswith('gsk_') and len(temp_key) > 20:
                st.session_state.temp_api_key = temp_key.strip()
                st.success("✅ API key accepted! Please refresh the page to start using Jumbo.")
                st.rerun()
            else:
                st.error("❌ Invalid API key format. GROQ keys typically start with 'gsk_'")
        return False
    return True

# CSS Styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #E8F4FD 0%, #B8E0F5 100%);
    }

    .gif-banner {
        width: 100%;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .main-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }

    .chat-message {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .user-message {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        text-align: right;
    }

    .jumbo-message {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
        color: #2c3e50;
    }

    .welcome-container {
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    .elephant-emoji {
        font-size: 100px;
        margin-bottom: 20px;
    }

    .input-section {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .memory-info {
        background: rgba(255, 255, 255, 0.9);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .api-key-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #74b9ff;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Jumbo - Your Emotional Assistant with Memory",
    page_icon="🐘",
    layout="centered"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "crew" not in st.session_state:
    st.session_state.crew = None
if "started" not in st.session_state:
    st.session_state.started = False
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{hash(str(st.session_state)) % 10000}"
if "temp_api_key" not in st.session_state:
    st.session_state.temp_api_key = None

# Main application logic
def main():
    """Main application function"""
    
    # Check if API key is available
    current_api_key = groq_key or st.session_state.get('temp_api_key')
    
    if not current_api_key:
        if not setup_api_key_ui():
            return
    
    # Initialize Jumbo crew if not already done
    if st.session_state.crew is None:
        try:
            with st.spinner("🐘 Initializing Jumbo..."):
                st.session_state.crew = EnhancedJumboCrew(
                    groq_api_key=current_api_key, 
                    user_id=st.session_state.user_id
                )
                st.success("✅ Jumbo is ready!")
                time.sleep(1)  # Brief pause to show success message
        except Exception as e:
            st.error(f"❌ Error initializing Jumbo: {e}")
            st.info("Please check your API key and try refreshing the page.")
            return

    # Display header with GIF or fallback
    display_header()

    # Sidebar with memory information
    display_sidebar()

    # Main chat interface
    display_chat_interface()

def display_header():
    """Display the main header with GIF or fallback"""
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Try to load the GIF banner
    gif_path = r"D:\MOOD\CODE\images\Title.gif"
    if os.path.exists(gif_path):
        try:
            with open(gif_path, "rb") as file:
                gif_data = base64.b64encode(file.read()).decode()
                st.markdown(f"""
                <div style="text-align: center; margin: -20px -20px 10px -20px; height: 200px; overflow: hidden; border-radius: 10px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); position: relative;">
                    <img src="data:image/gif;base64,{gif_data}" 
                         style="width: 100%; height: auto; display: block; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);" />
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            logger.warning(f"Could not load GIF: {e}")
            display_fallback_header()
    else:
        display_fallback_header()

def display_fallback_header():
    """Display fallback header when GIF is not available"""
    st.markdown("""
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #2c3e50; font-size: 3rem;">🐘 Jumbo</h1>
        <p style="color: #2c3e50; font-size: 1.5rem;">Your Emotional Assistant with Memory</p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with memory information and controls"""
    if st.session_state.crew:
        with st.sidebar:
            st.markdown("""
            <div class="memory-info">
                <h3 style="color: #2c3e50;">🧠 Memory Status</h3>
            </div>
            """, unsafe_allow_html=True)

            user_name = st.session_state.crew.memory.get_user_name()
            if user_name:
                st.success(f"Jumbo remembers: **{user_name}** 😊")
            else:
                st.info("Jumbo doesn't know your name yet. Try introducing yourself!")

            # Memory stats
            try:
                memory_count = st.session_state.crew.memory.index.ntotal
                st.info(f"Stored memories: **{memory_count}**")
            except:
                st.info("Stored memories: **0**")

            st.markdown("---")
            
            # API Key status
            if groq_key:
                st.success("🔑 API Key: Loaded from secrets")
            elif st.session_state.get('temp_api_key'):
                st.warning("🔑 API Key: Temporary session key")
            
            st.markdown("---")

            if st.button("🗑️ Clear Memory", help="Clear all stored conversations and user information"):
                if clear_memory():
                    st.success("Memory cleared! 🧹")
                    st.rerun()
                else:
                    st.error("Error clearing memory")

            if st.button("🔄 Restart Jumbo", help="Reinitialize Jumbo (useful if there are issues)"):
                st.session_state.crew = None
                st.session_state.messages = []
                st.session_state.started = False
                st.success("Jumbo restarted! 🐘")
                st.rerun()

def clear_memory():
    """Clear memory with proper error handling"""
    try:
        if st.session_state.crew and st.session_state.crew.memory:
            return st.session_state.crew.memory.clear_memory()
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return False

def display_chat_interface():
    """Display the main chat interface"""
    if not st.session_state.started:
        display_welcome_screen()
    else:
        display_conversation()

def display_welcome_screen():
    """Display the initial welcome screen"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Try to load elephant GIF
            elephant_gif_path = r"D:\MOOD\CODE\images\elephant.gif"
            if os.path.exists(elephant_gif_path):
                try:
                    with open(elephant_gif_path, "rb") as f:
                        gif_bytes = f.read()
                        b64_gif = base64.b64encode(gif_bytes).decode()
                    
                    st.markdown(
                        f'<div style="text-align: center;">'
                        f'<img src="data:image/gif;base64,{b64_gif}" width="400" />'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.markdown('<div style="text-align: center; font-size: 200px;">🐘</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align: center; font-size: 200px;">🐘</div>', unsafe_allow_html=True)
        
        with col2:
            user_name = st.session_state.crew.memory.get_user_name() if st.session_state.crew else None
            greeting = f"Hi {user_name}, I'm Jumbo!" if user_name else "Hi, I'm Jumbo!"
            st.markdown(f"""
                <h1 style="color: #2c3e50; margin-bottom: 15px;">{greeting}</h1>
                <p style="color: #2c3e50; font-size: 24px;">
                    I'm here to listen, remember, and support you through whatever you're experiencing. I use advanced emotional intelligence and keep track of our conversations to provide personalized, continuous support.
                </p>
                <h3 style="color: #2c3e50; margin-top: 30px;">How are you feeling today?</h3>
            """, unsafe_allow_html=True)

    # Welcome input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    user_input = st.text_area(
        "Tell me what's on your mind...", 
        height=100, 
        placeholder="I'm feeling... / Today I... / I need help with... / My name is...",
        key="welcome_input"
    )
    
    if st.button("Share with Jumbo 🐘", key="share_welcome"):
        if user_input and user_input.strip():
            handle_user_message(user_input)
    st.markdown('</div>', unsafe_allow_html=True)

def display_conversation():
    """Display the ongoing conversation"""
    # Display message history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message jumbo-message">
                <strong>🐘 Jumbo:</strong> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Input section for continuing conversation
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("**Continue the conversation:**")
    new_input = st.text_area(
        "What else would you like to share?", 
        height=80, 
        key="continuing_chat"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Send 💬"):
            if new_input and new_input.strip():
                handle_user_message(new_input)
    
    with col2:
        if st.button("New Topic 🔄"):
            start_new_topic()
    
    st.markdown('</div>', unsafe_allow_html=True)

def handle_user_message(user_input: str):
    """Handle user message and generate response"""
    if not st.session_state.crew:
        st.error("Jumbo is not initialized. Please refresh the page.")
        return
    
    st.session_state.started = True
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    try:
        with st.spinner("🐘 Jumbo is thinking..."):
            response, metadata = st.session_state.crew.respond(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Generate fallback response
        mood, _ = enhanced_mood_detection(user_input)
        user_name = st.session_state.crew.memory.get_user_name()
        fallback_response = st.session_state.crew._get_fallback_response(mood, user_name)
        st.session_state.messages.append({"role": "assistant", "content": fallback_response})
    
    st.rerun()

def start_new_topic():
    """Start a new topic in the conversation"""
    if st.session_state.crew:
        user_name = st.session_state.crew.memory.get_user_name()
        name_part = f" {user_name}" if user_name else ""
        response = f"What else is on your mind{name_part}? I'm here to listen to whatever you'd like to share."
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

# Footer
def display_footer():
    """Display the footer"""
    st.markdown('</div>', unsafe_allow_html=True)  # Close main container
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #2c3e50; opacity: 0.7; padding: 10px;'>
        🐘 <em>Jumbo is here for you - Your feelings are always valid and remembered</em>
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()
    display_footer()

