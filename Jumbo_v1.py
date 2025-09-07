import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass
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
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew, Process
import faiss
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from textblob import TextBlob
import re

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
    load_dotenv()
    api_key = None
    
    # 1. Try Streamlit Secrets first (for deployed apps)
    try:
        if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
            logger.info("✅ API key loaded from Streamlit Secrets")
    except Exception:
        # Suppress the warning for local development - this is expected
        pass
    
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
    """Enhanced mood detection with better handling of introductions and neutral statements"""
    try:
        text_lower = text.lower().strip()
        
        # Check for trust/safety keywords first
        trust_keywords = ["trust", "safe", "safety", "secure", "private", "confidential", "remember", "friend"]
        if any(keyword in text_lower for keyword in trust_keywords):
            return "trust", 0.9
        
        # Check for name introductions - these should be treated as neutral/friendly
        name_introduction_patterns = [
            r"my name is",
            r"i'm \w+",
            r"i am \w+", 
            r"call me",
            r"name's",
            r"i go by",
            r"hi.*my name",
            r"hello.*my name",
            r"hi i'm",
            r"hello i'm"
        ]
        
        if any(re.search(pattern, text_lower) for pattern in name_introduction_patterns):
            return "friendly_introduction", 0.8
        
        # Check for simple greetings - should be neutral/friendly
        greeting_patterns = [
            r"^hi$", r"^hello$", r"^hey$", r"^good morning$", 
            r"^good afternoon$", r"^good evening$", r"how are you",
            r"what's up", r"how's it going"
        ]
        
        if any(re.search(pattern, text_lower) for pattern in greeting_patterns):
            return "greeting", 0.7
        
        # Only proceed with emotional analysis if the text suggests emotional content
        emotional_indicators = [
            "feel", "feeling", "emotional", "mood", "upset", "happy", "sad", 
            "angry", "frustrated", "excited", "worried", "anxious", "stressed",
            "depressed", "overwhelmed", "scared", "nervous", "hurt", "pain"
        ]
        
        has_emotional_content = any(word in text_lower for word in emotional_indicators)
        
        # If no emotional indicators and short message, likely neutral
        if not has_emotional_content and len(text.split()) <= 10:
            return "neutral", 0.6
        
        # Proceed with keyword-based mood detection for emotional content
        mood_scores = {}
        for mood, keywords in MOOD_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                mood_scores[mood] = score / len(keywords)
        
        # Use sentiment analysis only if we have emotional content
        if has_emotional_content:
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
                
                # Compare keyword and sentiment analysis
                if mood_scores:
                    top_keyword_mood = max(mood_scores.items(), key=lambda x: x[1])
                    if top_keyword_mood[1] > sentiment_confidence:
                        return top_keyword_mood[0], top_keyword_mood[1]
                    else:
                        return sentiment_mood, sentiment_confidence
                else:
                    return sentiment_mood, sentiment_confidence
                    
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
        
        # Return keyword-based mood if available
        if mood_scores:
            top_mood = max(mood_scores.items(), key=lambda x: x[1])
            return top_mood[0], top_mood[1]
        
        # Default to neutral for everything else
        return "neutral", 0.5
        
    except Exception as e:
        logger.error(f"Mood detection failed: {e}")
        return "neutral", 0.5

# Add new response categories for introductions and greetings
JUMBO_RESPONSES.update({
    "friendly_introduction": [
        "Nice to meet you! I'm Jumbo, and I'm really glad you're here.",
        "It's wonderful to meet you! I'm Jumbo, your friendly elephant companion.",
        "Hello there! I'm Jumbo, and I'm excited to get to know you.",
        "Great to meet you! I'm Jumbo, and I'm here whenever you need someone to talk to."
    ],
    "greeting": [
        "Hello! I'm Jumbo, and I'm happy you're here. How are you doing today?",
        "Hi there! I'm Jumbo, your friendly elephant companion. What's on your mind?",
        "Hey! I'm Jumbo, and I'm glad to see you. How are things going?",
        "Hello! I'm Jumbo, and I'm here to listen. What would you like to talk about?"
    ]
})

def _generate_response(self, user_message: str, user_name: Optional[str], 
                      mood: str, confidence: float, memory_context: str,
                      relevant_memories: List[Dict]) -> str:
    """Generate response with better handling of different interaction types"""
    
    # Handle name introductions specially
    if mood == "friendly_introduction":
        if user_name:
            return f"Nice to meet you, {user_name}! I'm Jumbo, your friendly elephant companion. I'll remember your name and our conversations. How are you feeling today?"
        else:
            return "Hello! I'm Jumbo, your friendly elephant companion. I'm here to listen and support you through whatever you're experiencing. How are you doing today?"
    
    # Handle simple greetings
    if mood == "greeting":
        name_part = f", {user_name}" if user_name else ""
        return f"Hi there{name_part}! I'm Jumbo, and I'm glad you're here. What's on your mind today?"
    
    # For neutral conversations, use a gentler approach
    if mood == "neutral" and confidence < 0.7:
        listen_task = Task(
            description=f"""Analyze this message: '{user_message}'

            User Information:
            - Name: {user_name or 'Unknown'}
            - Message appears neutral/conversational
            - Available memory: {len(relevant_memories)} past conversations

            {memory_context}

            Determine:
            1. Is this a simple statement, question, or does it contain emotional content?
            2. What kind of response would be most appropriate?
            3. Should this be treated as casual conversation or emotional support?
            
            Provide a brief analysis focused on the appropriate response type.""",
            agent=self.listener,
            expected_output="Analysis of message type and appropriate response approach"
        )
        
        companion_task = Task(
            description=f"""Create a warm, appropriate response as Jumbo.

            Message: '{user_message}'
            User: {user_name or 'Unknown'}
            Context: This appears to be a neutral/conversational message

            {memory_context}

            RESPONSE GUIDELINES:
            1. Be warm and friendly, not overly therapeutic
            2. Use their name ({user_name}) if known
            3. Respond to what they actually said
            4. Keep it conversational and natural
            5. Ask a gentle follow-up question
            6. Don't assume they're in crisis
            7. Match their energy level
            8. Reference past conversations if relevant

            Create a friendly, conversational response that acknowledges what they shared.""",
            agent=self.companion,
            context=[listen_task],
            expected_output="Friendly, conversational response from Jumbo"
        )
    else:
        # Use the original emotional support approach for clearly emotional content
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
            
            Focus on providing actionable insights for response generation.""",
            agent=self.listener,
            expected_output="Detailed emotional analysis with support recommendations"
        )
        
        companion_task = Task(
            description=f"""Create a warm, empathetic response as Jumbo.

            Message: '{user_message}'
            User: {user_name or 'Unknown'}
            Mood: {mood} (confidence: {confidence:.2f})

            {memory_context}

            RESPONSE REQUIREMENTS:
            1. Use "you" language ("You sound...", "You seem...")
            2. Address the SPECIFIC content they shared
            3. Use their name ({user_name}) naturally if known
            4. Reference past conversations when relevant
            5. Match the emotional tone appropriately
            6. Be conversational, not clinical
            7. End with a thoughtful follow-up question
            8. Show genuine understanding

            Create a response that directly addresses their message content.""",
            agent=self.companion,
            context=[listen_task],
            expected_output="Empathetic response from Jumbo addressing specific user content"
        )
    
    summariser_task = Task(
        description=f"""Ensure the response meets quality standards:

        Requirements:
        1. Natural and conversational tone
        2. Appropriate to the message type (introduction/greeting/emotional/neutral)
        3. Uses their name ({user_name}) naturally if known
        4. Ends with an appropriate follow-up question
        5. Shows Jumbo's warm personality
        6. Is concise but meaningful
        7. Avoids over-dramatization

        The response should make the user feel welcomed and understood.""",
        agent=self.summariser,
        context=[companion_task],
        expected_output="Polished, appropriate response from Jumbo"
    )
    
    # Create and execute crew
    crew = Crew(
        agents=[self.listener, self.companion, self.summariser],
        tasks=[listen_task, companion_task, summariser_task],
        verbose=False,
        process=Process.sequential
    )
    
    result = crew.kickoff()
    return str(result).strip()

def extract_name_from_text(text: str) -> Optional[str]:
    """IMPROVED: Enhanced name extraction with better patterns and validation"""
    text_clean = text.lower().strip()
    
    # Expanded patterns for name detection
    patterns = [
        (r"my name is ([a-zA-Z]{2,20})", 1),
        (r"i'm ([a-zA-Z]{2,20})", 1),
        (r"i am ([a-zA-Z]{2,20})", 1),
        (r"call me ([a-zA-Z]{2,20})", 1),
        (r"name's ([a-zA-Z]{2,20})", 1),
        (r"i go by ([a-zA-Z]{2,20})", 1),
        (r"everyone calls me ([a-zA-Z]{2,20})", 1),
        (r"you can call me ([a-zA-Z]{2,20})", 1),
        (r"they call me ([a-zA-Z]{2,20})", 1),
        (r"i'm known as ([a-zA-Z]{2,20})", 1),
        (r"it's ([a-zA-Z]{2,20}) here", 1),
        (r"this is ([a-zA-Z]{2,20})", 1)
    ]
    
    # Common words to exclude (not names)
    excluded_words = {
        'feeling', 'good', 'bad', 'okay', 'fine', 'great', 'happy', 'sad', 
        'angry', 'tired', 'stressed', 'worried', 'excited', 'nervous',
        'confused', 'lost', 'ready', 'here', 'there', 'going', 'coming',
        'working', 'studying', 'thinking', 'wondering', 'hoping', 'trying'
    }
    
    for pattern, group in patterns:
        match = re.search(pattern, text_clean)
        if match:
            name = match.group(group).strip()
            if (len(name) >= 2 and name.isalpha() and 
                name.lower() not in excluded_words):
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
    words = text.lower().split()
    embedding = np.zeros(dim)
    
    for i, word in enumerate(words):
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
    """IMPROVED: Enhanced memory system with better error handling and persistence"""
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
        self._initialize_memory()

    def _initialize_memory(self):
        """IMPROVED: Better initialization with error recovery"""
        try:
            if os.path.exists(self.index_path) and os.path.getsize(self.index_path) > 0:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Created new FAISS index")
            
            # Load metadata with validation
            if os.path.exists(self.metadata_path) and os.path.getsize(self.metadata_path) > 0:
                try:
                    with open(self.metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)
                    # Validate metadata length matches index
                    if len(self.metadata) != self.index.ntotal:
                        logger.warning("Metadata length mismatch with index, rebuilding...")
                        self._rebuild_memory()
                except (pickle.PickleError, EOFError):
                    logger.warning("Corrupted metadata file, resetting...")
                    self.metadata = []
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                self.metadata = []
                
            # Load user info
            if os.path.exists(self.user_info_path) and os.path.getsize(self.user_info_path) > 0:
                try:
                    with open(self.user_info_path, 'r') as f:
                        self.user_info = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("Corrupted user info file, resetting...")
                    self.user_info = {}
            else:
                self.user_info = {}
                
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            # Create fresh instances
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            self.user_info = {}

    def _rebuild_memory(self):
        """Rebuild memory when there are inconsistencies"""
        try:
            # Keep only metadata entries that have valid structure
            valid_metadata = []
            for meta in self.metadata:
                if isinstance(meta, dict) and 'user_message' in meta and 'jumbo_response' in meta:
                    valid_metadata.append(meta)
            
            # Recreate index with valid metadata
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            
            for meta in valid_metadata:
                conversation_text = f"User: {meta['user_message']}\nJumbo: {meta['jumbo_response']}"
                embedding = simple_text_embedding(conversation_text, self.embedding_dim)
                self.index.add(embedding.reshape(1, -1))
                self.metadata.append(meta)
                
            logger.info(f"Rebuilt memory with {len(self.metadata)} valid entries")
            self._save_to_disk()
            
        except Exception as e:
            logger.error(f"Failed to rebuild memory: {e}")
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []

    def _save_to_disk(self):
        """IMPROVED: Save with better error handling and atomic writes"""
        try:
            # Use temporary files for atomic writes
            temp_index_path = self.index_path + ".tmp"
            temp_metadata_path = self.metadata_path + ".tmp"
            temp_user_info_path = self.user_info_path + ".tmp"
            
            # Save to temporary files first
            faiss.write_index(self.index, temp_index_path)
            with open(temp_metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            with open(temp_user_info_path, 'w') as f:
                json.dump(self.user_info, f, indent=2)
            
            # Move temporary files to final locations (atomic)
            if os.path.exists(temp_index_path):
                os.replace(temp_index_path, self.index_path)
            if os.path.exists(temp_metadata_path):
                os.replace(temp_metadata_path, self.metadata_path)
            if os.path.exists(temp_user_info_path):
                os.replace(temp_user_info_path, self.user_info_path)
                
        except Exception as e:
            logger.error(f"Failed to save memory to disk: {e}")
            # Clean up temporary files
            for temp_path in [temp_index_path, temp_metadata_path, temp_user_info_path]:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass

    def store_conversation(self, user_message: str, jumbo_response: str, mood: str, 
                          confidence: float, user_name: str = None) -> bool:
        """IMPROVED: Store conversation with better validation"""
        try:
            # Validate inputs
            if not user_message or not jumbo_response:
                logger.warning("Empty message or response, skipping storage")
                return False
                
            conversation_text = f"User: {user_message}\nJumbo: {jumbo_response}"
            
            # Create embedding
            embedding = simple_text_embedding(conversation_text, self.embedding_dim)
            
            # Create metadata with validation
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "mood": mood or "neutral",
                "confidence": max(0.0, min(1.0, confidence)),  # Clamp between 0-1
                "type": "conversation",
                "message_length": len(user_message),
                "response_length": len(jumbo_response),
                "user_message": user_message[:1000],  # Limit length to prevent bloat
                "jumbo_response": jumbo_response[:1000]
            }
            if user_name and isinstance(user_name, str):
                metadata["user_name"] = user_name[:50]  # Limit name length
            
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

    def store_user_info(self, info_type: str, info_value: str):
        """IMPROVED: Store user information with validation"""
        try:
            if not info_type or not info_value:
                return
                
            # Validate and sanitize inputs
            info_type = str(info_type).strip()[:50]
            info_value = str(info_value).strip()[:100]
            
            if not info_type or not info_value:
                return
                
            self.user_info[info_type] = {
                "value": info_value,
                "timestamp": datetime.now().isoformat()
            }
            self._save_to_disk()
            logger.info(f"Stored user info: {info_type} = {info_value}")
            
        except Exception as e:
            logger.error(f"Failed to store user info: {e}")

    def get_user_name(self) -> Optional[str]:
        """IMPROVED: Retrieve user's name with validation"""
        try:
            name_info = self.user_info.get("name")
            if name_info and isinstance(name_info, dict):
                name = name_info.get("value")
                if name and isinstance(name, str) and len(name.strip()) > 0:
                    return name.strip()
        except Exception as e:
            logger.error(f"Failed to get user name: {e}")
        return None

    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """IMPROVED: Retrieve relevant memories with better error handling"""
        try:
            if not query or not query.strip():
                return []
                
            if self.index.ntotal == 0:
                return []
            
            # Create query embedding
            query_embedding = simple_text_embedding(query.strip(), self.embedding_dim)
            
            # Search for similar conversations
            k = min(limit, self.index.ntotal)
            distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            memories = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if 0 <= idx < len(self.metadata):
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
        """IMPROVED: Clear memory with better cleanup"""
        try:
            # Reset index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.metadata = []
            self.user_info = {}
            
            # Remove files safely
            for path in [self.index_path, self.metadata_path, self.user_info_path]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"Could not remove {path}: {e}")
            
            logger.info("Memory cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False

    def _cleanup_old_memories(self):
        """IMPROVED: Clean up old memories with better logic"""
        try:
            # Only cleanup if we have too many memories
            if self.index.ntotal < 100:  # Don't cleanup until we have at least 100 memories
                return
                
            cutoff_date = datetime.now() - timedelta(days=Config.MEMORY_CLEANUP_DAYS)
            cutoff_str = cutoff_date.isoformat()
            
            # Find indices to keep
            indices_to_keep = []
            new_metadata = []
            
            for i, meta in enumerate(self.metadata):
                timestamp = meta.get("timestamp", "")
                if timestamp >= cutoff_str:
                    indices_to_keep.append(i)
                    new_metadata.append(meta)
            
            # Only rebuild if we're removing a significant number
            removed_count = len(self.metadata) - len(indices_to_keep)
            if removed_count > 10:  # Only cleanup if removing more than 10 items
                logger.info(f"Cleaning up {removed_count} old memories")
                
                # Rebuild index with only recent memories
                if indices_to_keep:
                    vectors = []
                    for i in indices_to_keep:
                        try:
                            vector = self.index.reconstruct(i)
                            vectors.append(vector)
                        except Exception as e:
                            logger.warning(f"Could not reconstruct vector {i}: {e}")
                    
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

class EnhancedJumboCrew:
    """IMPROVED: Enhanced emotional wellness chatbot with better performance"""
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
        """IMPROVED: Generate response with enhanced error handling and retries"""
        response_metadata = {
            "timestamp": datetime.now().isoformat(),
            "processing_time": 0,
            "mood_detected": "neutral",
            "confidence": 0.5,
            "memories_used": 0,
            "success": False,
            "error": None,
            "name_detected": False
        }
        start_time = time.time()
        
        try:
            # Extract and store name with validation
            extracted_name = extract_name_from_text(user_message)
            if extracted_name:
                current_name = self.memory.get_user_name()
                if not current_name or current_name != extracted_name:
                    self.memory.store_user_info("name", extracted_name)
                    response_metadata["name_detected"] = True
                    logger.info(f"Name detected and stored: {extracted_name}")
            
            user_name = self.memory.get_user_name()
            detected_mood, confidence = enhanced_mood_detection(user_message)
            response_metadata.update({
                "mood_detected": detected_mood,
                "confidence": confidence
            })
            
            # Get relevant memories with better filtering
            relevant_memories = self.memory.get_relevant_memories(user_message, limit=3)
            response_metadata["memories_used"] = len(relevant_memories)
            
            # Build memory context more efficiently
            memory_context = ""
            if relevant_memories:
                memory_context = "\n\nRelevant past conversations:\n"
                for memory in relevant_memories[:2]:  # Only use top 2 most relevant
                    user_msg = memory.get('user_message', '')[:100]
                    jumbo_resp = memory.get('jumbo_response', '')[:100]
                    timestamp = memory.get('timestamp', '')[:10]  # Just date part
                    memory_context += f"- [{timestamp}] User: {user_msg}... | Jumbo: {jumbo_resp}...\n"
            
            # Generate response with retries
            for attempt in range(max_retries):
                try:
                    response = self._generate_response(
                        user_message, user_name, detected_mood, 
                        confidence, memory_context, relevant_memories
                    )
                    
                    # Validate response quality
                    if response and len(response.strip()) >= 10:
                        # Store conversation in memory
                        storage_success = self.memory.store_conversation(
                            user_message, response, detected_mood, confidence, user_name
                        )
                        
                        response_metadata.update({
                            "processing_time": time.time() - start_time,
                            "success": True,
                            "storage_success": storage_success
                        })
                        return response, response_metadata
                    else:
                        logger.warning(f"Generated response too short or empty: {response}")
                        
                except Exception as e:
                    logger.warning(f"Response generation attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.5)  # Brief pause between retries
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            response_metadata["error"] = str(e)
            
            # Generate fallback response
            detected_mood, confidence = enhanced_mood_detection(user_message)
            user_name = self.memory.get_user_name()
            response = self._get_fallback_response(detected_mood, user_name)
            
            # Try to store the fallback conversation
            try:
                self.memory.store_conversation(
                    user_message, response, detected_mood, confidence, user_name
                )
            except Exception as storage_error:
                logger.error(f"Failed to store fallback conversation: {storage_error}")
            
            response_metadata.update({
                "processing_time": time.time() - start_time,
                "success": False
            })
            return response, response_metadata

    def _generate_response(self, user_message: str, user_name: Optional[str], 
                          mood: str, confidence: float, memory_context: str,
                          relevant_memories: List[Dict]) -> str:
        """IMPROVED: Generate response using CrewAI with better context"""
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
            5. Any patterns or themes you notice
            
            Keep analysis focused and relevant to provide actionable insights for response generation.""",
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
            11. End with a thoughtful follow-up question

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

            Ensure the final response would make the user feel truly heard and understood.
            The response should be complete and ready to send - no additional formatting needed.""",
            agent=self.summariser,
            context=[companion_task],
            expected_output="Perfectly crafted empathetic response from Jumbo (2-3 sentences with gentle question)"
        )
        
        # Create and execute crew
        crew = Crew(
            agents=[self.listener, self.companion, self.summariser],
            tasks=[listen_task, companion_task, summariser_task],
            verbose=False,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        return str(result).strip()

    def _get_fallback_response(self, mood: str, user_name: Optional[str]) -> str:
        """IMPROVED: Generate mood-appropriate fallback responses"""
        name_part = f", {user_name}" if user_name else ""
        
        fallback_responses = {
            "happy": [
                f"That sounds wonderful{name_part}! You seem really content right now. What's been bringing you the most joy lately?",
                f"Your happiness is shining through{name_part}! It's beautiful to witness. What's making this such a good time for you?",
                f"You sound genuinely happy{name_part}! That's lovely to hear. What's been lifting your spirits?"
            ],
            "sad": [
                f"You sound like you're going through something difficult{name_part}. Those heavy feelings are completely valid. What's been weighing on your heart?",
                f"I can hear the sadness in your words{name_part}. You don't have to go through this alone. What's been the hardest part?",
                f"It sounds like you're carrying some pain right now{name_part}. That must be really tough. What's been on your mind?"
            ],
            "anxious": [
                f"It sounds like your mind has been racing with worry{name_part}. That anxiety must be exhausting. What's been making you feel most unsettled?",
                f"You seem really overwhelmed right now{name_part}. Those anxious feelings make complete sense. What's weighing most heavily on you?",
                f"Your worries are completely understandable{name_part}. Anxiety can be so draining. What's been causing you the most stress?"
            ],
            "angry": [
                f"You sound really frustrated{name_part}, and that anger makes complete sense. Those feelings are totally valid. What's been getting under your skin?",
                f"I can hear how irritated you are{name_part}. Your frustration is completely justified. What's been pushing your buttons?",
                f"That frustration is really coming through{name_part}. You have every right to feel upset. What's been bothering you most?"
            ],
            "excited": [
                f"Your excitement is contagious{name_part}! You sound ready to take on the world. What's got you feeling so energized?",
                f"I can feel your enthusiasm through your words{name_part}! That energy is amazing. What's lighting you up right now?",
                f"You sound absolutely pumped{name_part}! That excitement is wonderful. What's got you so fired up?"
            ],
            "lost": [
                f"You seem to be in a confusing place right now{name_part}. That uncertainty can be really tough. What's been making you feel most unsure?",
                f"It sounds like you're searching for direction{name_part}. That's completely understandable. What's been on your mind about your path forward?",
                f"That feeling of being lost is so hard{name_part}. You're not alone in this uncertainty. What's been weighing on you?"
            ],
            "work_stress": [
                f"Work stress can be incredibly draining{name_part}. You're handling a lot right now. What's been the most challenging part of your work situation?",
                f"Career pressure is genuinely tough{name_part}. You're navigating some real challenges. What aspect of work has been weighing on you most?",
                f"Job stress is no joke{name_part}. You're dealing with a lot. What's been the biggest source of pressure at work?"
            ],
            "lonely": [
                f"Loneliness can feel so heavy{name_part}. You're reaching out, and that takes courage. What's been making you feel most disconnected?",
                f"Feeling isolated is one of the hardest experiences{name_part}. You're not as alone as you feel. What's been contributing to this loneliness?",
                f"That sense of loneliness is really painful{name_part}. You're taking a brave step by sharing. What's been making you feel most alone?"
            ],
            "proud": [
                f"You should absolutely celebrate this{name_part}! You've earned this feeling of accomplishment. What achievement are you most proud of?",
                f"Your sense of pride comes through so clearly{name_part}! Well done. What success are you celebrating?",
                f"That pride is so well-deserved{name_part}! You should feel good about this. What accomplishment has you feeling proud?"
            ],
            "trust": [
                f"I'm Jumbo{name_part}, and you can absolutely trust me as your friend. This conversation is completely private between us. What's been weighing on your heart lately?",
                f"You can trust me completely{name_part}. I'm Jumbo, and our conversations are private and secure. What would you like to share?",
                f"I'm here for you{name_part} - I'm Jumbo, and this space is totally safe. What's been on your mind?"
            ]
        }
        
        # Get responses for the mood, or use neutral fallback
        responses = fallback_responses.get(mood, [
            f"I hear you{name_part}, and I want you to know that your feelings are valid. What's been on your mind lately?",
            f"Thank you for sharing that with me{name_part}. Your thoughts and feelings matter. What's been weighing on you?",
            f"You've reached out{name_part}, and that matters. What would you like to talk about?"
        ])
        
        return random.choice(responses)

# IMPROVED: API Key validation and UI setup
def setup_api_key_ui():
    """Setup API key validation and input UI with better UX"""
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

# IMPROVED: CSS Styling
st.markdown("""
<style>
    .stApp {
        background: white;
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
        word-wrap: break-word;
    }

    .user-message {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        text-align: right;
        margin-left: 20%;
    }

    .jumbo-message {
        background: linear-gradient(135deg, #a8e6cf 0%, #7fcdcd 100%);
        color: #2c3e50;
        margin-right: 20%;
    }

    .welcome-container {
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    .input-section {
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        position: sticky;
        bottom: 0;
        z-index: 100;
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

    /* IMPROVED: Better scrolling and input handling */
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        transition: border-color 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: #74b9ff !important;
        box-shadow: 0 0 0 2px rgba(116, 185, 255, 0.2) !important;
    }

    .stButton button {
        border-radius: 10px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="Jumbo - Your Emotional Assistant with Memory",
    page_icon="🐘",
    layout="centered",
    initial_sidebar_state="expanded"
)

# IMPROVED: Initialize session state with better defaults
def initialize_session_state():
    """Initialize session state variables with validation"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "crew" not in st.session_state:
        st.session_state.crew = None
    if "started" not in st.session_state:
        st.session_state.started = False
    if "user_id" not in st.session_state:
        # Create more unique user ID
        unique_id = f"user_{hash(str(st.session_state) + str(time.time())) % 100000}"
        st.session_state.user_id = unique_id
    if "temp_api_key" not in st.session_state:
        st.session_state.temp_api_key = None
    if "last_input_key" not in st.session_state:
        st.session_state.last_input_key = 0
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0

# Main application logic
def main():
    """IMPROVED: Main application function with better error handling"""
    initialize_session_state()
    
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
                st.success("✅ Jumbo is ready to chat!")
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
    gif_path = "images/Title.gif"
    if os.path.exists(gif_path):
        try:
            with open(gif_path, "rb") as file:
                gif_data = base64.b64encode(file.read()).decode()
                st.markdown(f"""
                <div style="text-align: center; margin: -20px -20px 10px -20px; height: 200px; overflow: hidden; border-radius: 0px; box-shadow: 0 0px 0px rgba(0, 0, 0, 0); position: relative;">
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
    """IMPROVED: Display sidebar with enhanced memory information and controls"""
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
                st.info("Jumbo doesn't know your name yet. Try saying 'My name is...' or 'I'm...'")

            # Memory stats with better display
            try:
                memory_count = st.session_state.crew.memory.index.ntotal
                if memory_count > 0:
                    st.info(f"💭 Stored memories: **{memory_count}**")
                    st.info(f"💬 This session: **{st.session_state.conversation_count}** exchanges")
                else:
                    st.info("💭 No memories stored yet")
            except Exception as e:
                st.warning("⚠️ Memory status unavailable")

            st.markdown("---")
            
            # API Key status
            if groq_key:
                st.success("🔑 API Key: Loaded from secrets")
            elif st.session_state.get('temp_api_key'):
                st.warning("🔑 API Key: Temporary session key")
            
            st.markdown("---")

            # Better button layout
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear\nMemory", help="Clear all stored conversations and user information"):
                    if clear_memory():
                        st.success("Memory cleared! 🧹")
                        st.rerun()
                    else:
                        st.error("Error clearing memory")

            with col2:
                if st.button("🔄 Restart\nJumbo", help="Reinitialize Jumbo (useful if there are issues)"):
                    restart_jumbo()

            # Additional help section
            with st.expander("ℹ️ How Jumbo's Memory Works"):
                st.markdown("""
                **Jumbo remembers:**
                - Your name when you introduce yourself
                - Past conversations and emotions
                - Context from previous sessions
                
                **Privacy:**
                - All data is stored locally on your device
                - No information is shared externally
                - You can clear memory anytime
                
                **Name Recognition:**
                - Say "My name is [Name]" or "I'm [Name]"
                - "Call me [Name]" also works
                - Jumbo will remember and use your name
                """)

def clear_memory():
    """IMPROVED: Clear memory with proper error handling and user feedback"""
    try:
        if st.session_state.crew and st.session_state.crew.memory:
            success = st.session_state.crew.memory.clear_memory()
            if success:
                st.session_state.conversation_count = 0
                return True
        return False
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        return False

def restart_jumbo():
    """IMPROVED: Restart Jumbo with better state management"""
    try:
        # Clear session state
        st.session_state.crew = None
        st.session_state.messages = []
        st.session_state.started = False
        st.session_state.conversation_count = 0
        st.session_state.last_input_key += 1  # Force input refresh
        
        st.success("Jumbo restarted! 🐘")
        time.sleep(0.5)
        st.rerun()
    except Exception as e:
        logger.error(f"Error restarting Jumbo: {e}")
        st.error("Error restarting Jumbo")

def display_chat_interface():
    """IMPROVED: Display the main chat interface with better UX"""
    if not st.session_state.started:
        display_welcome_screen()
    else:
        display_conversation()

def display_welcome_screen():
    """IMPROVED: Display the initial welcome screen"""
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Try to load elephant GIF
            elephant_gif_path = "images/elephant.gif"
            if os.path.exists(elephant_gif_path):
                try:
                    with open(elephant_gif_path, "rb") as f:
                        gif_bytes = f.read()
                        b64_gif = base64.b64encode(gif_bytes).decode()
                    
                    st.markdown(
                        f'<div style="text-align: center;">'
                        f'<img src="data:image/gif;base64,{b64_gif}" style="max-width: 100%; height: auto; border-radius: 10px;" />'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                except Exception:
                    st.markdown('<div style="text-align: center; font-size: 150px;">🐘</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div style="text-align: center; font-size: 150px;">🐘</div>', unsafe_allow_html=True)
        
        with col2:
            user_name = st.session_state.crew.memory.get_user_name() if st.session_state.crew else None
            greeting = f"Hi {user_name}, I'm Jumbo!" if user_name else "Hi there, I'm Jumbo!"
            
            st.markdown(f"""
                <h1 style="color: #2c3e50; margin-bottom: 15px;">{greeting}</h1>
                <p style="color: #2c3e50; font-size: 20px; line-height: 1.6;">
                    I'm here to listen, remember, and support you through whatever you're experiencing. 
                    I use emotional intelligence and keep track of our conversations to provide 
                    personalized, continuous support.
                </p>
                <h3 style="color: #2c3e50; margin-top: 30px;">How are you feeling today?</h3>
            """, unsafe_allow_html=True)

       
    user_input = st.text_area(
        "Tell me what's on your mind...", 
        height=120, 
        placeholder="I'm feeling... / Today I... / I need help with... / My name is... / Call me...",
        key=f"welcome_input_{st.session_state.last_input_key}",
        help="Share anything - your feelings, what happened today, or just introduce yourself!"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Share with Jumbo 🐘", key="share_welcome", use_container_width=True):
            if user_input and user_input.strip():
                handle_user_message(user_input)
            else:
                st.warning("Please enter something to share with Jumbo!")
                
    st.markdown('</div>', unsafe_allow_html=True)

def handle_user_message(user_input: str):
    """Handle user message and generate Jumbo's response"""
    if not st.session_state.crew:
        st.error("Jumbo is not initialized. Please refresh the page.")
        return
    
    try:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.started = True
        st.session_state.conversation_count += 1
        
        # Show processing indicator
        with st.spinner("🐘 Jumbo is thinking..."):
            # Generate response using the crew
            response, metadata = st.session_state.crew.respond(user_input)
        
        # Add Jumbo's response to session state
        st.session_state.messages.append({"role": "jumbo", "content": response})
        
        # Show success indicators
        if metadata.get("name_detected"):
            st.success("✅ Jumbo learned your name!")
        
        if metadata.get("memories_used", 0) > 0:
            st.info(f"🧠 Jumbo referenced {metadata['memories_used']} past conversations")
        
        # Increment input key to refresh input field
        st.session_state.last_input_key += 1
        
        # Rerun to update the display
        st.rerun()
        
    except Exception as e:
        logger.error(f"Error handling user message: {e}")
        st.error(f"Sorry, I encountered an error: {e}")
        
        # Add fallback response
        fallback_response = "I apologize, but I'm having some technical difficulties right now. Please try again, and if the problem persists, you might want to restart me using the sidebar."
        st.session_state.messages.append({"role": "jumbo", "content": fallback_response})

def display_conversation():
    """Display the ongoing conversation with better layout"""
    # Display message history with improved styling
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
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
    
    # Input section for ongoing conversation
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Continue the conversation...", 
        height=120, 
        placeholder="Tell me more... / How are you feeling now... / I wanted to share...",
        key=f"conversation_input_{st.session_state.last_input_key}",
        help="Share your thoughts, feelings, or ask Jumbo anything!"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Send to Jumbo 🐘", key="send_message", use_container_width=True):
            if user_input and user_input.strip():
                handle_user_message(user_input)
            else:
                st.warning("Please enter a message to send to Jumbo!")
    
    # Quick response buttons for common emotions
    st.markdown("**Quick responses:**")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_responses = [
        ("I'm feeling good today", "😊"),
        ("I'm stressed", "😰"), 
        ("I need support", "🤗"),
        ("Tell me about yourself", "🐘")
    ]
    
    for i, (response_text, emoji) in enumerate(quick_responses):
        with [col1, col2, col3, col4][i]:
            if st.button(f"{emoji}\n{response_text}", key=f"quick_{i}", use_container_width=True):
                handle_user_message(response_text)
                
    st.markdown('</div>', unsafe_allow_html=True)

# Fix the main function call at the end
if __name__ == "__main__":
    main()
    # Remove the duplicate display_conversation() call


