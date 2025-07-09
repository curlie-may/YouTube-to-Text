#Incorporates yt-dlp, removes youtube-transcript-api, keep SmarProxy
#Add ['en.*'] to line subtitleslangs to accept all English captions 
import re
import asyncio
from typing import List, Dict
import os
import smtplib
import email.mime.text
import email.mime.multipart
import difflib
import random
import requests
import yt_dlp
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Smartproxy (Decodo) configuration
SMARTPROXY_USERNAME = os.getenv('SMARTPROXY_USERNAME')
SMARTPROXY_PASSWORD = os.getenv('SMARTPROXY_PASSWORD')
SMARTPROXY_ENDPOINT = os.getenv('SMARTPROXY_ENDPOINT', 'us.decodo.com')
SMARTPROXY_PORT = os.getenv('SMARTPROXY_PORT', '10001')

# Global counter for conversions (in production, use a database)
conversion_counter = 0

class TranscriptRequest(BaseModel):
    youtube_url: str

class TranscriptResponse(BaseModel):
    original_text: str
    cleaned_text: str
    corrections_made: List[Dict[str, str]]
    confidence_score: float

class ChunkProgress(BaseModel):
    chunk_number: int
    total_chunks: int
    input_length: int
    output_length: int
    status: str  # "processing", "completed", "error"

class FeedbackRequest(BaseModel):
    name: str = ""
    email: str = ""
    comments: str

def get_proxy_config():
    """Generate proxy configuration for requests"""
    if not SMARTPROXY_USERNAME or not SMARTPROXY_PASSWORD:
        return None
    
    # Simple authentication - no sessions for country endpoint
    proxy_user = f"{SMARTPROXY_USERNAME}"
    
    proxy_config = {
        'http': f'http://{proxy_user}:{SMARTPROXY_PASSWORD}@{SMARTPROXY_ENDPOINT}:{SMARTPROXY_PORT}',
        'https': f'http://{proxy_user}:{SMARTPROXY_PASSWORD}@{SMARTPROXY_ENDPOINT}:{SMARTPROXY_PORT}'
    }
    
    return proxy_config

class CaptionCleaningAgent:
    def __init__(self):
        self.correction_patterns = {
            r'\bchachi bt\b': 'ChatGPT',
            r'\bchat gpt\b': 'ChatGPT',
            r'\belon must\b': 'Elon Musk',
            r'\bapple soft\b': 'Microsoft',
            r'\bgoogle\s+chrome\b': 'Google Chrome',
            r'\byou\s+tube\b': 'YouTube',
            r'\bface\s+book\b': 'Facebook',
            r'\blinked\s+in\b': 'LinkedIn',
            r'\btik\s+tok\b': 'TikTok',
            r'\binsta\s+gram\b': 'Instagram',
        }
    
    async def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError("Invalid YouTube URL")
    
    async def get_captions(self, video_id: str) -> str:
        """Get captions from YouTube video using yt-dlp with proxy"""
        try:
            # Prepare yt-dlp options
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en.*'],   #en.* accepts all English captions
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            # Add proxy configuration if available
            if SMARTPROXY_USERNAME and SMARTPROXY_PASSWORD:
                # Simple yt-dlp proxy format for country endpoint
                proxy_user = f"{SMARTPROXY_USERNAME}"
                proxy_url = f"http://{proxy_user}:{SMARTPROXY_PASSWORD}@{SMARTPROXY_ENDPOINT}:{SMARTPROXY_PORT}"
                ydl_opts['proxy'] = proxy_url
                print(f"Using Smartproxy: {SMARTPROXY_ENDPOINT}:{SMARTPROXY_PORT}")
                print(f"Proxy URL format: http://{proxy_user}:****@{SMARTPROXY_ENDPOINT}:{SMARTPROXY_PORT}")
            else:
                print("No proxy configuration found, using direct connection")
            
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info and subtitles
                info = ydl.extract_info(video_url, download=False)
                
                # Get subtitles
                subtitles = info.get('subtitles', {})
                auto_subtitles = info.get('automatic_captions', {})
                
                # Try to get English subtitles
                transcript_text = ""
                
                if 'en' in subtitles:
                    # Use manual subtitles first (higher quality)
                    subtitle_entries = subtitles['en']
                    transcript_text = await self.extract_text_from_subtitle_entries(subtitle_entries, True)
                elif 'en' in auto_subtitles:
                    # Fall back to auto-generated subtitles
                    subtitle_entries = auto_subtitles['en']
                    transcript_text = await self.extract_text_from_subtitle_entries(subtitle_entries, True)
                else:
                    raise Exception("No English subtitles found for this video")
                
                print(f"Successfully retrieved transcript for video {video_id} (using proxy: {SMARTPROXY_USERNAME is not None})")
                return transcript_text.strip()
                
        except Exception as e:
            print(f"Error getting captions for video {video_id}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Could not retrieve captions: {str(e)}")
    
    async def extract_text_from_subtitle_entries(self, subtitle_entries: List[Dict], use_proxy: bool = False) -> str:
        """Extract text from subtitle entries"""
        try:
            # Find the best subtitle format (prefer vtt, then srt)
            best_entry = None
            for entry in subtitle_entries:
                if entry.get('ext') == 'vtt':
                    best_entry = entry
                    break
                elif entry.get('ext') == 'srv1':  # YouTube's JSON format
                    best_entry = entry
                    break
                elif entry.get('ext') == 'srt':
                    best_entry = entry
            
            if not best_entry:
                best_entry = subtitle_entries[0]  # Use first available
            
            subtitle_url = best_entry['url']
            
            # Download subtitle content
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            proxies = {}
            if use_proxy and SMARTPROXY_USERNAME:
                # Simple proxy format for country endpoint
                proxy_user = f"{SMARTPROXY_USERNAME}"
                proxy_url = f"http://{proxy_user}:{SMARTPROXY_PASSWORD}@{SMARTPROXY_ENDPOINT}:{SMARTPROXY_PORT}"
                proxies = {'http': proxy_url, 'https': proxy_url}
            
            response = requests.get(subtitle_url, headers=headers, proxies=proxies, timeout=30)
            response.raise_for_status()
            
            # DEBUG: Add debugging information
            print(f"Response status: {response.status_code}")
            print(f"Response content type: {response.headers.get('content-type', 'unknown')}")
            print(f"Response content (first 200 chars): {response.text[:200]}")
            print(f"Subtitle format detected: {best_entry.get('ext')}")
            print(f"Subtitle URL: {subtitle_url}")
            
            # Parse subtitle content based on format
            if best_entry.get('ext') == 'srv1':
                # Handle YouTube's XML format (srv1 is XML, not JSON)
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(response.text)
                    text_parts = []
                    
                    # Extract text from <text> elements
                    for text_elem in root.findall('text'):
                        text_content = text_elem.text
                        if text_content:
                            # Decode HTML entities
                            import html
                            decoded_text = html.unescape(text_content)
                            text_parts.append(decoded_text)
                    
                    return ' '.join(text_parts)
                    
                except ET.ParseError as e:
                    print(f"XML parsing error: {e}")
                    print(f"Response content: {response.text[:500]}")
                    raise Exception(f"Failed to parse XML subtitle content: {e}")
            
            else:
                # Handle VTT/SRT formats - extract text only
                lines = response.text.split('\n')
                text_parts = []
                
                for line in lines:
                    line = line.strip()
                    # Skip timestamp lines and metadata
                    if (line and 
                        not line.startswith('WEBVTT') and 
                        not line.startswith('NOTE') and
                        not '-->' in line and
                        not line.isdigit() and
                        not line.startswith('<')):
                        
                        # Remove VTT formatting tags
                        clean_line = re.sub(r'<[^>]+>', '', line)
                        if clean_line:
                            text_parts.append(clean_line)
                
                return ' '.join(text_parts)
                
        except Exception as e:
            print(f"Error extracting subtitle text: {str(e)}")
            raise Exception(f"Failed to extract subtitle text: {str(e)}")
    
    async def detect_video_context(self, text: str) -> str:
        """Analyze video content to understand context/domain"""
        context_prompt = f"""
        Analyze this video transcript excerpt and identify the main topic/domain:
        
        Text: "{text[:500]}..."
        
        Respond with just the primary domain (e.g., "technology", "business", "education", "entertainment", "science", etc.)
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": context_prompt}],
                max_tokens=50,
                temperature=0.1
            )
            return response.choices[0].message.content.strip().lower()
        except:
            return "general"
    
    async def llm_correct_term(self, term: str, context: str) -> str:
        """Use LLM to correct a specific term given context"""
        prompt = f"""
        Given the context "{context}", what is the most likely correct spelling/form of "{term}"?
        
        Only respond with the corrected term, nothing else. If the term seems correct already, respond with the original term.
        
        Examples:
        - "chachi bt" in technology context → "ChatGPT"
        - "elon must" in business context → "Elon Musk"
        - "apple soft" in technology context → "Microsoft"
        
        Term to correct: "{term}"
        Context: "{context}"
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except:
            return term
    
    async def apply_pattern_corrections(self, text: str) -> tuple[str, List[Dict[str, str]]]:
        """Apply known correction patterns"""
        corrections = []
        corrected_text = text
        
        for pattern, replacement in self.correction_patterns.items():
            matches = re.finditer(pattern, corrected_text, re.IGNORECASE)
            for match in matches:
                original = match.group(0)
                corrections.append({
                    "original": original,
                    "corrected": replacement,
                    "method": "pattern_match"
                })
                corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text, corrections
    
    async def chunk_text(self, text: str, max_chunk_size: int = 4500, overlap: int = 300) -> List[str]:
        """Split long text into overlapping chunks for better context preservation"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            # If we're not at the end of the text, try to break at a sentence
            if end < len(text):
                # Look back for a sentence ending within last 300 chars
                break_point = text.rfind('.', start, end)
                if break_point > start + (max_chunk_size - 300):
                    end = break_point + 1
                else:
                    # If no sentence ending found, look for other punctuation
                    for punct in ['!', '?', ';']:
                        break_point = text.rfind(punct, start, end)
                        if break_point > start + (max_chunk_size - 300):
                            end = break_point + 1
                            break
            
            chunks.append(text[start:end].strip())
            
            # Move start position with overlap (but not for first chunk)
            if start == 0:
                start = end
            else:
                start = end - overlap  # Overlap with previous chunk
                
            # Safety check to prevent infinite loop
            if start >= len(text):
                break
        
        return chunks
    
    async def ai_clean_chunk(self, chunk: str, context: str) -> str:
        """Use AI to clean a chunk of text"""
        prompt = f"""
        You are a YouTube caption error corrector. Your ONLY job is to fix obvious errors while keeping the text exactly as natural speech.
        
        CRITICAL: You MUST preserve all natural sentence breaks and speaking patterns.
        
        Make these changes:
        1. Fix obvious misheard names: "chachi bt" → "ChatGPT", "elon must" → "Elon Musk"
        2. Fix clear brand errors: "apple soft" → "Microsoft", "you tube" → "YouTube"  
        3. Add periods at the end of complete thoughts 
        4. Remove filler words "um", "uh" but KEEP natural pauses and speech rhythm
        5. A sentence should have at minimum a subject and a verb
        6. Remove repeated sentence fragments that often occur at chunk overlap 
        
        ABSOLUTELY DO NOT:
        - Change the speaker's natural speaking style
        - Make sentences longer or shorter than they naturally are
        - Change casual language to formal language
        
        KEEP EXACTLY:
        - Natural speaking rhythm and pauses
        - Conversational tone and style
        - Original sentence length and structure
        
        The video context is: {context}
        
        Text to minimally correct:
        {chunk}
        
        Return the text with ONLY the obvious name/brand fixes and minimal punctuation additions.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return chunk
    
    async def ai_clean_chunk_with_progress(self, chunk: str, context: str, chunk_index: int, total_chunks: int, progress_callback=None) -> str:
        """Clean a chunk with progress reporting"""
        
        try:
            cleaned_chunk = await self.ai_clean_chunk(chunk, context)
            return cleaned_chunk
            
        except Exception as e:
            raise e
    
    def normalize_for_comparison(self, sentence):
        """Normalize sentence for better duplicate detection"""
        # Convert to lowercase
        normalized = sentence.lower().strip()
        # Replace number words with digits
        number_replacements = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
        }
        for word, digit in number_replacements.items():
            normalized = re.sub(r'\b' + word + r'\b', digit, normalized)
        
        # Remove extra punctuation and spaces for comparison
        normalized = re.sub(r'[,;:\-\(\)]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    async def post_process_cleanup(self, text: str) -> str:
        """Enhanced cleanup with improved fuzzy matching to remove overlapping content and fix basic formatting"""
        
        # Remove obvious duplicate sentences that appear due to overlapping
        sentences = text.split('.')
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Ignore very short fragments
                
                # Check against the last few sentences for fuzzy duplicates
                is_duplicate = False
                normalized_sentence = self.normalize_for_comparison(sentence)
                
                for j in range(max(0, len(cleaned_sentences) - 5), len(cleaned_sentences)):
                    prev_sentence = cleaned_sentences[j].strip()
                    normalized_prev = self.normalize_for_comparison(prev_sentence)
                    
                    # Calculate similarity ratio using difflib
                    similarity = difflib.SequenceMatcher(None, normalized_sentence, normalized_prev).ratio()
                    
                    # Lowered threshold to 70% to catch more duplicates
                    if similarity > 0.7:
                        is_duplicate = True
                        break
                    
                    # Enhanced word overlap detection
                    sentence_words = set(normalized_sentence.split())
                    prev_words = set(normalized_prev.split())
                    
                    if len(sentence_words) > 3 and len(prev_words) > 3:
                        word_overlap = len(sentence_words & prev_words) / max(len(sentence_words), len(prev_words))
                        # Lowered word overlap threshold and increased length tolerance
                        if word_overlap > 0.6 and abs(len(sentence) - len(prev_sentence)) < 100:
                            is_duplicate = True
                            break
                    
                    # Check for containment (one sentence contains most of another)
                    if len(sentence_words) > 0 and len(prev_words) > 0:
                        if (len(sentence_words & prev_words) / len(sentence_words) > 0.8 or 
                            len(sentence_words & prev_words) / len(prev_words) > 0.8):
                            is_duplicate = True
                            break
                
                # Remove common fragment patterns - using simple string concatenation to avoid regex issues
                is_fragment = False
                sentence_lower = sentence.lower().strip()
                
                # Check for specific fragment patterns
                fragment_starts = [
                    'hand ',
                    'you\'ve been taught finally',
                    'ss esp especially',
                    'this criteria to count',
                    'this means that you need to be more cognizant about which'
                ]
                
                for fragment in fragment_starts:
                    if sentence_lower.startswith(fragment):
                        is_fragment = True
                        break
                
                # Check for very short fragments (1-3 words)
                word_count = len(sentence.split())
                if word_count <= 3 and word_count > 0:
                    # Check if it's just "number one", "number two", etc.
                    if sentence_lower.startswith('number '):
                        is_fragment = True
                
                if not is_duplicate and not is_fragment:
                    cleaned_sentences.append(sentence)
        
        # Rejoin sentences
        text = '. '.join(cleaned_sentences)
        
        # Basic formatting cleanup
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'\.+', '.', text)  # Multiple periods
        text = text.strip()
        
        # Ensure it ends with a period
        if text and not text.endswith('.'):
            text += '.'
            
        return text
    
    async def ai_deep_clean(self, text: str, context: str, progress_callback=None) -> tuple[str, List[Dict[str, str]]]:
        """Use AI to perform deep cleaning of the text with sequential chunk processing"""
        # Split text into chunks if it's too long
        chunks = await self.chunk_text(text)
        cleaned_chunks = []
        
        # Process chunks sequentially (one at a time)
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")  # Debug logging
            print(f"Chunk {i+1} input length: {len(chunk)} characters")  # New debug line
            try:
                cleaned_chunk = await self.ai_clean_chunk(chunk, context)
                
                cleaned_chunks.append(cleaned_chunk)
                print(f"Chunk {i+1} output length: {len(cleaned_chunk)} characters")  # New debug line
                print(f"Successfully processed chunk {i+1}")  # Debug logging
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")  # Debug logging
                # If a chunk fails, use the original text for that chunk
                cleaned_chunks.append(chunk)
        
        # Join all cleaned chunks
        raw_text = "\n\n".join(cleaned_chunks)
        
        # Apply post-processing cleanup
        final_text = await self.post_process_cleanup(raw_text)
        
        # Debug logging
        print(f"Total chunks joined: {len(cleaned_chunks)}")
        print(f"Raw text length: {len(raw_text)} characters")
        print(f"Final cleaned text length: {len(final_text)} characters")
        print(f"First 100 characters: {final_text[:100]}")
        print(f"Last 100 characters: {final_text[-100:]}")
        
        corrections = [{
            "original": f"Processed {len(chunks)} text chunks sequentially",
            "corrected": "Grammar, punctuation, formatting, and context fixes applied",
            "method": "ai_deep_clean_with_post_processing"
        }]
        
        return final_text, corrections
    
    async def calculate_confidence(self, original: str, cleaned: str, corrections: List[Dict]) -> float:
        """Calculate confidence score for the cleaning"""
        if not corrections:
            return 0.5
        
        # Simple confidence calculation based on number and type of corrections
        pattern_corrections = len([c for c in corrections if c["method"] == "pattern_match"])
        ai_corrections = len([c for c in corrections if c["method"] == "ai_deep_clean"])
        
        # Higher confidence for pattern matches, moderate for AI corrections
        confidence = min(0.95, 0.6 + (pattern_corrections * 0.1) + (ai_corrections * 0.2))
        return confidence
    
    async def process_transcript(self, request: TranscriptRequest, progress_callback=None) -> TranscriptResponse:
        """Main agent workflow to process transcript"""
        global conversion_counter
        
        try:
            # Step 1: Extract video ID and get captions
            video_id = await self.extract_video_id(request.youtube_url)
            original_text = await self.get_captions(video_id)
            
            # Step 2: Detect video context
            context = await self.detect_video_context(original_text)
            
            # Step 3: Apply pattern corrections
            pattern_corrected, pattern_corrections = await self.apply_pattern_corrections(original_text)
            
            # Step 4: AI deep cleaning
            final_text, ai_corrections = await self.ai_deep_clean(pattern_corrected, context)
            
            # Step 5: Combine corrections and calculate confidence
            all_corrections = pattern_corrections + ai_corrections
            confidence = await self.calculate_confidence(original_text, final_text, all_corrections)
            
            # Increment conversion counter on successful processing
            conversion_counter += 1
            
            return TranscriptResponse(
                original_text=original_text,
                cleaned_text=final_text,
                corrections_made=all_corrections,
                confidence_score=confidence
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize agent
agent = CaptionCleaningAgent()

# Email sending function
async def send_feedback_email(feedback: FeedbackRequest):
    """Send feedback email via SMTP"""
    try:
        # Get SMTP configuration from environment variables
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if not smtp_user or not smtp_password:
            raise HTTPException(status_code=500, detail="SMTP configuration not set")
        
        # Create message
        msg = email.mime.multipart.MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = "lmantese6@gmail.com"
        msg['Subject'] = "YouTube Transcript App - User Feedback"
        
        # Create email body
        body = f"""
New feedback received from YouTube Transcript App:

Name: {feedback.name if feedback.name else 'Not provided'}
Email: {feedback.email if feedback.email else 'Not provided'}

Comments:
{feedback.comments}

---
Sent from YouTube Transcript App
        """
        
        msg.attach(email.mime.text.MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        
        return {"message": "Feedback sent successfully"}
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to send feedback")

@app.post("/get-transcript-info")
async def get_transcript_info(request: TranscriptRequest):
    """Get transcript length for timing estimation"""
    try:
        video_id = await agent.extract_video_id(request.youtube_url)
        original_text = await agent.get_captions(video_id)
        return {
            "transcript_length": len(original_text),
            "estimated_processing_time": max(60, len(original_text) // 100)  # 1 sec per 100 chars, minimum 1 minute
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean-transcript", response_model=TranscriptResponse)
async def clean_transcript(request: TranscriptRequest):
    """Clean YouTube video transcript"""
    result = await agent.process_transcript(request)
    return result

@app.post("/send-feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback"""
    return await send_feedback_email(feedback)

@app.get("/conversion-count")
async def get_conversion_count():
    """Get the current conversion count"""
    return {"count": conversion_counter}

@app.get("/health")
async def health_check():
    """Health check endpoint with proxy status"""
    proxy_status = "configured" if SMARTPROXY_USERNAME else "not configured"
    return {
        "status": "healthy", 
        "proxy_status": proxy_status,
        "proxy_endpoint": f"{SMARTPROXY_ENDPOINT}:{SMARTPROXY_PORT}" if SMARTPROXY_USERNAME else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
