"""
STEP 4: RAG CHATBOT ENGINE
━━━━━━━━━━━━━━━━━━━━━━━━━
Professor's Note:
  RAG = Retrieval Augmented Generation.
  The core insight: LLMs have limited context windows.
  You can't dump a 2-hour transcript (50k words) into GPT-4.
  
  Solution — Split, Embed, Retrieve, Generate:
  
    1. SPLIT: Divide transcript into small overlapping chunks
       Why overlapping? Answers often span chunk boundaries.
    
    2. EMBED: Convert each chunk to a vector (list of numbers)
       that captures its semantic meaning.
       Similar text → similar vectors.
    
    3. STORE: Save vectors in ChromaDB (a vector database)
       Think of it as a "semantic search engine" for your text.
    
    4. RETRIEVE: When user asks a question:
       - Convert question to vector
       - Find chunks with most similar vectors (cosine similarity)
       - Return top-k most relevant chunks
    
    5. GENERATE: Send retrieved chunks + question to LLM.
       LLM answers using ONLY that context.
  
  Why ChromaDB?
    - Runs locally (no external API needed)
    - Persistent storage (survives app restarts)
    - Fast similarity search even on large datasets
    - Open source and actively maintained
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved text chunk with metadata."""
    text: str
    speaker: str
    timestamp: str
    relevance_score: float
    chunk_id: str


class MeetingRAG:
    """
    RAG-powered chatbot for meeting Q&A.
    
    After a meeting is processed, this allows natural language queries:
      "What did John say about the budget?"
      "What are my action items?"
      "What was the main disagreement about?"
    """

    def __init__(
        self,
        collection_name: str = "meeting_transcript",
        persist_directory: str = "./data/chromadb",
        embedding_model: str = "all-MiniLM-L6-v2",  # fast, good quality
        chunk_size: int = 300,      # characters per chunk
        chunk_overlap: int = 50,    # overlap between chunks
        top_k: int = 4             # how many chunks to retrieve
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        
        self._chroma_client = None
        self._collection = None
        self._embedder = None
        self.chat_history = []  # for multi-turn conversation

    def _get_embedder(self):
        """
        Load sentence-transformers embedding model.
        
        We use 'all-MiniLM-L6-v2':
          - 384-dimensional embeddings
          - Excellent speed/quality tradeoff
          - 80MB model size
          - Trained on 1B sentence pairs
        
        Alternatives:
          - 'all-mpnet-base-v2': better quality, slower
          - 'text-embedding-ada-002': OpenAI's API (costs money)
        """
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model}")
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _get_collection(self):
        """Get or create ChromaDB collection."""
        if self._collection is None:
            import chromadb
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            self._chroma_client = chromadb.PersistentClient(
                path=self.persist_directory
            )
            
            # Delete existing collection for fresh meeting
            try:
                self._chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass
            
            self._collection = self._chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # cosine similarity for text
            )
        return self._collection

    def _chunk_transcript(self, segments: list) -> list[dict]:
        """
        Split transcript into overlapping chunks for indexing.
        
        Strategy: We chunk by SPEAKER TURNS first, then by size.
        This preserves conversational context — keeping one speaker's
        continuous utterance together is semantically valuable.
        
        Each chunk gets metadata:
          - speaker name
          - timestamp
          - position in document
        """
        chunks = []
        
        for seg in segments:
            text = f"[{seg.speaker}]: {seg.text}"
            mins = int(seg.start // 60)
            secs = int(seg.start % 60)
            timestamp = f"{mins:02d}:{secs:02d}"
            
            # If segment fits in one chunk, add directly
            if len(text) <= self.chunk_size:
                chunks.append({
                    "text": text,
                    "speaker": seg.speaker,
                    "timestamp": timestamp,
                    "start": seg.start,
                    "chunk_id": f"{seg.speaker}_{timestamp}"
                })
            else:
                # Split long segments with overlap
                words = text.split()
                current_chunk = []
                current_size = 0
                
                for word in words:
                    current_chunk.append(word)
                    current_size += len(word) + 1
                    
                    if current_size >= self.chunk_size:
                        chunk_text = " ".join(current_chunk)
                        chunks.append({
                            "text": chunk_text,
                            "speaker": seg.speaker,
                            "timestamp": timestamp,
                            "start": seg.start,
                            "chunk_id": f"{seg.speaker}_{timestamp}_{len(chunks)}"
                        })
                        # Keep overlap
                        overlap_words = current_chunk[-self.chunk_overlap//5:]
                        current_chunk = overlap_words
                        current_size = sum(len(w) + 1 for w in overlap_words)
                
                if current_chunk:
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "speaker": seg.speaker,
                        "timestamp": timestamp,
                        "start": seg.start,
                        "chunk_id": f"{seg.speaker}_{timestamp}_end"
                    })
        
        return chunks

    def index_meeting(self, segments: list, summary: str = "", emotion_report: str = ""):
        """
        Index meeting content into ChromaDB.
        
        We index:
          1. Speaker segments (chunked)
          2. Meeting summary (for high-level questions)
          3. Emotion report (for sentiment questions)
        
        Professor's Note on Embeddings:
          Each text chunk gets converted to a vector by the embedding model.
          Vector = [0.23, -0.11, 0.89, ...] — 384 numbers that encode meaning.
          "budget discussion" and "financial review" → similar vectors.
          "lunch break" and "financial review" → very different vectors.
        """
        collection = self._get_collection()
        embedder = self._get_embedder()
        
        logger.info("Chunking transcript...")
        chunks = self._chunk_transcript(segments)
        
        # Add summary and emotion report as special chunks
        if summary:
            chunks.append({
                "text": f"[MEETING SUMMARY]: {summary}",
                "speaker": "SYSTEM",
                "timestamp": "00:00",
                "start": 0,
                "chunk_id": "meeting_summary"
            })
        
        if emotion_report:
            chunks.append({
                "text": f"[EMOTION ANALYSIS]: {emotion_report}",
                "speaker": "SYSTEM",
                "timestamp": "00:00",
                "start": 0,
                "chunk_id": "emotion_report"
            })
        
        logger.info(f"Embedding {len(chunks)} chunks...")
        
        # Batch embed for efficiency
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True).tolist()
        
        # Store in ChromaDB
        collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "speaker": c["speaker"],
                "timestamp": c["timestamp"],
                "start": c["start"]
            } for c in chunks]
        )
        
        logger.info(f"Indexed {len(chunks)} chunks successfully!")

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Retrieve most relevant chunks for a query.
        
        Process:
          1. Embed the query using same model used for indexing
          2. ChromaDB computes cosine similarity: query_vec · chunk_vec
          3. Returns top-k closest chunks
        
        Cosine similarity: measures angle between vectors.
          Score = 1.0 → identical meaning
          Score = 0.0 → completely unrelated
        """
        collection = self._get_collection()
        embedder = self._get_embedder()
        
        query_embedding = embedder.encode([query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(self.top_k, collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            retrieved.append(RetrievedChunk(
                text=doc,
                speaker=meta["speaker"],
                timestamp=meta["timestamp"],
                relevance_score=1 - dist,  # convert distance to similarity
                chunk_id=f"{meta['speaker']}_{meta['timestamp']}"
            ))
        
        return retrieved

    def chat(self, user_message: str, analyzer) -> tuple[str, list[RetrievedChunk]]:
        """
        Full RAG pipeline: retrieve relevant context, then generate answer.
        
        Returns:
          - answer string
          - retrieved chunks (for transparency/debugging)
        
        Multi-turn memory:
          We maintain chat_history so users can ask follow-up questions.
          "What did John say?" → "What was his reasoning?"
          The second question needs prior context to understand "his".
        """
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(user_message)
        
        if not retrieved_chunks:
            return "I couldn't find relevant information in this meeting.", []
        
        # Format context for LLM
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"[Chunk {i+1} | {chunk.speaker} at {chunk.timestamp} | "
                f"Relevance: {chunk.relevance_score:.0%}]\n{chunk.text}"
            )
        context = "\n\n".join(context_parts)
        
        # Add conversation history for follow-up questions
        if self.chat_history:
            recent_history = "\n".join([
                f"Q: {turn['question']}\nA: {turn['answer']}"
                for turn in self.chat_history[-3:]  # last 3 turns
            ])
            context = f"Recent conversation:\n{recent_history}\n\nMeeting content:\n{context}"
        
        # Generate answer
        answer = analyzer.answer_question(user_message, context)
        
        # Store in history
        self.chat_history.append({
            "question": user_message,
            "answer": answer
        })
        
        return answer, retrieved_chunks

    def clear_history(self):
        """Reset conversation history."""
        self.chat_history = []
        logger.info("Chat history cleared.")

    def get_suggested_questions(self, summary) -> list[str]:
        """
        Auto-generate suggested questions based on the meeting content.
        Helps users discover what they can ask.
        """
        questions = [
            "What were the main decisions made?",
            "What are my action items?",
            "Who was responsible for what?",
        ]
        
        if summary and summary.participants:
            for participant in summary.participants[:2]:
                questions.append(f"What did {participant} contribute to the discussion?")
        
        if summary and summary.open_questions:
            questions.append("What questions were left unresolved?")
        
        return questions
