"""
STEP 5: MEETMIND PIPELINE ORCHESTRATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Professor's Note:
  This is the "glue" module. It ties all our components together
  into a single coherent pipeline.

  Design Pattern: Pipeline / Chain of Responsibility
    Each step receives output from the previous step.
    Steps are loosely coupled — you can swap any component.
    
  State management:
    MeetMindState holds everything generated during a session.
    This makes it easy to regenerate parts without rerunning everything.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MeetMindState:
    """Holds all state for a processed meeting session."""
    # Inputs
    source_path: str = ""
    source_type: str = ""  # "audio" or "transcript"
    
    # Outputs from each pipeline stage
    segments: list = field(default_factory=list)
    emotion_results: dict = field(default_factory=dict)
    speaker_profiles: list = field(default_factory=list)
    summary: object = None
    emotion_report: str = ""
    
    # Processing metadata
    processing_time: float = 0.0
    is_indexed: bool = False
    error: Optional[str] = None


class MeetMindPipeline:
    """
    Orchestrates the full MeetMind processing pipeline.
    
    Usage:
        pipeline = MeetMindPipeline(openai_api_key="sk-...")
        state = pipeline.process_audio("meeting.mp3")
        answer, chunks = pipeline.chat("What were the action items?", state)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        hf_token: Optional[str] = None,
        whisper_model: str = "base",
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        use_diarization: bool = True,
        persist_dir: str = "./data/chromadb"
    ):
        from transcriber import MeetingTranscriber
        from emotion_analyzer import EmotionAnalyzer
        from meeting_analyzer import MeetingAnalyzer
        from rag_chatbot import MeetingRAG

        self.transcriber = MeetingTranscriber(
            whisper_model=whisper_model,
            use_diarization=use_diarization,
            hf_token=hf_token
        )
        self.emotion_analyzer = EmotionAnalyzer()
        self.meeting_analyzer = MeetingAnalyzer(
            provider=llm_provider,
            model=llm_model,
            api_key=openai_api_key
        )
        self.rag = MeetingRAG(persist_directory=persist_dir)

    # ─── Step A: Transcription ─────────────────────────────────────────────────
    def transcribe(self, source_path: str) -> tuple[list, str]:
        """
        Auto-detect source type and transcribe.
        Returns (segments, source_type)
        """
        path = Path(source_path)
        ext = path.suffix.lower()
        
        AUDIO_FORMATS = {'.mp3', '.mp4', '.m4a', '.wav', '.ogg', '.flac', '.webm'}
        TEXT_FORMATS  = {'.txt', '.md', '.text'}
        
        if ext in AUDIO_FORMATS:
            logger.info(f"Detected audio file: {ext}")
            segments = self.transcriber.transcribe(source_path)
            return segments, "audio"
        
        elif ext in TEXT_FORMATS:
            logger.info(f"Detected text transcript: {ext}")
            with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_text = f.read()
            segments = self.transcriber.transcribe_text(raw_text)
            return segments, "transcript"
        
        else:
            # Try as text
            with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_text = f.read()
            segments = self.transcriber.transcribe_text(raw_text)
            return segments, "transcript"

    # ─── Step B: Analyze Emotions ─────────────────────────────────────────────
    def analyze_emotions(self, segments: list) -> tuple[dict, list, str]:
        """
        Run emotion analysis on all segments.
        Returns (emotion_results_dict, speaker_profiles, emotion_report_str)
        """
        emotion_results = self.emotion_analyzer.analyze_transcript(segments)
        speaker_profiles = self.emotion_analyzer.build_speaker_profiles(
            segments, emotion_results
        )
        emotion_report = self.emotion_analyzer.generate_emotion_report(speaker_profiles)
        return emotion_results, speaker_profiles, emotion_report

    # ─── Step C: Extract Meeting Intelligence ────────────────────────────────
    def analyze_meeting(self, segments: list, duration_minutes: float):
        """Run LLM analysis for summary and action items."""
        import datetime
        date = datetime.datetime.now().strftime("%B %d, %Y")
        summary = self.meeting_analyzer.extract_meeting_insights(
            segments, duration_minutes, date
        )
        return summary

    # ─── Step D: Index for RAG ────────────────────────────────────────────────
    def index(self, segments: list, summary_text: str, emotion_report: str):
        """Index everything into ChromaDB for chatbot queries."""
        self.rag.index_meeting(segments, summary_text, emotion_report)

    # ─── Full Pipeline ────────────────────────────────────────────────────────
    def process(self, source_path: str, progress_callback=None) -> MeetMindState:
        """
        Run the complete pipeline on an audio file or transcript.
        
        Args:
            source_path: path to audio file or text transcript
            progress_callback: optional function(step, message) for UI updates
        
        Returns:
            MeetMindState with all results
        """
        state = MeetMindState(source_path=source_path)
        start_time = time.time()

        def progress(step: str, msg: str):
            logger.info(f"[{step}] {msg}")
            if progress_callback:
                progress_callback(step, msg)

        try:
            # ── Stage 1: Transcription ──────────────────────────────────────
            progress("TRANSCRIBE", "Transcribing audio and identifying speakers...")
            segments, source_type = self.transcribe(source_path)
            state.segments = segments
            state.source_type = source_type
            
            if not segments:
                state.error = "No speech detected in the input."
                return state
            
            progress("TRANSCRIBE", f"✅ Transcribed {len(segments)} speaker turns")

            # ── Stage 2: Emotion Analysis ───────────────────────────────────
            progress("EMOTION", "Analyzing emotional tone per speaker...")
            emotion_results, speaker_profiles, emotion_report = self.analyze_emotions(segments)
            state.emotion_results = emotion_results
            state.speaker_profiles = speaker_profiles
            state.emotion_report = emotion_report
            progress("EMOTION", f"✅ Analyzed emotions for {len(speaker_profiles)} speakers")

            # ── Stage 3: LLM Analysis ───────────────────────────────────────
            progress("ANALYZE", "Extracting summary and action items with AI...")
            duration = max(seg.end for seg in segments) / 60 if segments else 0
            summary = self.analyze_meeting(segments, duration)
            state.summary = summary
            progress("ANALYZE", f"✅ Found {len(summary.action_items)} action items")

            # ── Stage 4: Index for RAG ──────────────────────────────────────
            progress("INDEX", "Building searchable index for chatbot...")
            self.index(
                segments,
                summary_text=summary.executive_summary + "\n\n" + summary.to_markdown(),
                emotion_report=emotion_report
            )
            state.is_indexed = True
            progress("INDEX", "✅ Meeting indexed — chatbot ready!")

            state.processing_time = time.time() - start_time
            progress("DONE", f"✅ Processing complete in {state.processing_time:.1f}s")

        except Exception as e:
            state.error = str(e)
            logger.error(f"Pipeline error: {e}", exc_info=True)

        return state

    def chat(self, question: str, state: MeetMindState) -> tuple[str, list]:
        """
        Ask a question about the processed meeting.
        
        Returns:
            (answer_str, retrieved_chunks)
        """
        if not state.is_indexed:
            return "Please process a meeting first.", []
        
        return self.rag.chat(question, self.meeting_analyzer)

    def reset_chat(self):
        """Clear conversation history."""
        self.rag.clear_history()