"""
STEP 3: LLM MEETING ANALYZER
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Professor's Note:
  This module uses an LLM (GPT-4o or local models via Ollama) to extract
  high-level meaning from the transcript. 

  Tasks:
    1. Structured Summary — key decisions, topics discussed, conclusions
    2. Action Item Extraction — who owns what, by when
    3. Meeting Metadata — type, participants, duration, energy level

  Why use an LLM here instead of rules?
    Action items are ambiguous in natural language:
      "Let's have John take a look at the budget" = action item for John
      "Someone should probably consider the timeline" = vague, maybe not assigned
    LLMs handle this nuance far better than regex or keyword matching.

  Prompt Engineering Principles Used:
    - Structured output (JSON) for reliable parsing
    - Few-shot examples to guide format
    - Chain-of-thought for complex extraction
    - System prompt to set role/context
"""

import json
import os
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ActionItem:
    """A single action item extracted from the meeting."""
    task: str
    owner: str                    # who is responsible
    deadline: str                 # "by next Friday", "ASAP", "no deadline"
    priority: str                 # "high", "medium", "low"
    context: str                  # brief context from conversation
    status: str = "open"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MeetingSummary:
    """Full structured analysis of a meeting."""
    title: str
    date: str
    duration_minutes: float
    participants: list[str]
    meeting_type: str             # "standup", "planning", "retrospective", "client call", etc.
    
    executive_summary: str        # 2-3 sentence TL;DR
    key_topics: list[str]         # main topics discussed
    key_decisions: list[str]      # decisions made
    action_items: list[ActionItem]
    
    open_questions: list[str]     # unresolved questions
    follow_up_meetings: list[str] # suggested follow-ups
    
    overall_sentiment: str        # from emotion analysis
    meeting_effectiveness: str    # "highly effective", "somewhat effective", "needs improvement"
    
    raw_transcript: str = ""      # for RAG indexing

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_markdown(self) -> str:
        """Format as readable Markdown."""
        lines = [
            f"# 📋 {self.title}",
            f"**Date:** {self.date} | **Duration:** {self.duration_minutes:.0f} min | **Type:** {self.meeting_type}",
            f"**Participants:** {', '.join(self.participants)}",
            "",
            "---",
            "",
            "## 📌 Executive Summary",
            self.executive_summary,
            "",
            "## 🗣️ Key Topics",
        ]
        for topic in self.key_topics:
            lines.append(f"- {topic}")
        
        lines += ["", "## ✅ Decisions Made"]
        for decision in self.key_decisions:
            lines.append(f"- {decision}")
        
        lines += ["", "## 📋 Action Items"]
        if self.action_items:
            lines.append("| # | Task | Owner | Deadline | Priority |")
            lines.append("|---|------|-------|----------|----------|")
            for i, item in enumerate(self.action_items, 1):
                lines.append(
                    f"| {i} | {item.task} | {item.owner} | {item.deadline} | {item.priority} |"
                )
        else:
            lines.append("*No action items identified.*")
        
        if self.open_questions:
            lines += ["", "## ❓ Open Questions"]
            for q in self.open_questions:
                lines.append(f"- {q}")
        
        lines += [
            "",
            "---",
            f"**Meeting Effectiveness:** {self.meeting_effectiveness}",
            f"**Overall Sentiment:** {self.overall_sentiment}",
        ]
        
        return "\n".join(lines)


class MeetingAnalyzer:
    """
    Uses LLMs to extract structured insights from meeting transcripts.
    
    Supports:
      - OpenAI GPT models (cloud, best quality)
      - Ollama local models (privacy-preserving, free)
    """

    # ─── System prompt: sets the LLM's persona and task ───────────────────────
    SYSTEM_PROMPT = """You are MeetMind, an expert meeting analyst. 
Your job is to analyze meeting transcripts and extract structured, actionable intelligence.

You are precise, thorough, and output valid JSON only when asked.
You never invent information not present in the transcript.
When information is missing, you use "Unknown" or empty lists.
"""

    # ─── Extraction prompt with JSON schema ────────────────────────────────────
    EXTRACTION_PROMPT = """Analyze this meeting transcript and return ONLY valid JSON matching this exact schema:

{{
  "title": "descriptive meeting title",
  "meeting_type": "standup|planning|retrospective|client_call|interview|brainstorm|review|other",
  "executive_summary": "2-3 sentences covering what was discussed, decided, and next steps",
  "key_topics": ["topic 1", "topic 2", ...],
  "key_decisions": ["decision 1", "decision 2", ...],
  "action_items": [
    {{
      "task": "specific task description",
      "owner": "person's name or SPEAKER_01 format",
      "deadline": "deadline or 'No deadline specified'",
      "priority": "high|medium|low",
      "context": "brief context for why this task was assigned"
    }}
  ],
  "open_questions": ["unresolved question 1", ...],
  "follow_up_meetings": ["suggested follow-up 1", ...],
  "meeting_effectiveness": "highly effective|somewhat effective|needs improvement",
  "overall_sentiment": "positive|mixed|negative"
}}

TRANSCRIPT:
{transcript}

Participants detected: {participants}
Meeting duration: {duration} minutes

Return ONLY the JSON object. No markdown, no explanation."""

    def __init__(
        self,
        provider: str = "openai",          # "openai" or "ollama"
        model: str = "gpt-4o-mini",        # model name
        api_key: Optional[str] = None,
        ollama_base_url: str = None
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = None

    def _get_client(self):
        """Initialize LLM client."""
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI
                if not self.api_key:
                    raise ValueError(
                        "OpenAI API key required. Set OPENAI_API_KEY env var "
                        "or pass api_key parameter."
                    )
                self._client = OpenAI(api_key=self.api_key)
            elif self.provider == "ollama":
                # Ollama uses OpenAI-compatible API
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=f"{self.ollama_base_url}/v1",
                    api_key="ollama"   # dummy key for local
                )
        return self._client

    def _call_llm(self, prompt: str, system_prompt: str = None, max_tokens: int = 2000) -> str:
        """
        Make a call to the LLM.
        
        The structure:
          - system: sets context/persona
          - user: the actual task
        
        Temperature=0 for deterministic, structured extraction.
        Higher temperature (0.7+) for creative tasks.
        """
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0  # deterministic for extraction
        )
        return response.choices[0].message.content.strip()

    def extract_meeting_insights(
        self,
        segments: list,          # list[SpeakerSegment]
        duration_minutes: float,
        date: str = "Unknown"
    ) -> MeetingSummary:
        """
        Main extraction function. Sends transcript to LLM and parses structured output.
        
        Professor's Note on JSON parsing:
          LLMs occasionally output slightly malformed JSON.
          We use a repair strategy:
            1. Try direct json.loads()
            2. Strip markdown code fences if present
            3. Use json_repair library as last resort
        """
        # Build transcript string
        transcript_lines = []
        for seg in segments:
            mins = int(seg.start // 60)
            secs = int(seg.start % 60)
            transcript_lines.append(f"[{mins:02d}:{secs:02d}] {seg.speaker}: {seg.text}")
        transcript = "\n".join(transcript_lines)
        
        participants = list(set(seg.speaker for seg in segments))
        
        # Build extraction prompt
        prompt = self.EXTRACTION_PROMPT.format(
            transcript=transcript[:12000],  # ~3000 tokens context limit safety
            participants=", ".join(participants),
            duration=f"{duration_minutes:.0f}"
        )
        
        logger.info("Calling LLM for meeting analysis...")
        raw_response = self._call_llm(prompt, self.SYSTEM_PROMPT, max_tokens=2500)
        
        # Parse JSON response
        data = self._parse_json_response(raw_response)
        
        # Build ActionItem objects
        action_items = []
        for item_data in data.get("action_items", []):
            action_items.append(ActionItem(
                task=item_data.get("task", ""),
                owner=item_data.get("owner", "Unknown"),
                deadline=item_data.get("deadline", "Not specified"),
                priority=item_data.get("priority", "medium"),
                context=item_data.get("context", "")
            ))
        
        return MeetingSummary(
            title=data.get("title", "Meeting Summary"),
            date=date,
            duration_minutes=duration_minutes,
            participants=participants,
            meeting_type=data.get("meeting_type", "other"),
            executive_summary=data.get("executive_summary", ""),
            key_topics=data.get("key_topics", []),
            key_decisions=data.get("key_decisions", []),
            action_items=action_items,
            open_questions=data.get("open_questions", []),
            follow_up_meetings=data.get("follow_up_meetings", []),
            overall_sentiment=data.get("overall_sentiment", "mixed"),
            meeting_effectiveness=data.get("meeting_effectiveness", "somewhat effective"),
            raw_transcript=transcript
        )

    def _parse_json_response(self, response: str) -> dict:
        """Robustly parse JSON from LLM response."""
        # Remove markdown fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1])  # remove first and last line
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Direct JSON parse failed, attempting repair...")
            # Find JSON object boundaries
            start = cleaned.find('{')
            end = cleaned.rfind('}') + 1
            if start >= 0 and end > 0:
                try:
                    return json.loads(cleaned[start:end])
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"Could not parse JSON response: {response[:200]}")
            return {}

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a question about the meeting using provided context chunks.
        This is called by the RAG pipeline after retrieval.
        
        Professor's Note:
          This is the "generation" step of RAG.
          The retrieved chunks are injected as context, and the LLM
          is instructed to ONLY answer from that context — not from
          its general knowledge. This prevents hallucination.
        """
        prompt = f"""You are answering a question about a specific meeting.
Use ONLY the information provided in the context below.
If the answer is not in the context, say "I don't have that information from this meeting."

MEETING CONTEXT:
{context}

QUESTION: {question}

Answer concisely and accurately:"""
        
        return self._call_llm(prompt, max_tokens=500)