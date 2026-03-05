"""
STEP 1: TRANSCRIPTION ENGINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Whisper is OpenAI's speech-to-text model. It outputs raw text, but we 
  need speaker labels too. For that we use pyannote.audio for "diarization"
  — the process of answering "who spoke when?" Then we merge the two outputs.

  Pipeline:
    Audio File → Whisper → Raw Transcript
    Audio File → Pyannote → Speaker Segments [{speaker, start, end}]
    Merge → [{speaker, text, start, end}]
"""

import os
import json
import whisper
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Represents one spoken turn in the meeting."""
    speaker: str
    text: str
    start: float   # seconds
    end: float     # seconds
    confidence: float = 1.0

    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)


class MeetingTranscriber:
    """
    Handles audio → structured transcript conversion.
    
    Why two models?
    - Whisper: WHAT was said (text)
    - Pyannote: WHO said it (speaker labels)
    We align them by timestamp overlap.
    """

    def __init__(
        self,
        whisper_model: str = "base",       # tiny/base/small/medium/large — tradeoff: speed vs accuracy
        use_diarization: bool = True,
        hf_token: Optional[str] = None,    # needed for pyannote (gated model)
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load Whisper
        logger.info(f"Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)

        # Load diarization pipeline (optional — needs HuggingFace token)
        self.diarization_pipeline = None
        if use_diarization:
            self._load_diarizer(hf_token)

    def _load_diarizer(self, hf_token: Optional[str]):
        """
        Pyannote requires accepting a model license on HuggingFace.
        Steps:
          1. Create account at huggingface.co
          2. Accept license at hf.co/pyannote/speaker-diarization-3.1
          3. Generate token at hf.co/settings/tokens
          4. Pass token here or set HF_TOKEN env var
        """
        try:
            from pyannote.audio import Pipeline
            token = hf_token or os.getenv("HF_TOKEN")
            if not token:
                logger.warning(
                    "No HuggingFace token found. Diarization disabled.\n"
                    "Set HF_TOKEN env var or pass hf_token parameter.\n"
                    "Speakers will be labeled by segment instead."
                )
                return
            
            logger.info("Loading speaker diarization model...")
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token
            ).to(torch.device(self.device))
            logger.info("Diarization model loaded!")
        except ImportError:
            logger.warning("pyannote.audio not installed. Diarization disabled.")
        except Exception as e:
            logger.warning(f"Could not load diarization: {e}")

    def transcribe_audio(self, audio_path: str) -> dict:
        """
        Run Whisper transcription.
        
        Returns Whisper's full output including word-level timestamps.
        We use word_timestamps=True for precise speaker alignment.
        """
        logger.info(f"Transcribing: {audio_path}")
        result = self.whisper_model.transcribe(
            audio_path,
            word_timestamps=True,    # crucial for speaker alignment
            verbose=False
        )
        logger.info(f"Transcription complete. {len(result['segments'])} segments.")
        return result

    def diarize_audio(self, audio_path: str) -> list[dict]:
        """
        Run speaker diarization.
        Returns list of: {speaker: "SPEAKER_01", start: 0.0, end: 3.5}
        
        Pyannote uses a neural network trained to distinguish voice characteristics.
        """
        if not self.diarization_pipeline:
            return []
        
        logger.info("Running speaker diarization...")
        diarization = self.diarization_pipeline(audio_path)
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end
            })
        logger.info(f"Found {len(set(s['speaker'] for s in segments))} unique speakers.")
        return segments

    def _assign_speaker(self, segment_start: float, segment_end: float,
                        diar_segments: list[dict]) -> str:
        """
        For a given Whisper text segment, find the most overlapping diarization speaker.
        
        Algorithm: For each diarization segment, compute overlap with the text segment.
        Assign whichever speaker has the most overlap time.
        """
        if not diar_segments:
            return "SPEAKER_UNKNOWN"

        overlap_scores = {}
        for d in diar_segments:
            # Overlap = min(ends) - max(starts) — classic interval overlap formula
            overlap = min(segment_end, d["end"]) - max(segment_start, d["start"])
            if overlap > 0:
                spk = d["speaker"]
                overlap_scores[spk] = overlap_scores.get(spk, 0) + overlap

        if not overlap_scores:
            return "SPEAKER_UNKNOWN"
        return max(overlap_scores, key=overlap_scores.get)

    def merge_transcript_and_diarization(
        self,
        whisper_result: dict,
        diar_segments: list[dict]
    ) -> list[SpeakerSegment]:
        """
        Combine Whisper text with pyannote speaker labels.
        
        For each Whisper segment (a chunk of text with timestamps),
        we find which speaker was talking during that time.
        Then we merge consecutive segments from the same speaker.
        """
        merged = []
        for seg in whisper_result["segments"]:
            speaker = self._assign_speaker(seg["start"], seg["end"], diar_segments)
            
            # If last segment is same speaker, extend it (avoid fragmentation)
            if merged and merged[-1].speaker == speaker:
                merged[-1].text += " " + seg["text"].strip()
                merged[-1].end = seg["end"]
            else:
                merged.append(SpeakerSegment(
                    speaker=speaker,
                    text=seg["text"].strip(),
                    start=seg["start"],
                    end=seg["end"]
                ))
        return merged

    def transcribe(self, audio_path: str) -> list[SpeakerSegment]:
        """
        Main entry point. Returns structured transcript with speaker labels.
        
        Usage:
            transcriber = MeetingTranscriber(whisper_model="base")
            segments = transcriber.transcribe("meeting.mp3")
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        whisper_result = self.transcribe_audio(audio_path)
        diar_segments = self.diarize_audio(audio_path)
        segments = self.merge_transcript_and_diarization(whisper_result, diar_segments)
        return segments

    def transcribe_text(self, raw_text: str) -> list[SpeakerSegment]:
        """
        If user uploads a plain transcript (no audio), parse it.
        
        Expected format (common meeting transcript format):
            [00:01:23] John: Let's discuss the Q3 results.
            [00:01:45] Sarah: I think we need to focus on growth.
        
        Fallback: treat as single unknown speaker.
        """
        import re
        segments = []
        
        # Try parsing timestamped format: [HH:MM:SS] Speaker: text
        pattern = r'\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s+([^:]+):\s+(.+)'
        lines = raw_text.strip().split('\n')
        
        def time_to_seconds(t: str) -> float:
            parts = list(map(float, t.split(':')))
            if len(parts) == 2:
                return parts[0] * 60 + parts[1]
            return parts[0] * 3600 + parts[1] * 60 + parts[2]

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            match = re.match(pattern, line)
            if match:
                time_str, speaker, text = match.groups()
                start = time_to_seconds(time_str)
                # Estimate end from next segment or +30s
                end = start + 30
                segments.append(SpeakerSegment(
                    speaker=speaker.strip().upper().replace(' ', '_'),
                    text=text.strip(),
                    start=start,
                    end=end
                ))
            elif ':' in line:
                # Simple "Speaker: text" format without timestamps
                speaker, _, text = line.partition(':')
                segments.append(SpeakerSegment(
                    speaker=speaker.strip().upper().replace(' ', '_'),
                    text=text.strip(),
                    start=float(i * 30),
                    end=float((i + 1) * 30)
                ))
            else:
                # Raw text — single speaker
                segments.append(SpeakerSegment(
                    speaker="SPEAKER_01",
                    text=line,
                    start=float(i * 30),
                    end=float((i + 1) * 30)
                ))

        return segments

    def format_transcript(self, segments: list[SpeakerSegment]) -> str:
        """Convert segments to readable text format for downstream processing."""
        lines = []
        for seg in segments:
            mins = int(seg.start // 60)
            secs = int(seg.start % 60)
            lines.append(f"[{mins:02d}:{secs:02d}] {seg.speaker}: {seg.text}")
        return "\n".join(lines)

    def save_transcript(self, segments: list[SpeakerSegment], output_path: str):
        """Save structured transcript as JSON."""
        data = {
            "segments": [s.to_dict() for s in segments],
            "speakers": list(set(s.speaker for s in segments)),
            "total_duration": max(s.end for s in segments) if segments else 0,
            "total_segments": len(segments)
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved transcript to {output_path}")
