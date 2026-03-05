"""
STEP 2: EMOTION & SENTIMENT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  We use a pre-trained RoBERTa model fine-tuned on emotion detection.
  This is a classic text classification task.

  Model: j-hartmann/emotion-english-distilroberta-base
    - Input: text string
    - Output: probability scores for 7 emotions:
      anger, disgust, fear, joy, neutral, sadness, surprise

  For meetings, we care about:
    - Tone per speaker (is Speaker A consistently frustrated?)
    - Meeting-wide sentiment arc (did energy improve or drop?)
    - Flagging high-emotion moments (potential conflict or breakthrough)

  Why RoBERTa over BERT?
    RoBERTa removed NSP pre-training and uses dynamic masking.
    This made it significantly better at downstream NLP tasks.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Emotion analysis for a single speaker segment."""
    dominant_emotion: str
    scores: dict[str, float]          # all emotion probabilities
    sentiment: str                     # "positive", "negative", "neutral"
    sentiment_score: float             # -1.0 to 1.0
    text_snippet: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SpeakerEmotionProfile:
    """Aggregated emotion profile for one speaker across the meeting."""
    speaker: str
    dominant_emotion: str
    avg_sentiment: float
    emotion_distribution: dict[str, float]
    sentiment_trend: str               # "improving", "declining", "stable"
    flagged_moments: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Emotion → Sentiment mapping
EMOTION_SENTIMENT = {
    "joy": ("positive", 1.0),
    "surprise": ("positive", 0.4),
    "neutral": ("neutral", 0.0),
    "fear": ("negative", -0.6),
    "sadness": ("negative", -0.7),
    "disgust": ("negative", -0.8),
    "anger": ("negative", -0.9),
}


class EmotionAnalyzer:
    """
    Analyzes emotion and sentiment per speaker segment.
    
    Two-level analysis:
      1. Segment-level: emotion per utterance
      2. Speaker-level: aggregated profile across the meeting
    """

    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        """
        Lazy-load the model (only loads when first called).
        This keeps app startup fast.
        """
        self.model_name = model_name
        self._pipeline = None

    def _get_pipeline(self):
        """Load model on first use (lazy loading pattern)."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading emotion model: {self.model_name}")
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    top_k=None,        # return ALL emotion scores, not just top-1
                    truncation=True,
                    max_length=512
                )
                logger.info("Emotion model loaded!")
            except Exception as e:
                logger.error(f"Failed to load emotion model: {e}")
                raise
        return self._pipeline

    def analyze_segment(self, text: str) -> EmotionResult:
        """
        Analyze emotion in a single text segment.
        
        The model returns probabilities for each emotion class.
        We pick the highest as dominant, then map to sentiment polarity.
        """
        if not text or len(text.strip()) < 3:
            return EmotionResult(
                dominant_emotion="neutral",
                scores={"neutral": 1.0},
                sentiment="neutral",
                sentiment_score=0.0,
                text_snippet=text[:100]
            )

        try:
            pipe = self._get_pipeline()
            # model returns list of [{label: "joy", score: 0.85}, ...]
            raw_results = pipe(text[:512])[0]
            
            scores = {r['label']: round(r['score'], 4) for r in raw_results}
            dominant = max(scores, key=scores.get)
            sentiment, polarity = EMOTION_SENTIMENT.get(dominant, ("neutral", 0.0))

            # Compute weighted sentiment score across all emotions
            weighted_score = sum(
                EMOTION_SENTIMENT.get(emotion, ("neutral", 0.0))[1] * score
                for emotion, score in scores.items()
            )

            return EmotionResult(
                dominant_emotion=dominant,
                scores=scores,
                sentiment=sentiment,
                sentiment_score=round(weighted_score, 4),
                text_snippet=text[:100] + "..." if len(text) > 100 else text
            )
        except Exception as e:
            logger.warning(f"Emotion analysis failed for segment: {e}")
            return EmotionResult(
                dominant_emotion="neutral",
                scores={"neutral": 1.0},
                sentiment="neutral",
                sentiment_score=0.0
            )

    def analyze_transcript(
        self,
        segments: list,   # list of SpeakerSegment objects
        batch_size: int = 16
    ) -> dict[str, list[EmotionResult]]:
        """
        Analyze all segments. Returns dict: {speaker: [EmotionResult, ...]}
        
        We process in batches for efficiency. 
        On GPU, this is 10-50x faster than one-at-a-time inference.
        """
        results: dict[str, list[EmotionResult]] = {}
        
        logger.info(f"Analyzing emotions for {len(segments)} segments...")
        for seg in segments:
            emotion = self.analyze_segment(seg.text)
            emotion.text_snippet = seg.text[:100]
            
            if seg.speaker not in results:
                results[seg.speaker] = []
            results[seg.speaker].append(emotion)

        return results

    def build_speaker_profiles(
        self,
        segments: list,
        emotion_results: dict[str, list[EmotionResult]]
    ) -> list[SpeakerEmotionProfile]:
        """
        Aggregate per-segment emotions into per-speaker profiles.
        
        This is where we build the "meeting intelligence":
        - Who was engaged/positive vs disengaged/negative?
        - Was there a point where someone's tone shifted?
        - Were there high-tension moments?
        """
        profiles = []
        
        for speaker, emotions in emotion_results.items():
            if not emotions:
                continue

            # Aggregate emotion distribution
            emotion_dist: dict[str, float] = {}
            for e in emotions:
                for emotion, score in e.scores.items():
                    emotion_dist[emotion] = emotion_dist.get(emotion, 0) + score
            
            # Normalize to percentages
            total = sum(emotion_dist.values())
            emotion_dist = {k: round(v / total * 100, 1) for k, v in emotion_dist.items()}
            
            # Dominant emotion overall
            dominant = max(emotion_dist, key=emotion_dist.get)
            
            # Average sentiment
            avg_sentiment = np.mean([e.sentiment_score for e in emotions])

            # Trend: compare first half vs second half of speaker's turns
            mid = len(emotions) // 2
            if mid > 0:
                first_half_avg = np.mean([e.sentiment_score for e in emotions[:mid]])
                second_half_avg = np.mean([e.sentiment_score for e in emotions[mid:]])
                diff = second_half_avg - first_half_avg
                if diff > 0.1:
                    trend = "improving 📈"
                elif diff < -0.1:
                    trend = "declining 📉"
                else:
                    trend = "stable ➡️"
            else:
                trend = "stable ➡️"

            # Flag high-negative moments (anger/disgust/fear > 0.5)
            flagged = []
            for i, (seg, emotion) in enumerate(zip(
                [s for s in segments if s.speaker == speaker], emotions
            )):
                if emotion.dominant_emotion in ("anger", "disgust") and \
                   emotion.scores.get(emotion.dominant_emotion, 0) > 0.5:
                    flagged.append({
                        "timestamp": f"{int(seg.start // 60):02d}:{int(seg.start % 60):02d}",
                        "emotion": emotion.dominant_emotion,
                        "score": emotion.scores[emotion.dominant_emotion],
                        "text": seg.text[:80]
                    })

            profiles.append(SpeakerEmotionProfile(
                speaker=speaker,
                dominant_emotion=dominant,
                avg_sentiment=round(float(avg_sentiment), 4),
                emotion_distribution=emotion_dist,
                sentiment_trend=trend,
                flagged_moments=flagged
            ))

        return profiles

    def generate_emotion_report(self, profiles: list[SpeakerEmotionProfile]) -> str:
        """
        Human-readable summary of emotion analysis results.
        This goes into the final meeting summary and the RAG chatbot context.
        """
        lines = ["## 🎭 Emotional Tone Analysis\n"]
        
        for profile in profiles:
            sentiment_emoji = "😊" if profile.avg_sentiment > 0.2 else \
                             "😐" if profile.avg_sentiment > -0.2 else "😟"
            
            lines.append(f"### {profile.speaker} {sentiment_emoji}")
            lines.append(f"- **Dominant emotion:** {profile.dominant_emotion.title()}")
            lines.append(f"- **Average sentiment:** {profile.avg_sentiment:+.2f} ({profile.sentiment_trend})")
            
            # Show top 3 emotions
            top_emotions = sorted(
                profile.emotion_distribution.items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            emotion_str = ", ".join(f"{e.title()}: {v:.0f}%" for e, v in top_emotions)
            lines.append(f"- **Emotion mix:** {emotion_str}")
            
            if profile.flagged_moments:
                lines.append(f"- ⚠️ **High-tension moments:** {len(profile.flagged_moments)} detected")
                for moment in profile.flagged_moments[:2]:  # show max 2
                    lines.append(f"  - [{moment['timestamp']}] {moment['text'][:60]}...")
            lines.append("")

        return "\n".join(lines)
