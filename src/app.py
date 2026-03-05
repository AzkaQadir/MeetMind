"""
STEP 6: GRADIO UI
━━━━━━━━━━━━━━━━━
Professor's Note:
  Gradio turns Python functions into interactive web interfaces.
  It's the fastest way to demo ML models — no frontend skills needed.

  Our UI has two tabs:
    Tab 1: Upload & Process — upload audio/transcript, run pipeline, view results
    Tab 2: Chat — ask questions about the meeting

  Gradio State:
    Gradio's gr.State() stores Python objects across interactions.
    We store our MeetMindState so the chatbot can access results
    from the analysis tab without re-running everything.
"""

import os
import sys
import tempfile
import gradio as gr
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# ─── Global Pipeline (loaded once at startup) ─────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline import MeetMindPipeline
        _pipeline = MeetMindPipeline(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            hf_token=os.getenv("HF_TOKEN"),
            whisper_model=os.getenv("WHISPER_MODEL", "base"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            use_diarization=os.getenv("USE_DIARIZATION", "true").lower() == "true"
        )
    return _pipeline


# ─── Processing Function ───────────────────────────────────────────────────────
def process_meeting(audio_file, transcript_text, progress=gr.Progress()):
    """
    Main processing handler — called when user clicks "Analyze Meeting".
    
    Gradio's gr.Progress() automatically shows a progress bar in the UI.
    We yield intermediate results so the UI updates in real-time.
    """
    pipeline = get_pipeline()
    
    # Determine input source
    source_path = None
    if audio_file is not None:
        source_path = audio_file
    elif transcript_text and transcript_text.strip():
        # Write text to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(transcript_text)
            source_path = f.name
    else:
        return (
            "❌ Please upload an audio file or paste a transcript.",
            "", "", "", None
        )

    # Progress tracking
    progress_messages = []
    
    def on_progress(step, msg):
        progress_messages.append(f"**{step}:** {msg}")
    
    # Run pipeline
    progress(0, desc="Starting analysis...")
    state = pipeline.process(source_path, progress_callback=on_progress)
    
    if state.error:
        return (
            f"❌ Error: {state.error}",
            "", "", "", state
        )

    # Format outputs
    progress(0.9, desc="Formatting results...")
    
    # 1. Transcript
    transcript_lines = []
    for seg in state.segments:
        mins = int(seg.start // 60)
        secs = int(seg.start % 60)
        transcript_lines.append(f"**[{mins:02d}:{secs:02d}] {seg.speaker}:** {seg.text}")
    transcript_output = "\n\n".join(transcript_lines)

    # 2. Summary
    summary_md = state.summary.to_markdown() if state.summary else "No summary generated."

    # 3. Emotion Report
    emotion_output = state.emotion_report or "No emotion analysis available."

    # 4. Status
    n_actions = len(state.summary.action_items) if state.summary else 0
    n_speakers = len(set(s.speaker for s in state.segments))
    status = (
        f"✅ **Analysis Complete!** "
        f"Processed {len(state.segments)} segments • "
        f"{n_speakers} speakers • "
        f"{n_actions} action items • "
        f"{state.processing_time:.1f}s"
    )

    progress(1.0, desc="Done!")
    return status, transcript_output, summary_md, emotion_output, state


# ─── Chat Function ─────────────────────────────────────────────────────────────
def chat_with_meeting(message, history, state):
    """
    Handle a chatbot message.
    
    Args:
        message: user's current question
        history: list of [user_msg, bot_msg] pairs (Gradio format)
        state: MeetMindState from processing tab
    """
    if state is None:
        return history + [[message, "⚠️ Please analyze a meeting first in the **Upload** tab."]], state
    
    if not state.is_indexed:
        return history + [[message, "⚠️ Meeting processing failed or is incomplete."]], state
    
    pipeline = get_pipeline()
    answer, retrieved_chunks = pipeline.chat(message, state)
    
    # Add source citations
    if retrieved_chunks:
        top_chunk = retrieved_chunks[0]
        citation = f"\n\n*💡 Based on {top_chunk.speaker}'s statement at {top_chunk.timestamp}*"
        answer += citation
    
    history.append([message, answer])
    return history, state


def reset_chat_fn(state):
    """Reset the conversation history."""
    if state:
        get_pipeline().reset_chat()
    return [], state


# ─── Build Gradio Interface ────────────────────────────────────────────────────
def build_ui():
    
    with gr.Blocks(
        title="MeetMind — AI Meeting Intelligence",
    ) as demo:
        
        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="meeting-header">
            <h1>🧠 MeetMind</h1>
            <p style="color: #666; font-size: 18px;">AI-Powered Meeting Intelligence System</p>
            <p style="color: #999">Upload a recording or transcript → Get summaries, action items, speaker analysis & a chatbot</p>
        </div>
        """)
        
        # ── Shared state between tabs ────────────────────────────────────────
        meeting_state = gr.State(None)

        # ════════════════════════════════════════════════════════════════════
        # TAB 1: Upload & Analyze
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("📤 Upload & Analyze"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    
                    audio_input = gr.Audio(
                        label="🎙️ Upload Audio Recording",
                        type="filepath",
                        sources=["upload"]
                    )
                    
                    gr.Markdown("**OR paste a transcript below:**")
                    transcript_input = gr.Textbox(
                        label="📝 Text Transcript",
                        placeholder="""[00:00] SPEAKER_01: Let's get started. Today we're reviewing Q3 results.
[00:15] SPEAKER_02: The revenue numbers look strong, up 23% from last quarter.
[00:30] SPEAKER_01: Great. John, can you take ownership of the follow-up report?
[00:45] SPEAKER_03: Sure, I'll have it ready by Friday.""",
                        lines=8,
                        max_lines=20
                    )
                    
                    analyze_btn = gr.Button(
                        "🚀 Analyze Meeting",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=2):
                    status_output = gr.Markdown(
                        value="*Upload a file and click Analyze to begin...*",
                        elem_id="status-box"
                    )

            # ── Results Tabs ──────────────────────────────────────────────────
            with gr.Tabs():
                with gr.Tab("📋 Summary & Action Items"):
                    summary_output = gr.Markdown(label="Meeting Summary")
                
                with gr.Tab("🗣️ Speaker Transcript"):
                    transcript_output = gr.Markdown(label="Transcript")
                
                with gr.Tab("🎭 Emotion Analysis"):
                    emotion_output = gr.Markdown(label="Emotion Report")

            analyze_btn.click(
                fn=process_meeting,
                inputs=[audio_input, transcript_input],
                outputs=[status_output, transcript_output, summary_output, 
                         emotion_output, meeting_state],
                show_progress=True
            )

        # ════════════════════════════════════════════════════════════════════
        # TAB 2: Chatbot
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("💬 Ask About the Meeting"):
            gr.Markdown("""
            ### RAG-Powered Meeting Chatbot
            Ask questions about the meeting in plain English.
            The system retrieves relevant transcript sections and answers with evidence.
            
            **Try asking:**
            - *"What were the main decisions made?"*
            - *"Who is responsible for the Q3 report?"*
            - *"What did we decide about the budget?"*
            - *"Were there any unresolved issues?"*
            - *"Summarize what Speaker_02 said"*
            """)
            
            chatbot = gr.Chatbot(
                label="MeetMind Chat",
                height=450
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about the meeting...",
                    label="",
                    scale=5,
                    submit_btn="Send"
                )
                reset_btn = gr.Button("🔄 Reset Chat", scale=1, variant="secondary")

            msg_input.submit(
                fn=chat_with_meeting,
                inputs=[msg_input, chatbot, meeting_state],
                outputs=[chatbot, meeting_state]
            ).then(lambda: "", outputs=msg_input)

            reset_btn.click(
                fn=reset_chat_fn,
                inputs=[meeting_state],
                outputs=[chatbot, meeting_state]
            )

        # ════════════════════════════════════════════════════════════════════
        # TAB 3: Help & Info
        # ════════════════════════════════════════════════════════════════════
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## How MeetMind Works
            
            ### The Pipeline
            ```
            Audio/Text Input
                  ↓
            [Whisper] Speech-to-Text transcription
                  ↓
            [Pyannote] Speaker Diarization (who spoke when)
                  ↓  
            [RoBERTa] Emotion Analysis per speaker
                  ↓
            [GPT-4o-mini] Summary + Action Item Extraction
                  ↓
            [ChromaDB + SentenceTransformers] RAG Index
                  ↓
            Gradio Interface — Chat, Summaries, Reports
            ```
            
            ### Supported Input Formats
            - **Audio:** MP3, MP4, M4A, WAV, OGG, FLAC, WEBM
            - **Text:** Plain text transcripts with format:
              `[HH:MM:SS] SPEAKER: text` or `SPEAKER: text`
            
            ### Environment Variables
            | Variable | Description | Default |
            |----------|-------------|---------|
            | `OPENAI_API_KEY` | OpenAI API key | Required |
            | `HF_TOKEN` | HuggingFace token (for diarization) | Optional |
            | `WHISPER_MODEL` | Whisper model size | `base` |
            | `LLM_MODEL` | LLM model to use | `gpt-4o-mini` |
            | `USE_DIARIZATION` | Enable speaker ID | `true` |
            
            ### Whisper Model Sizes
            | Model | Size | Speed | Quality |
            |-------|------|-------|---------|
            | tiny | 39MB | ~10x | Basic |
            | base | 74MB | ~5x | Good |
            | small | 244MB | ~2x | Better |
            | medium | 769MB | ~1x | Great |
            | large | 1550MB | ~0.5x | Best |
            """)

    return demo


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        show_error=True,
        theme=gr.themes.Soft(primary_hue="blue")
    )