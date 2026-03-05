# MeetMind — Complete Walkthrough Guide

---

## 🗺️ Big Picture: What We're Building

```
┌─────────────────────────────────────────────────────────────────┐
│                        MEETMIND PIPELINE                         │
│                                                                   │
│  [Audio/Text] → [Whisper] → [Speaker Segments]                   │
│                                  ↓                               │
│                          [Emotion Analysis]  ← RoBERTa           │
│                                  ↓                               │
│                          [Meeting Analysis]  ← GPT-4o-mini       │
│                          (Summary + Actions)                      │
│                                  ↓                               │
│                     [ChromaDB Vector Index]  ← SentenceTransf.   │
│                                  ↓                               │
│                          [Gradio Chatbot]                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
meetmind/
├── src/
│   ├── transcriber.py      # Step 1: Audio → Text + Speaker Labels
│   ├── emotion_analyzer.py # Step 2: Text → Emotion Scores per Speaker
│   ├── meeting_analyzer.py # Step 3: LLM → Summary + Action Items (JSON)
│   ├── rag_chatbot.py      # Step 4: ChromaDB + Embeddings → Chat
│   ├── pipeline.py         # Step 5: Orchestrates all steps
│   └── app.py              # Step 6: Gradio UI
├── data/
│   └── chromadb/           # Vector database files (auto-created)
├── requirements.txt
├── Dockerfile
├── .env.example
└── WALKTHROUGH.md          ← you are here
```

---

## ⚙️ Setup — Step by Step

### 1. Clone and install

```bash
git clone https://github.com/yourname/meetmind
cd meetmind
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Get your OpenAI key at: https://platform.openai.com/api-keys

### 3. (Optional) Enable Speaker Diarization

Speaker diarization identifies *who* spoke when.
Without it, all speakers are labeled "SPEAKER_UNKNOWN".

To enable:
1. Create account at https://huggingface.co
2. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept the model license agreement
4. Generate a token at https://huggingface.co/settings/tokens
5. Add `HF_TOKEN=hf_xxxxx` to your `.env` file

### 4. Run the app

```bash
cd src
python app.py
# Open http://localhost:7860
```

---

## 🧠 Concept Deep Dives

### What is Whisper?

Whisper is a neural sequence-to-sequence model trained on 680,000 hours 
of multilingual audio. It uses an encoder-decoder Transformer:
- **Encoder**: processes the audio spectrogram
- **Decoder**: auto-regressively generates the transcript token by token

Why it's great: Zero-shot multilingual, highly accurate, free.

```python
# What's actually happening under the hood:
import whisper
model = whisper.load_model("base")
result = model.transcribe("meeting.mp3", word_timestamps=True)
# result["segments"] = [{text, start, end, words: [{word, start, end}]}]
```

### What is Speaker Diarization?

Diarization = "who spoke when?"

It uses a speaker embedding model (x-vector or d-vector) to 
compute a unique "voice fingerprint" for each person, then clusters 
audio segments by fingerprint similarity.

Result: [{speaker: "SPEAKER_01", start: 0.0, end: 3.5}, ...]

We then *align* this with Whisper's timestamps to assign speaker 
labels to each text segment.

### What is Sentence Embedding?

When you search "budget discussion", you want to find chunks that say 
"financial planning" or "cost analysis" — not just exact words.

Sentence transformers map text to dense vectors where:
- Similar meaning → similar vector direction (high cosine similarity)
- Different meaning → different vector direction (low cosine similarity)

The model all-MiniLM-L6-v2 produces 384-dimensional vectors.
It was trained with contrastive learning — similar sentences are pushed 
close together in vector space, dissimilar ones pushed apart.

### What is RAG?

RAG = Retrieval Augmented Generation. Three-phase process:

**Phase 1 — Indexing** (done once after meeting is processed):
- Chunk transcript into overlapping pieces (~300 chars)
- Embed each chunk with sentence-transformers
- Store in ChromaDB (vector database)

**Phase 2 — Retrieval** (done per query):
- Embed the user's question
- Find top-k chunks by cosine similarity
- Return the most semantically relevant passages

**Phase 3 — Generation** (done per query):
- Inject retrieved chunks as context into LLM prompt
- LLM answers using ONLY that context
- Prevents hallucination (LLM can't invent meeting content)

Why not just dump the whole transcript into the LLM?
- Context limits: GPT-4 can handle ~100k tokens, but long meetings exceed this
- Cost: more tokens = more money
- Quality: LLMs perform better with focused, relevant context
- Latency: smaller prompt = faster response

### What is Chain-of-Thought Prompting?

When extracting action items, complex reasoning helps.
Compare:

**Naive prompt**: "List action items from this transcript"
**CoT prompt**: "Read the transcript. First identify all commitments made. 
Then determine who was assigned each commitment. Then classify priority..."

The structured thinking leads to more accurate extraction.

---

## 🚀 Deployment

### Option A: Local (Development)

```bash
python src/app.py
```

### Option B: Docker

```bash
# Build image
docker build -t meetmind .

# Run with env vars
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-xxx \
  -e HF_TOKEN=hf-xxx \
  meetmind
```

### Option C: HuggingFace Spaces (Free!)

1. Create account at huggingface.co
2. Create new Space: Docker SDK
3. Push your code:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/meetmind
   git push hf main
   ```
4. Add secrets in Space Settings:
   - OPENAI_API_KEY
   - HF_TOKEN

Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/meetmind`

### Option D: Ollama (No API Key Needed!)

Run LLM locally — free, private, no internet needed.

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (Llama 3 is excellent for this task)
ollama pull llama3.1

# In your .env:
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1

python src/app.py
```

---

## 📊 Using the AMI Meeting Corpus (Dataset)

The AMI corpus has 100 hours of real meeting recordings with transcripts.

```bash
# Download a sample
wget https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip

# Or use HuggingFace datasets:
pip install datasets
```

```python
from datasets import load_dataset

# AMI corpus on HuggingFace
dataset = load_dataset("edinburghcstr/ami", "ihm", split="train")
sample = dataset[0]

# sample keys: 'meeting_id', 'audio', 'text', 'speaker_id', 'begin_time', 'end_time'
print(sample['text'])
print(sample['speaker_id'])
```

---

## 🎯 Interview Talking Points

When presenting this project:

1. **"I built an end-to-end NLP pipeline"** — covers data ingestion, processing, storage, retrieval, and generation

2. **"I implemented RAG from scratch"** — explain the chunking strategy, why you chose cosine similarity, what top-k means

3. **"I combined multiple AI models"** — Whisper (ASR), RoBERTa (classification), GPT (generation), SentenceTransformers (embeddings)

4. **"I thought about real production concerns"** — speaker diarization, JSON schema for reliable LLM output, error handling, lazy loading, Docker deployment

5. **"I understand the tradeoffs"** — Whisper model size tradeoff, why RAG over full-context, why emotion analysis adds value vs just sentiment

---

## 🐛 Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| `ffmpeg not found` | Missing system dep | `brew install ffmpeg` or `apt install ffmpeg` |
| Diarization fails | No HF token | Add HF_TOKEN to .env, accept license |
| Empty transcript | Audio too quiet | Check audio quality, try larger Whisper model |
| LLM returns invalid JSON | Model inconsistency | The parser handles this — retry if persistent |
| ChromaDB error | Corrupt DB | Delete `data/chromadb/` folder and re-index |
| GPU OOM | Model too large | Use `WHISPER_MODEL=tiny` or add batch_size limits |

## ⚠️ Disclaimer

This project was built as a **learning exercise** to explore AI/ML concepts including 
RAG pipelines, speech recognition, emotion analysis, and LLM integration. 
It is not production-ready and may contain bugs or incomplete features. 
Contributions, suggestions, and feedback are welcome!

Built with curiosity and lots of debugging. 🛠️
