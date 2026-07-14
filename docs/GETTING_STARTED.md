# Getting started

From a fresh clone to chatting with a synthesized persona. The repo ships three
synthetic interview transcripts, so you can run the whole pipeline before you
have any research data of your own.

---

## 1. Install

Python 3.11 or newer.

```bash
git clone https://github.com/meghanaRangarajan/OmniVera.git
cd OmniVera

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

## 2. Add your keys

```bash
cp .env.example .env
```

Then open `.env` and fill in:

| Key | Needed for | Where to get it |
|---|---|---|
| `ANTHROPIC_API_KEY` | Synthesis and chat | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| `OPENAI_API_KEY` | Embeddings only | [platform.openai.com](https://platform.openai.com/api-keys) |
| `REDDIT_*` | Optional — Reddit ingestion | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) |

Both required keys are pay-as-you-go. A full run over the three sample
transcripts costs well under a dollar.

## 3. Run the pipeline

```bash
python scripts/load_samples.py       # parse the 3 sample interviews   (no API calls)
python scripts/build_index.py        # chunk, embed, index into ChromaDB
python scripts/run_pipeline.py       # synthesize the ICP with citations
python scripts/run_chat_server.py    # chat with the personas at localhost:5000
```

Each step writes to disk before the next one reads, so you can stop and resume
anywhere. `build_index.py` prints chunk stats and runs a few proof-of-retrieval
queries, so you'll see whether the index is sane before spending money on
synthesis.

To run all four as a single agentic loop instead:

```bash
python scripts/run_agent.py
```

---

## Using your own data

### Interview transcripts

This is the important one, because **the parser is strict** and a transcript it
can't read produces zero chunks rather than an error.

Supported formats: `.txt` `.md` `.pdf` `.docx` `.vtt` `.srt`

A transcript must look like this:

```
In-Depth Interview – Priya Raman

Age: 23
Generation: Gen Z
Location: Portland, OR
Occupation: Junior UX Designer

1. BACKGROUND

INTERVIEWER: Tell me about your weekends.
PRIYA: Trying to get out of the city, mostly...

2. PAINS

INTERVIEWER: What frustrates you most?
PRIYA: That the cheap ones are cheap in the ways that matter...
```

Three rules the parser enforces:

1. **Header line** — must start with `In-Depth Interview` (optionally
   `In-Depth Qualitative Interview` or `... Interview Transcript`), followed by
   a dash and the interviewee's name. This is what names the transcript. If a
   single file contains two or more of these headers, it's split into separate
   interviews automatically.
2. **Metadata block** — `Key: Value` lines before the first section. `Age`,
   `Generation`, `Location`, `Occupation`, `Device`, and `Family` are carried
   onto every chunk as metadata, which is what makes filtered retrieval
   (`{"generation": "Gen Z"}`) work.
3. **Section headings** — numbered and in caps: `1. BACKGROUND`, `2. PAINS`.
   Each speaker turn is tagged with the section it falls under, and the chunker
   refuses to split across a section boundary.
4. **Speaker turns** — `SPEAKER: text`, one per line.

The three files in `samples/transcripts/` are working examples. Copy one and
replace the content.

Once your transcripts are ready, either upload them through the intake form:

```bash
python scripts/run_web_intake.py     # opens a form at localhost:5000
```

…or drop them in a folder and point `load_samples.py` at it. `build_index.py`
will find the output of either path automatically.

### Reddit threads

`scripts/ingest_reddit.py` reads a CSV from
`data/raw/test_reddit_data/Raw_Reddit_Data.csv` with three required columns:

```csv
subreddit,username,thread
Ultralight,trailmix_pdx,"Anyone else priced out of GPS watches? ..."
```

A sample lives at `samples/reddit/sample_reddit_data.csv`. To use it:

```bash
mkdir -p data/raw/test_reddit_data
cp samples/reddit/sample_reddit_data.csv data/raw/test_reddit_data/Raw_Reddit_Data.csv
python scripts/ingest_reddit.py
```

Reddit chunks are tagged `trust_weight=0.6` — they corroborate and supply
vocabulary, but they never outweigh a customer interview.

### Deep-research PDFs

`scripts/ingest_deep_research.py` reads every PDF in
`data/raw/test_deep_research_report/`, splitting on `##` markdown-style section
headers. Tagged `trust_weight=0.8` — analyst framing, weighted below customer
voice but above public chatter.

A sample report lives at `samples/deep_research/deep_research_report.pdf`. To
ingest it:

```bash
mkdir -p data/raw/test_deep_research_report
cp samples/deep_research/deep_research_report.pdf data/raw/test_deep_research_report/
python scripts/ingest_deep_research.py
```

**The `##` headers matter.** The ingester splits the extracted text on lines
beginning with `## `, and everything before the first one is swept into a single
leading section. A PDF exported from a word processor that styles headings
visually — bold, larger font — but drops the literal `##` characters will parse
as *one* giant section. If you're producing your own report, keep the markers in
the text.

The markdown source for the sample is committed alongside the PDF
(`deep_research_report.md`), and `scripts/build_sample_pdf.py` regenerates the
PDF from it. That's a working reference for the format.

Both ingestion scripts write straight into ChromaDB and deduplicate against
what's already indexed, so re-running them is safe. Note that they upsert into
the index that `build_index.py` creates — run `build_index.py` first.

---

## Troubleshooting

**`No parsed transcripts found`** — you skipped step 3.1. Run
`python scripts/load_samples.py` (or the intake form) before `build_index.py`.

**Zero chunks, no error** — your transcript didn't match the header/section
format above. Compare against `samples/transcripts/interview_01_priya.md`. The
most common miss is a header line that doesn't start with `In-Depth Interview`.

**`OPENAI_API_KEY` errors during `build_index.py`** — the embedding step is the
first thing that spends money. Check `.env` exists at the project root and the
key is on the `OPENAI_API_KEY=` line with no quotes.

**Re-running `build_index.py` doesn't pick up changes** — it deletes and rebuilds
the ChromaDB store from scratch each time, so this shouldn't happen. If it does,
delete `data/processed/chroma/` by hand and run it again.

---

## What's where

```
samples/
  transcripts/     3 synthetic interviews — the primary evidence lane (trust 1.0)
  reddit/          Sample CSV — the corroboration lane (trust 0.6)
  deep_research/   Sample market report, .md source + rendered .pdf (trust 0.8)
scripts/           CLI entry points, one per pipeline stage
src/icp_agent/     The library: parsing, RAG, synthesis, chat
src/icp_agent/agent/   The orchestrator's tool-use loop
data/              All generated artifacts. Gitignored. Yours never leaves your machine.
```

Architecture and design rationale: [ARCHITECTURE.md](ARCHITECTURE.md).
