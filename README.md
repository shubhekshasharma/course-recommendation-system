# Course Recommendation System

An app that recommends university courses based on user interests and preferred workload, using ML clustering and a LLM-backed reasoning and recommendation.

## Requirements

- Python 3.9

## Setup

**1. Create and activate a virtual environment**

```bash
python3.9 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure secrets**

You need two secrets: `LLM_LITE_TOKEN` and `LLM_LITE_URL`.

**Option A — Streamlit secrets (recommended)**

Create the file `.streamlit/secrets.toml`:

```toml
LLM_LITE_TOKEN = "your-token"
LLM_LITE_URL = "your-litellm-base-url"
```

**Option B — Environment variables**

```bash
export LLM_LITE_TOKEN="your-token"
export LLM_LITE_URL="your-litellm-base-url"
```

> Note: `.streamlit/secrets.toml` is gitignored and should never be committed.

## Run

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`.
