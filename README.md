# Course Recommendation System

An app that recommends university courses based on user interests and preferred workload, using ML clustering and a LLM-backed reasoning and recommendation.

---

## Project Structure

```
├── app.py                     # Streamlit app
├── recommendations/           # ML + LLM logic (shared)
├── pickles/                   # Trained ML model files
├── courses_with_cluster.csv   # Course dataset
├── requirements.txt           # Streamlit dependencies
├── api/
│   └── recommend.py           # Vercel Python serverless function
│   └── requirements.txt       # Serverless dependencies
└── web/                       # Next.js frontend
```

---

## Streamlit App

### Requirements

- Python 3.9

### Setup

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

### Run

```bash
streamlit run app.py
```

---

## Next.js Web App

### Requirements

- Node.js 18+

### Setup

```bash
cd web
npm install
```

### Run (frontend only)

```bash
cd web
npm run dev
```

> Note: API calls to `/api/recommend` will not work in this mode. Use `vercel dev` for full-stack local development.

### Full-stack local development

Run the Python API and Next.js frontend in two separate terminals.

**Terminal 1 — Python API** (from repo root, venv active):

```bash
export LLM_LITE_TOKEN="your-token"
export LLM_LITE_URL="your-litellm-base-url"
cd api && python local_server.py
```

**Terminal 2 — Next.js frontend:**

```bash
cd web && npm run dev
```

The Next.js app at `http://localhost:3000` will proxy `/api/*` requests to the Python server at `http://localhost:8000`.

### Full-stack local development (Vercel CLI)

Alternatively, install the Vercel CLI and run both together (requires a one-time login):

```bash
npm install -g vercel
export LLM_LITE_TOKEN="your-token"
export LLM_LITE_URL="your-litellm-base-url"
vercel dev
```

---

## Deploying to Vercel

1. Push the repo to GitHub
2. Import the project in [vercel.com](https://vercel.com)
3. Set the following environment variables in the Vercel project settings:
   - `LLM_LITE_TOKEN`
   - `LLM_LITE_URL`
4. Deploy — Vercel will build the Next.js app and deploy the Python serverless function automatically
