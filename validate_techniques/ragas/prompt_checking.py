import json
import os
import time
import requests
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

MAX_PAGES = 50

OUTPUT_DIR = "../../repositories/wiki_ragas"
DOCS_DIR = os.path.join(OUTPUT_DIR, "documents")
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

os.makedirs(DOCS_DIR, exist_ok=True)

# IMPORTANT: Wikipedia requires a proper User-Agent
HEADERS = {
    "User-Agent": "RAGCorpusBuilder/1.0 (contact: your_email@example.com)"
}


# =========================
# STEP 1: LOAD DATASET
# =========================

def load_wikieval_sources():
    ds = load_dataset("vibrantlabsai/WikiEval")
    data = ds["train"]

    sources = data["source"]

    seen = set()
    pages = []

    for s in sources:
        if s and s not in seen:
            seen.add(s)
            pages.append(s)

    return pages


# =========================
# STEP 2: CLEAN FILE NAME
# =========================

def clean_filename(title):
    return title.replace(" ", "_").replace("/", "_")[:150]


# =========================
# STEP 3: WIKIPEDIA FETCH (ROBUST)
# =========================

def fetch_wikipedia_page(title):

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": 1
    }

    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)

        if r.status_code != 200:
            print(f"[FAIL HTTP {r.status_code}] {title}")
            return None

        data = r.json()

        pages = data["query"]["pages"]
        page = next(iter(pages.values()))

        text = page.get("extract", "")

        if not text or len(text) < 200:
            print(f"[EMPTY PAGE] {title}")
            return None

        return {
            "title": page.get("title", title),
            "text": text
        }

    except Exception as e:
        print(f"[ERROR] {title} -> {e}")
        return None


# =========================
# STEP 4: SAVE DOCUMENT
# =========================

def save_document(title, text, idx):

    filename = f"{idx}_{clean_filename(title)}.txt"
    path = os.path.join(DOCS_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return filename


# =========================
# STEP 5: BUILD CORPUS
# =========================

def build_corpus(pages):

    metadata = []

    for i, title in enumerate(tqdm(pages[:MAX_PAGES])):

        print(f"\n[PROCESSING] {title}")

        page_data = fetch_wikipedia_page(title)

        if not page_data:
            print(f"[SKIP] {title}")
            continue

        file_name = save_document(page_data["title"], page_data["text"], i)

        metadata.append({
            "id": i,
            "source": title,
            "resolved_title": page_data["title"],
            "file": file_name,
            "char_length": len(page_data["text"])
        })

        print(f"[SAVED] {page_data['title']}")

        time.sleep(0.5)  # polite rate limiting

    return metadata


# =========================
# STEP 6: SAVE METADATA
# =========================

def save_metadata(metadata):

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("Loading WikiEval dataset...")
    pages = load_wikieval_sources()

    print(f"Total unique sources: {len(pages)}")

    print("Building RAG corpus...")

    metadata = build_corpus(pages)

    print(f"\nSuccessfully saved {len(metadata)} documents")

    save_metadata(metadata)

    print(f"\nDONE → Output folder: {OUTPUT_DIR}")