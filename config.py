import os
from dotenv import load_dotenv

load_dotenv()


def load_eval_config(config_path="config/fullwiki.yaml"):
    """Load configuration dictionary from YAML file or return empty dict if missing."""
    if not config_path or not os.path.isfile(config_path):
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        config_data = {}
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if val.isdigit():
                        val = int(val)
                    elif val.replace(".", "", 1).isdigit() and val.count(".") == 1:
                        val = float(val)
                    config_data[key] = val
        return config_data


# Local GCE Inference Server (vLLM / Ollama / Local OpenAI Server)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

# Prevent Pyserini / OpenAI SDK top-level import error when running local models without OpenAI key
os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY if OPENAI_API_KEY else "EMPTY")

MAX_AGENT_HOPS = int(os.getenv("MAX_AGENT_HOPS", "7"))
REACT_SEARCH_TOP_K = int(os.getenv("REACT_SEARCH_TOP_K", "10"))
REACT_MAX_EVIDENCE_DOCUMENTS = int(os.getenv("REACT_MAX_EVIDENCE_DOCUMENTS", "15"))
REACT_MAX_OBSERVATION_CHARS = int(os.getenv("REACT_MAX_OBSERVATION_CHARS", "22000"))
# ReAct evidence-memory reranker using BAAI/bge-reranker-base.
REACT_MEMORY_RERANKER_MODEL = os.getenv(
    "REACT_MEMORY_RERANKER_MODEL", "BAAI/bge-reranker-base"
)
REACT_MEMORY_RERANKER_DEVICE = os.getenv("REACT_MEMORY_RERANKER_DEVICE", "cuda")
REACT_MEMORY_RERANKER_MAX_LENGTH = int(os.getenv("REACT_MEMORY_RERANKER_MAX_LENGTH", "512"))
REACT_MEMORY_RERANKER_BATCH_SIZE = int(os.getenv("REACT_MEMORY_RERANKER_BATCH_SIZE", "64"))
BASELINE_SEARCH_TOP_K = int(os.getenv("BASELINE_SEARCH_TOP_K", "7"))
WIKIPEDIA_USER_AGENT = os.getenv(
    "USER_AGENT", "HotpotQAReActAgent/1.0 (https://github.com/example/hotpot)"
)

HOTPOT_DEV_DATASET = "hotpot_qa"
HOTPOT_CONFIG = "fullwiki"
HOTPOT_SPLIT = "validation"
HOTPOT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"

# HotpotQA global FullWiki retrieval (official October 1, 2017 intro-paragraph corpus)
FULLWIKI_ARCHIVE_URL = os.getenv(
    "FULLWIKI_ARCHIVE_URL",
    "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2",
)
FULLWIKI_ARCHIVE_MD5 = "01edf64cd120ecc03a2745352779514c"
FULLWIKI_DATA_DIR = os.getenv("FULLWIKI_DATA_DIR", "data/fullwiki")
FULLWIKI_ARCHIVE_PATH = os.getenv(
    "FULLWIKI_ARCHIVE_PATH",
    os.path.join(FULLWIKI_DATA_DIR, "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"),
)
FULLWIKI_CORPUS_DIR = os.getenv(
    "FULLWIKI_CORPUS_DIR", os.path.join(FULLWIKI_DATA_DIR, "corpus")
)
FULLWIKI_INDEX_DIR = os.getenv("FULLWIKI_INDEX_DIR", "indexes/fullwiki")
FULLWIKI_BM25_INDEX_DIR = os.path.join(FULLWIKI_INDEX_DIR, "bm25")
FULLWIKI_DENSE_INDEX_PATH = os.path.join(FULLWIKI_INDEX_DIR, "dense.faiss")
FULLWIKI_INDEX_MANIFEST = os.path.join(FULLWIKI_INDEX_DIR, "manifest.json")
DENSE_EMBEDDING_MODEL = os.getenv("DENSE_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
DENSE_QUERY_DEVICE = os.getenv("DENSE_QUERY_DEVICE", "cpu")
FULLWIKI_FAISS_FACTORY = os.getenv("FULLWIKI_FAISS_FACTORY", "IVF4096,PQ96x8")
FULLWIKI_DENSE_TRAIN_SIZE = int(os.getenv("FULLWIKI_DENSE_TRAIN_SIZE", "200000"))
FULLWIKI_DENSE_NPROBE = int(os.getenv("FULLWIKI_DENSE_NPROBE", "32"))
FULLWIKI_SEARCH_CANDIDATES = int(os.getenv("FULLWIKI_SEARCH_CANDIDATES", "40"))
FULLWIKI_RRF_K = int(os.getenv("FULLWIKI_RRF_K", "60"))
