import os
from dotenv import load_dotenv

load_dotenv()

# Local GCE Inference Server (vLLM / Ollama / Local OpenAI Server)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

MAX_AGENT_HOPS = int(os.getenv("MAX_AGENT_HOPS", "7"))
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
FULLWIKI_DENSE_NPROBE = int(os.getenv("FULLWIKI_DENSE_NPROBE", "16"))
FULLWIKI_SEARCH_CANDIDATES = int(os.getenv("FULLWIKI_SEARCH_CANDIDATES", "20"))
FULLWIKI_RRF_K = int(os.getenv("FULLWIKI_RRF_K", "60"))
