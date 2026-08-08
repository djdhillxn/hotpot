import os
from dotenv import load_dotenv

load_dotenv()

# Local GCE Inference Server (vLLM / Ollama / Local OpenAI Server)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

MAX_AGENT_HOPS = int(os.getenv("MAX_AGENT_HOPS", "5"))
WIKIPEDIA_USER_AGENT = os.getenv(
    "USER_AGENT", "HotpotQAReActAgent/1.0 (https://github.com/example/hotpot)"
)

HOTPOT_DEV_DATASET = "hotpot_qa"
HOTPOT_CONFIG = "fullwiki"
HOTPOT_SPLIT = "validation"
HOTPOT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
