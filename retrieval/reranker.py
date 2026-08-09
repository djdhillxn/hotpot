import os
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class CrossEncoderEvidenceReranker:
    """Shared cross-encoder scorer for ReAct evidence memory.

    The model is loaded once per benchmark process and shared by all per-question
    retrieval sessions. Scoring is serialized to avoid concurrent PyTorch model
    forwards fighting over CPU/GPU resources when the evaluator uses many workers.
    """

    def __init__(self, model_name, device="cpu", max_length=512, batch_size=64):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Evidence-memory reranking requires sentence-transformers and torch."
            ) from exc

        self.model_name = str(model_name)
        self.requested_device = str(device)
        self.device = str(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self._lock = threading.Lock()

        # Limit internal PyTorch CPU threads when running on CPU to prevent worker thread thrashing
        if self.device == "cpu":
            num_threads = min(4, max(1, (os.cpu_count() or 4) // 4))
            torch.set_num_threads(num_threads)

        model_kwargs = {}
        self.dtype_str = "float32"
        if "cuda" in self.device.lower():
            model_kwargs["torch_dtype"] = torch.float16
            self.dtype_str = "float16"

        try:
            try:
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                    model_kwargs=model_kwargs,
                )
            except (TypeError, ValueError):
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                )

            if "cuda" in self.device.lower() and hasattr(self.model, "model"):
                self.model.model.to(torch.float16)

        except Exception as exc:
            if "cuda" in self.device.lower():
                print(f"[EvidenceReranker WARNING] Failed to load on {self.device} ({exc}). Falling back to CPU.")
                self.device = "cpu"
                self.dtype_str = "float32"
                num_threads = min(4, max(1, (os.cpu_count() or 4) // 4))
                torch.set_num_threads(num_threads)
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device="cpu",
                )
                if hasattr(self.model, "model"):
                    self.model.model.to("cpu").to(torch.float32)
            else:
                raise exc

        print(
            f"[EvidenceReranker] Successfully initialized '{self.model_name}' on device '{self.device}' "
            f"(torch_dtype={self.dtype_str}, batch_size={self.batch_size}, max_length={self.max_length})"
        )

    def score(self, question, passages):
        passages = [str(passage) for passage in passages]
        if not passages:
            return [], 0.0

        import torch

        pairs = [(str(question), passage) for passage in passages]
        started = time.perf_counter()
        with self._lock:
            try:
                with torch.inference_mode():
                    scores = self.model.predict(
                        pairs,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
            except torch.OutOfMemoryError as oom_exc:
                if "cuda" in self.device.lower():
                    print(
                        f"\n[EvidenceReranker WARNING] CUDA OOM during predict ({oom_exc}). "
                        "Fallback: moving CrossEncoder to CPU permanently."
                    )
                    torch.cuda.empty_cache()
                    self.device = "cpu"
                    self.dtype_str = "float32"
                    if hasattr(self.model, "model"):
                        self.model.model.to("cpu").to(torch.float32)
                    num_threads = min(4, max(1, (os.cpu_count() or 4) // 4))
                    torch.set_num_threads(num_threads)
                    with torch.inference_mode():
                        scores = self.model.predict(
                            pairs,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                else:
                    raise oom_exc

        latency = time.perf_counter() - started
        return [float(score) for score in scores], latency

    def describe(self):
        return {
            "model": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "training_corpus_note": "BAAI/bge-reranker-base cross-encoder; no HotpotQA labels used by this project",
        }
