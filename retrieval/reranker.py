import os
import queue
import threading
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class CrossEncoderEvidenceReranker:
    """Shared cross-encoder scorer for ReAct evidence memory.

    Features:
    1. Thread-safe LRU Pair Cache (query, passage) -> float
    2. Central Dynamic Microbatching Queue to combine concurrent requests across
       all evaluation threads into optimal GPU batches.
    """

    def __init__(self, model_name, device="cpu", max_length=512, batch_size=32):
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

        # Thread-safe LRU pair cache
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 100000

        # Central Dynamic Microbatching Queue
        self._queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._microbatch_worker_loop, daemon=True)
        self._worker_thread.start()

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

    def _predict_raw(self, pairs):
        if not pairs:
            return []
        import torch
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
            finally:
                if "cuda" in self.device.lower():
                    torch.cuda.empty_cache()
        return [float(s) for s in scores]

    def _microbatch_worker_loop(self):
        max_batch = self.batch_size
        while True:
            try:
                first_item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch_requests = [first_item]
            start_wait = time.perf_counter()
            while len(batch_requests) < max_batch and (time.perf_counter() - start_wait) < 0.002:
                try:
                    item = self._queue.get_nowait()
                    batch_requests.append(item)
                except queue.Empty:
                    break

            all_pairs = []
            for req in batch_requests:
                for pair in req["pairs"]:
                    all_pairs.append(pair)

            if all_pairs:
                try:
                    scores = self._predict_raw(all_pairs)
                    with self._cache_lock:
                        for pair, score in zip(all_pairs, scores):
                            if len(self._cache) < self._max_cache_size:
                                self._cache[pair] = score

                    idx = 0
                    for req in batch_requests:
                        num_pairs = len(req["pairs"])
                        req["results"] = scores[idx : idx + num_pairs]
                        idx += num_pairs
                        req["event"].set()
                except Exception as exc:
                    for req in batch_requests:
                        req["exception"] = exc
                        req["event"].set()

            for _ in range(len(batch_requests)):
                self._queue.task_done()

    def score(self, question, passages):
        passages = [str(passage) for passage in passages]
        if not passages:
            return [], 0.0

        started = time.perf_counter()
        pairs = [(str(question), passage) for passage in passages]

        cached_scores = {}
        uncached_pairs = []
        uncached_indices = []

        with self._cache_lock:
            for idx, pair in enumerate(pairs):
                if pair in self._cache:
                    cached_scores[idx] = self._cache[pair]
                else:
                    uncached_pairs.append(pair)
                    uncached_indices.append(idx)

        if not uncached_pairs:
            result = [cached_scores[i] for i in range(len(pairs))]
            return result, time.perf_counter() - started

        req_event = threading.Event()
        request_obj = {
            "pairs": uncached_pairs,
            "event": req_event,
            "results": None,
            "exception": None,
        }
        self._queue.put(request_obj)
        req_event.wait(timeout=60.0)

        if request_obj["exception"] is not None:
            raise request_obj["exception"]

        new_scores = request_obj["results"] or self._predict_raw(uncached_pairs)

        final_scores = [0.0] * len(pairs)
        for idx, score in cached_scores.items():
            final_scores[idx] = score
        for idx, score in zip(uncached_indices, new_scores):
            final_scores[idx] = score

        latency = time.perf_counter() - started
        return final_scores, latency

    def describe(self):
        return {
            "model": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "cache_size": len(self._cache),
            "training_corpus_note": "BAAI/bge-reranker-base cross-encoder; no HotpotQA labels used by this project",
        }
