import os
import queue
import threading
import time
from collections import OrderedDict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class CrossEncoderEvidenceReranker:
    """Shared cross-encoder scorer for ReAct evidence memory.

    Features:
    1. Bounded OrderedDict LRU Pair Cache (query, passage) -> float
    2. Central Dynamic Microbatching Queue bounded by pair count to combine concurrent requests across
       all evaluation threads into optimal GPU batches.
    3. Step-down CUDA OOM recovery (batch size reduction first before CPU fallback).
    4. Telemetry metrics for queue wait, model evaluation, pair counts, and cache hit rates.
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

        # Thread-safe Bounded OrderedDict LRU pair cache
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()
        self._max_cache_size = 100000

        # Telemetry metrics
        self._telemetry_lock = threading.Lock()
        self.stats_total_pairs = 0
        self.stats_cache_hits = 0
        self.stats_total_queue_wait_seconds = 0.0
        self.stats_total_eval_time_seconds = 0.0
        self.stats_total_eval_calls = 0

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
        eval_started = time.perf_counter()
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
                    torch.cuda.empty_cache()
                    if self.batch_size > 8:
                        self.batch_size = max(8, self.batch_size // 2)
                        print(
                            f"\n[EvidenceReranker WARNING] CUDA OOM during predict ({oom_exc}). "
                            f"Halving prediction batch_size to {self.batch_size}."
                        )
                        with torch.inference_mode():
                            scores = self.model.predict(
                                pairs,
                                batch_size=self.batch_size,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                            )
                    else:
                        print(
                            f"\n[EvidenceReranker WARNING] Permanent CUDA OOM. "
                            "Fallback: moving CrossEncoder to CPU permanently."
                        )
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

        eval_duration = time.perf_counter() - eval_started
        with self._telemetry_lock:
            self.stats_total_eval_time_seconds += eval_duration
            self.stats_total_eval_calls += 1

        return [float(s) for s in scores]

    def _microbatch_worker_loop(self):
        # Bound GPU batches by total PAIR count, not request count
        max_pair_batch = max(64, self.batch_size * 2)

        while True:
            try:
                first_item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch_requests = [first_item]
            all_pairs = list(first_item["pairs"])
            start_wait = time.perf_counter()

            while len(all_pairs) < max_pair_batch and (time.perf_counter() - start_wait) < 0.002:
                try:
                    item = self._queue.get_nowait()
                    item_pair_count = len(item["pairs"])
                    if len(all_pairs) + item_pair_count > max_pair_batch and len(all_pairs) > 0:
                        self._queue.put(item)
                        break
                    batch_requests.append(item)
                    all_pairs.extend(item["pairs"])
                except queue.Empty:
                    break

            if all_pairs:
                try:
                    scores = self._predict_raw(all_pairs)
                    with self._cache_lock:
                        for pair, score in zip(all_pairs, scores):
                            self._cache[pair] = score
                            self._cache.move_to_end(pair)
                            if len(self._cache) > self._max_cache_size:
                                self._cache.popitem(last=False)

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
                    self._cache.move_to_end(pair)
                else:
                    uncached_pairs.append(pair)
                    uncached_indices.append(idx)

        with self._telemetry_lock:
            self.stats_total_pairs += len(pairs)
            self.stats_cache_hits += len(cached_scores)

        if not uncached_pairs:
            result = [cached_scores[i] for i in range(len(pairs))]
            return result, time.perf_counter() - started

        req_event = threading.Event()
        request_obj = {
            "pairs": uncached_pairs,
            "event": req_event,
            "results": None,
            "exception": None,
            "enqueued_time": time.perf_counter(),
        }
        self._queue.put(request_obj)

        # Never perform duplicate fallback inference on timeout; wait for queue completion or raise TimeoutError
        signaled = req_event.wait(timeout=120.0)
        if not signaled:
            raise TimeoutError("Reranker request timed out waiting for dynamic microbatching worker.")

        queue_wait_duration = time.perf_counter() - request_obj["enqueued_time"]
        with self._telemetry_lock:
            self.stats_total_queue_wait_seconds += queue_wait_duration

        if request_obj["exception"] is not None:
            raise request_obj["exception"]

        new_scores = request_obj["results"]
        if new_scores is None:
            raise RuntimeError("Reranker worker returned None results without exception.")

        final_scores = [0.0] * len(pairs)
        for idx, score in cached_scores.items():
            final_scores[idx] = score
        for idx, score in zip(uncached_indices, new_scores):
            final_scores[idx] = score

        latency = time.perf_counter() - started
        return final_scores, latency

    def describe(self):
        with self._telemetry_lock:
            total_pairs = self.stats_total_pairs
            cache_hits = self.stats_cache_hits
            queue_wait_s = self.stats_total_queue_wait_seconds
            eval_time_s = self.stats_total_eval_time_seconds
            eval_calls = self.stats_total_eval_calls

        hit_rate = (cache_hits / total_pairs) if total_pairs > 0 else 0.0
        avg_wait_ms = ((queue_wait_s / max(1, total_pairs)) * 1000) if total_pairs > 0 else 0.0
        avg_eval_ms = ((eval_time_s / max(1, eval_calls)) * 1000) if eval_calls > 0 else 0.0

        return {
            "model": self.model_name,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "cache_size": len(self._cache),
            "max_cache_size": self._max_cache_size,
            "total_pairs_scored": total_pairs,
            "cache_hits": cache_hits,
            "cache_hit_rate": round(hit_rate, 4),
            "avg_queue_wait_ms": round(avg_wait_ms, 3),
            "avg_model_eval_ms": round(avg_eval_ms, 3),
            "training_corpus_note": "BAAI/bge-reranker-base cross-encoder; no HotpotQA labels used by this project",
        }
