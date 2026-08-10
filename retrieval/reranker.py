import os
import queue
import threading
import time
from collections import OrderedDict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class CrossEncoderReranker:
    """Shared query-passage cross-encoder scorer with true pair-level batching.

    Features:
    1. Bounded OrderedDict LRU Pair Cache (query, passage) -> float.
    2. Central Pair-Level Coordinator Queue: breaks logical multi-pair requests into pair items,
       buckets them by token length to minimize padding waste, and batches them into physical
       GPU batches of batch_size (32 or 64) across all concurrent evaluation workers.
    3. Step-down CUDA OOM recovery (batch size reduction first before CPU fallback).
    4. Separately tracks enqueue-to-first-batch delay, model execution time, physical batch count,
       pairs per physical batch, padding efficiency, and total request completion time.
    """

    def __init__(self, model_name, device="cpu", max_length=512, batch_size=32):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking requires sentence-transformers and torch."
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

        # Detailed Telemetry metrics
        self._telemetry_lock = threading.Lock()
        self.stats_total_requests = 0
        self.stats_total_pairs = 0
        self.stats_cache_hits = 0
        self.stats_total_enqueue_to_first_batch_s = 0.0
        self.stats_total_request_completion_s = 0.0
        self.stats_total_eval_time_s = 0.0
        self.stats_physical_batches = 0
        self.stats_total_evaluated_pairs = 0
        self.stats_total_tokens = 0
        self.stats_total_padding_tokens = 0

        # Central Pair-Level Coordinator Queue
        self._queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._pair_coordinator_worker_loop, daemon=True)
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
                print(f"[LocalReranker WARNING] Failed to load on {self.device} ({exc}). Falling back to CPU.")
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
            f"[LocalReranker] Successfully initialized '{self.model_name}' on device '{self.device}' "
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
                    torch.cuda.empty_cache()
                    if self.batch_size > 8:
                        self.batch_size = max(8, self.batch_size // 2)
                        print(
                            f"\n[LocalReranker WARNING] CUDA OOM during predict ({oom_exc}). "
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
                            f"\n[LocalReranker WARNING] Permanent CUDA OOM. "
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

        return [float(s) for s in scores]

    def _pair_coordinator_worker_loop(self):
        while True:
            try:
                first_item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            gathered_items = [first_item]
            max_gather = max(64, self.batch_size * 2)
            start_wait = time.perf_counter()

            while len(gathered_items) < max_gather and (time.perf_counter() - start_wait) < 0.002:
                try:
                    item = self._queue.get_nowait()
                    gathered_items.append(item)
                except queue.Empty:
                    break

            if not gathered_items:
                continue

            # Token length bucketing to minimize padding:
            # Estimate pair token length via len(q.split()) + len(p.split())
            gathered_items.sort(
                key=lambda x: len(x[2][0].split()) + len(x[2][1].split())
            )

            physical_batch_size = max(1, self.batch_size)
            for i in range(0, len(gathered_items), physical_batch_size):
                physical_chunk = gathered_items[i : i + physical_batch_size]
                if not physical_chunk:
                    continue

                now = time.perf_counter()
                for req, _, _ in physical_chunk:
                    if req["first_batch_time"] is None:
                        req["first_batch_time"] = now

                physical_pairs = [pair for _, _, pair in physical_chunk]

                # Compute token counts and padding waste for telemetry
                pair_token_lens = [
                    min(self.max_length, len(q.split()) + len(p.split()) + 3)
                    for q, p in physical_pairs
                ]
                max_token_len = max(pair_token_lens) if pair_token_lens else 1
                total_tokens = sum(pair_token_lens)
                batch_capacity = max_token_len * len(physical_pairs)
                padding_waste = max(0, batch_capacity - total_tokens)

                model_start = time.perf_counter()
                try:
                    scores = self._predict_raw(physical_pairs)
                    model_dur = time.perf_counter() - model_start

                    with self._cache_lock:
                        for pair, score in zip(physical_pairs, scores):
                            self._cache[pair] = score
                            self._cache.move_to_end(pair)
                            if len(self._cache) > self._max_cache_size:
                                self._cache.popitem(last=False)

                    completed_reqs = []
                    for (req, pair_idx, _), score in zip(physical_chunk, scores):
                        req["scores"][pair_idx] = score
                        req["completed_pairs"] += 1
                        if req["completed_pairs"] == req["total_pairs"]:
                            req["completion_time"] = time.perf_counter()
                            completed_reqs.append(req)
                            req["event"].set()

                    with self._telemetry_lock:
                        self.stats_physical_batches += 1
                        self.stats_total_evaluated_pairs += len(physical_pairs)
                        self.stats_total_eval_time_s += model_dur
                        self.stats_total_tokens += total_tokens
                        self.stats_total_padding_tokens += padding_waste

                        for req in completed_reqs:
                            self.stats_total_requests += 1
                            first_delay = req["first_batch_time"] - req["enqueued_time"]
                            comp_delay = req["completion_time"] - req["enqueued_time"]
                            self.stats_total_enqueue_to_first_batch_s += max(0.0, first_delay)
                            self.stats_total_request_completion_s += max(0.0, comp_delay)

                except Exception as exc:
                    for req, _, _ in physical_chunk:
                        req["exception"] = exc
                        req["event"].set()

            for _ in range(len(gathered_items)):
                self._queue.task_done()

    def score_pairs(self, pairs):
        pairs = [(str(q), str(p)) for q, p in pairs]
        if not pairs:
            return [], 0.0

        started = time.perf_counter()
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
            "total_pairs": len(uncached_pairs),
            "completed_pairs": 0,
            "scores": [0.0] * len(uncached_pairs),
            "event": req_event,
            "exception": None,
            "enqueued_time": time.perf_counter(),
            "first_batch_time": None,
            "completion_time": None,
        }

        for pair_idx, pair in enumerate(uncached_pairs):
            self._queue.put((request_obj, pair_idx, pair))

        signaled = req_event.wait(timeout=120.0)
        if not signaled:
            raise TimeoutError("Reranker request timed out waiting for pair coordinator worker.")

        if request_obj["exception"] is not None:
            raise request_obj["exception"]

        new_scores = request_obj["scores"]
        final_scores = [0.0] * len(pairs)
        for idx, score in cached_scores.items():
            final_scores[idx] = score
        for idx, score in zip(uncached_indices, new_scores):
            final_scores[idx] = score

        latency = time.perf_counter() - started
        return final_scores, latency

    def score(self, question, passages):
        passages = [str(passage) for passage in passages]
        if not passages:
            return [], 0.0
        pairs = [(str(question), passage) for passage in passages]
        return self.score_pairs(pairs)

    def describe(self):
        with self._telemetry_lock:
            total_pairs = self.stats_total_pairs
            cache_hits = self.stats_cache_hits
            total_reqs = self.stats_total_requests
            first_batch_s = self.stats_total_enqueue_to_first_batch_s
            completion_s = self.stats_total_request_completion_s
            eval_time_s = self.stats_total_eval_time_s
            physical_batches = self.stats_physical_batches
            eval_pairs = self.stats_total_evaluated_pairs
            tokens = self.stats_total_tokens
            padding = self.stats_total_padding_tokens

        hit_rate = (cache_hits / total_pairs) if total_pairs > 0 else 0.0
        avg_first_batch_ms = ((first_batch_s / max(1, total_reqs)) * 1000) if total_reqs > 0 else 0.0
        avg_completion_ms = ((completion_s / max(1, total_reqs)) * 1000) if total_reqs > 0 else 0.0
        avg_eval_ms = ((eval_time_s / max(1, physical_batches)) * 1000) if physical_batches > 0 else 0.0
        avg_pairs_per_batch = (eval_pairs / physical_batches) if physical_batches > 0 else 0.0

        total_tokens_with_pad = tokens + padding
        padding_efficiency = (tokens / total_tokens_with_pad) if total_tokens_with_pad > 0 else 1.0

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
            "total_requests": total_reqs,
            "avg_enqueue_to_first_batch_ms": round(avg_first_batch_ms, 3),
            "avg_request_completion_ms": round(avg_completion_ms, 3),
            "avg_model_eval_ms": round(avg_eval_ms, 3),
            "total_physical_batches": physical_batches,
            "avg_pairs_per_physical_batch": round(avg_pairs_per_batch, 1),
            "token_padding_efficiency": round(padding_efficiency, 4),
            "training_corpus_note": "BAAI/bge-reranker-base cross-encoder; no HotpotQA labels used by this project",
        }



# Backward-compatible import alias for older tests/scripts.
CrossEncoderEvidenceReranker = CrossEncoderReranker
