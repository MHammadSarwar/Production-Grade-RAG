from app.cache import InMemoryCache


class TestInMemoryCache:
	def test_miss_store_and_normalized_hit_avoids_repeat_call(self):
		cache = InMemoryCache[str]()
		model_call_count = 0
		prompt = "What is Python?"

		cached_response = cache.get(prompt)
		assert cached_response is None

		model_call_count += 1
		response = "Python is a programming language."
		cache.set(prompt, response)

		cached_response = cache.get("  WHAT IS PYTHON?  ")

		assert cached_response == response
		assert model_call_count == 1

	def test_keeps_namespaces_isolated(self):
		cache = InMemoryCache[str]()
		cache.set("What is Python?", "answer", namespace="model-a")

		assert cache.get("what is python?", namespace="model-a") == "answer"
		assert cache.get("what is python?", namespace="model-b") is None

	def test_expired_entry_is_a_miss(self, monkeypatch):
		clock = [100.0]
		monkeypatch.setattr("app.cache.monotonic", lambda: clock[0])
		cache = InMemoryCache[str](ttl_seconds=5)
		cache.set("question", "answer")

		clock[0] += 5

		assert cache.get("question") is None
		assert cache.stats()["entries"] == 0

	def test_evicts_least_recently_used_entry(self):
		cache = InMemoryCache[str](max_entries=2)
		cache.set("one", "first")
		cache.set("two", "second")
		assert cache.get("one") == "first"
		cache.set("three", "third")

		assert cache.get("two") is None
		assert cache.get("one") == "first"
		assert cache.get("three") == "third"

	def test_reports_hit_miss_store_and_eviction_stats(self):
		cache = InMemoryCache[str](max_entries=1)
		cache.set("one", "first")
		cache.set("two", "second")
		assert cache.get("one") is None
		assert cache.get("two") == "second"

		stats = cache.stats()
		assert stats["hits"] == 1
		assert stats["misses"] == 1
		assert stats["stores"] == 2
		assert stats["evictions"] == 1
		assert stats["entries"] == 1
