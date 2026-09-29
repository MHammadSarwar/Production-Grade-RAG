from app.cache import InMemoryCache


def test_cache_miss_store_and_case_insensitive_hit():
	cache = InMemoryCache[str]()
	model_call_count = 0
	prompt = "What is Python?"

	print(f"Cache MISS: {prompt}")
	cached_response = cache.get(prompt)
	assert cached_response is None

	print("Simulated LLM call: generating answer")
	model_call_count += 1
	response = "Python is a programming language."
	cache.set(prompt, response)
	print("Cache STORE: response saved")

	repeated_prompt = "  WHAT IS PYTHON?  "
	cached_response = cache.get(repeated_prompt)
	print(f"Cache HIT: {repeated_prompt.strip()}" if cached_response else "Cache MISS")

	assert cached_response == response
	assert model_call_count == 1
	print(f"Simulated LLM calls: {model_call_count}")
