from collections import OrderedDict
from hashlib import sha256
from threading import RLock
from time import monotonic
from typing import Generic, TypeVar
import unicodedata


T = TypeVar("T")


class InMemoryCache(Generic[T]):
    def __init__(self, ttl_seconds: float = 300, max_entries: int = 1_000) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if max_entries < 1:
            raise ValueError("max_entries must be at least one")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = RLock()

    def get(self, prompt: str, *, namespace: str = "default") -> T | None:
        key = self._make_key(prompt, namespace)
        now = monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return value

    def set(self, prompt: str, value: T, *, namespace: str = "default") -> None:
        key = self._make_key(prompt, namespace)
        now = monotonic()
        with self._lock:
            expired_keys = [
                entry_key
                for entry_key, (expires_at, _) in self._entries.items()
                if expires_at <= now
            ]
            for expired_key in expired_keys:
                del self._entries[expired_key]

            self._entries[key] = (now + self.ttl_seconds, value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def delete(self, prompt: str, *, namespace: str = "default") -> bool:
        key = self._make_key(prompt, namespace)
        with self._lock:
            if key not in self._entries:
                return False
            del self._entries[key]
            return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @staticmethod
    def _make_key(prompt: str, namespace: str) -> str:
        if not isinstance(prompt, str) or not isinstance(namespace, str):
            raise TypeError("prompt and namespace must be strings")
        normalized_prompt = " ".join(
            unicodedata.normalize("NFKC", prompt).casefold().split()
        )
        normalized_namespace = unicodedata.normalize("NFKC", namespace).casefold()
        key_material = f"{normalized_namespace}\0{normalized_prompt}"
        return sha256(key_material.encode("utf-8")).hexdigest()


response_cache: InMemoryCache[object] = InMemoryCache()
