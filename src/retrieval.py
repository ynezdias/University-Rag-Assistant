"""BM25 and reciprocal-rank fusion for the small local corpus."""
import math
import re
from collections import Counter
from functools import lru_cache

STOP = set("a an the is are was were what which how when do does for of to in and or i my it its about me tell please".split())


def tokens(text):
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOP]


@lru_cache(maxsize=8)
def lexical_index(collection_name, storage_path):
    import chromadb
    data = chromadb.PersistentClient(path=storage_path).get_collection(collection_name).get()
    records = []
    for identifier, text, metadata in zip(data["ids"], data["documents"], data["metadatas"]):
        words = tokens(text + " " + metadata.get("filename", "").replace("_", " "))
        records.append((identifier, text, metadata, Counter(words), len(words)))
    return records


def bm25(query, records):
    query_words = set(tokens(query))
    average = sum(r[4] for r in records) / max(len(records), 1)
    frequencies = {word: sum(word in r[3] for r in records) for word in query_words}
    scores = []
    for identifier, _, _, counts, length in records:
        score = 0.0
        for word in query_words:
            count = counts[word]
            if count:
                inverse = math.log(1 + (len(records) - frequencies[word] + 0.5) / (frequencies[word] + 0.5))
                score += inverse * count * 2.5 / (count + 1.5 * (0.25 + 0.75 * length / max(average, 1)))
        if score > 0:
            scores.append((identifier, score))
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def fuse(semantic_ids, lexical_ids):
    scores = Counter()
    for ranking in (semantic_ids, lexical_ids):
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] += 1 / (60 + rank)
    return sorted(scores, key=lambda identifier: (-scores[identifier], identifier))
