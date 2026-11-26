import json
import re
from pathlib import Path
from .config import ROOT, RAG_INDEX_PATH

SOURCE_PATH = ROOT / "backend" / "knowledge_sources.json"
SOURCES = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

class MedicalRetriever:
    def __init__(self):
        self.kind = "keyword"
        self.collection = None
        self.model = None
        try:
            import chromadb
            from chromadb.config import Settings
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
            # Health-assessment retrieval stays local and does not send Chroma telemetry.
            client = chromadb.PersistentClient(path=str(RAG_INDEX_PATH), settings=Settings(anonymized_telemetry=False))
            self.collection = client.get_or_create_collection("trusted_health_references", metadata={"hnsw:space": "cosine"})
            existing = self.collection.get(include=["documents"])
            indexed_documents = dict(zip(existing.get("ids", []), existing.get("documents", [])))
            expected_documents = {f"source-{i}": source["summary"] for i, source in enumerate(SOURCES)}
            if indexed_documents != expected_documents:
                if indexed_documents:
                    self.collection.delete(ids=list(indexed_documents))
                vectors = self.model.encode([x["summary"] for x in SOURCES], normalize_embeddings=True).tolist()
                self.collection.add(ids=list(expected_documents), embeddings=vectors,
                                    documents=[x["summary"] for x in SOURCES],
                                    metadatas=[{"url": x["url"], "title": x["title"], "publisher": x["publisher"]} for x in SOURCES])
            self.kind = "Sentence Transformers + ChromaDB"
        except Exception:
            pass

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.collection is not None:
            vector = self.model.encode([query], normalize_embeddings=True).tolist()
            result = self.collection.query(query_embeddings=vector, n_results=min(k, self.collection.count()))
            rows = []
            for metadata, document, distance in zip(result["metadatas"][0], result["documents"][0], result["distances"][0]):
                if distance > 0.65: continue
                rows.append({"title": metadata["title"], "url": metadata["url"], "publisher": metadata["publisher"], "summary": document})
            return rows
        stop = {"what", "when", "where", "which", "does", "with", "have", "about", "from", "that", "this", "your", "could", "would", "should", "into", "tell", "please", "explain"}
        terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2 and term not in stop}
        ranked = sorted(((len(terms & set(re.findall(r"[a-z0-9]+", (row["title"] + " " + row["summary"]).lower()))), row) for row in SOURCES), key=lambda pair: pair[0], reverse=True)
        return [row for score, row in ranked[:k] if score > 0]

retriever = MedicalRetriever()
