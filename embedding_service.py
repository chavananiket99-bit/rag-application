from pathlib import Path
import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "all-MiniLM-L6-v2"

class EmbeddingService:
    def __init__(self):
        logger.info("Loading embedding model: %s", MODEL_PATH)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Embedding model not found at {MODEL_PATH}")

        self.model = SentenceTransformer(str(MODEL_PATH), local_files_only=True)
        logger.info("Embedding model loaded successfully.")
        logger.info("Embedding model dimension: %s", self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts):

        if not texts:
            return[]

        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        return embeddings.tolist()

    def embed_text(self, text):
        if not text or not text.strip():
            return []
        
        embedding = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    
        return embedding.tolist()

embedding_service = EmbeddingService()
