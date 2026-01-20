import openai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

# ---------
# GLOBAL MODEL CACHE (关键)
# ---------
_local_model = None
_LOCAL_DIM = 384
_OPENAI_DIM = 1536


def get_local_model():
    global _local_model
    if _local_model is None:
        logger.info("Loading local SentenceTransformer model...")
        _local_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _local_model


class EmbeddingGenerator:
    def __init__(
        self,
        api_key: str = None,
        use_local: bool = False,
        use_google: bool = False,
    ):
        self.api_key = api_key
        self.use_local = use_local
        self.use_google = use_google

        if not use_local and not use_google:
            if api_key:
                openai.api_key = api_key
            self.embedding_dim = _OPENAI_DIM
        else:
            self.embedding_dim = _LOCAL_DIM

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        try:
            if self.use_local or self.use_google:
                return self._generate_local_embeddings(texts)
            else:
                return self._generate_openai_embeddings(texts)
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            logger.info("Falling back to local embeddings")
            self.use_local = True
            self.embedding_dim = _LOCAL_DIM
            return self._generate_local_embeddings(texts)

    def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=texts,
        )
        return [item.embedding for item in response.data]

    def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        model = get_local_model()
        embeddings = model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        return self.embedding_dim
