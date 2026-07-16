from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from .chunks import load_chunk_lookup
from .config import RagConfig, RagGraphState
from .generation import TransformersLocalLLM, load_model
from .prompts import build_prompt
from .retrieval import Hoi4QdrantRetriever


class Hoi4RagPipeline:
    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.embedder = SentenceTransformer(config.embedding_model)
        self.chunk_lookup = load_chunk_lookup(config.chunks_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.model = load_model(config.model_path)
        self.retriever = Hoi4QdrantRetriever(
            embedder=self.embedder,
            qdrant_url=config.qdrant_url,
            collection_name=config.collection_name,
            chunk_lookup=self.chunk_lookup,
            chunks_path=config.chunks_path,
            embedding_model_name=config.embedding_model,
            top_k=config.top_k,
            candidate_multiplier=config.candidate_multiplier,
        )
        self.llm = TransformersLocalLLM(
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=config.max_new_tokens,
        )
        self.graph = self._build_graph()

    def ask(self, question: str) -> RagGraphState:
        return self.graph.invoke({"question": question})

    def _build_graph(self):
        graph = StateGraph(RagGraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("prompt", self._build_prompt)
        graph.add_node("generate", self._generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "prompt")
        graph.add_edge("prompt", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _retrieve(self, state: RagGraphState) -> RagGraphState:
        question = state["question"]
        contexts = self.retriever.invoke(question)
        return {"contexts": contexts}

    def _build_prompt(self, state: RagGraphState) -> RagGraphState:
        prompt = build_prompt(
            self.tokenizer,
            state["question"],
            state["contexts"],
            enable_thinking=self.config.enable_thinking,
        )
        return {"prompt": prompt}

    def _generate(self, state: RagGraphState) -> RagGraphState:
        answer = self.llm.invoke(state["prompt"])
        return {"answer": answer}
