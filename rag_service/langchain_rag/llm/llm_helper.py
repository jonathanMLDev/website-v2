"""
LLM Helper Module

Orchestrates LLM processing: prompt generation -> agent call -> postprocessing
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
import structlog
from tqdm import tqdm

from .base_agent import BaseAgent
from .huggingface_agent import HuggingFaceAgent
from .openai_agent import OpenAIAgent
from .prompt_produce import PromptProducer

load_dotenv()

logger = structlog.get_logger(__name__)


class LLMHelper:
    """Helper for LLM-powered classification and extraction"""

    def __init__(
        self,
        agent: Optional[BaseAgent] = None,
        config: Optional[Any] = None,
        backend: str = "openai",  # "huggingface", "openai"
        batch_size: int = 5,
    ):
        """
        Initialize LLM helper

        Args:
            agent: Optional BaseAgent instance (if provided, other args are ignored)
            config: Config object (required if agent not provided,
                will be used to get model, temperature, max_tokens)
            backend: Backend to use ("openai" or "huggingface")
            batch_size: Batch size for processing documents
        """
        self.logger = logger.bind(component="LLMHelper")
        self.batch_size = batch_size

        # Use provided agent or create one based on backend
        if agent:
            self.agent = agent
            self.logger.info("Using provided LLM agent")
        elif backend.lower() == "openai":
            if not config:
                raise ValueError("config is required when using OpenAI backend")
            host_url = os.getenv("OPENAI_HOST_URL", "https://openrouter.ai/api/v1")
            api_key = os.getenv("OPENROUTER_API_KEY")
            self.agent = OpenAIAgent(config=config, api_key=api_key, host_url=host_url)
            # Log the model that was actually used
            actual_model = (
                config.openai_model
                if hasattr(config, "openai_model")
                else "gpt-3.5-turbo"
            )
            self.logger.info(f"Initialized OpenAI agent with model: {actual_model}")
        elif backend.lower() == "huggingface":
            if not config:
                raise ValueError("config is required when using HuggingFace backend")
            self.agent = HuggingFaceAgent(config=config)
            actual_model = (
                config.embedding_model
                if hasattr(config, "embedding_model")
                else "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.logger.info(
                "Initialized HuggingFace agent with model: %s", actual_model
            )
        else:
            raise ValueError(
                f"Unknown backend: {backend}. Use 'openai' or 'huggingface'"
            )
        # Initialize prompt producer
        self.prompt_producer = PromptProducer(self.agent)

    def process_pipeline(
        self,
        process_type: str,
        process_data: Any,
        topic: str = None,
        batch_size: int = None,
    ) -> Any:
        """
        Process data using the LLM agent

        Flow: prompt generation -> agent call -> postprocessing

        Args:
            process_type: Type of process to perform
            process_data: Data to process
            batch_size: Batch size (ignored for sequential processing)

        Returns:
            Processed result
        """
        batch_size = batch_size or self.batch_size
        documents = self.convert_to_documents(process_data)

        if process_type == "classify":
            return self._individual_process_from_documents(
                self._process_classify, documents
            )
        elif process_type == "sentiment":
            return self._individual_process_from_documents(
                self._process_sentiment, documents
            )
        elif process_type in ["libraries", "extract_libraries"]:
            return self._individual_process_from_documents(
                self._process_libraries, documents
            )
        elif process_type == "extract_all_meta":
            return self._individual_process_from_documents(
                self._process_extract_all_meta, documents
            )
        elif process_type == "extract_topics":
            return self._process_extract_topics(documents)
        elif process_type == "total_summarize":
            return self._process_total_summarize(documents)
        elif process_type == "chronological_summary":
            return self._process_summarize_chronologically(documents, topic)
        else:
            raise ValueError(f"Unknown process type: {process_type}")

    def convert_to_documents(self, process_data: Any) -> List[Document]:
        """Convert data to documents"""
        if not isinstance(process_data, list):
            process_data = [process_data]

        documents = []
        for item in process_data:
            if isinstance(item, Document):
                documents.append(item)
            elif isinstance(item, dict):
                content = item.get("content") or item.get("text", "")
                metadata = item.get("metadata", item)
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            elif isinstance(item, str):
                doc = Document(page_content=item)
                documents.append(doc)
            else:
                raise ValueError(f"Unknown data type: {type(item)}")
        return documents

    def get_model_info(self) -> Dict[str, str]:
        """Return basic information about the underlying LLM model."""
        if not hasattr(self, "agent") or self.agent is None:
            return {"model_type": "unknown", "model_name": "unknown"}

        model_type = self.agent.__class__.__name__
        model_name = getattr(self.agent, "model", None) or getattr(
            self.agent, "text_generator_model", None
        )
        if not model_name and hasattr(self.agent, "config"):
            model_name = getattr(self.agent.config, "openai_model", None)

        return {
            "model_type": model_type or "unknown",
            "model_name": model_name or "unknown",
        }

    def _individual_process_from_documents(
        self, function: Callable, documents: List[Document]
    ) -> Any:
        """Process individual document using a function"""
        results = []
        for doc in tqdm(documents, desc="Processing documents"):
            try:
                result = function(doc)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in {function.__name__}: {e}")
                results.append([])
        return results

    def _process_classify(self, document: Document) -> str:
        """Process classification"""
        prompt, system_prompt = self.prompt_producer.produce_prompt(
            "classify", document
        )
        raw_result = self.agent.run_llm(prompt, system_prompt, max_tokens=200)
        result = self._postprocess_classify(raw_result)
        return result

    def _process_sentiment(self, document: Document) -> str:
        """Process sentiment extraction"""
        prompt, system_prompt = self.prompt_producer.produce_prompt(
            "sentiment", document
        )
        raw_result = self.agent.run_llm(prompt, system_prompt, max_tokens=200)
        result = self._postprocess_sentiment(raw_result)
        return result

    def _process_libraries(self, document: Document) -> List[str]:
        """Process library extraction"""
        prompt, system_prompt = self.prompt_producer.produce_prompt(
            "extract_libraries", document
        )
        raw_result = self.agent.run_llm(prompt, system_prompt, max_tokens=240)
        result = self._postprocess_libraries(raw_result)
        return result

    def _process_extract_all_meta(self, document: Document) -> Dict[str, Any]:
        """Process extract all metadata"""
        prompt, system_prompt = self.prompt_producer.produce_prompt(
            "extract_all_meta", document
        )
        raw_result = self.agent.run_llm(prompt, system_prompt, max_tokens=350)
        result = self._postprocess_extract_all_meta(raw_result)
        return result

    def _process_extract_topics(self, documents: List[Document]) -> List[str]:
        """Process topic extraction"""
        if not documents:
            return []
        try:
            prompt, system_prompt = self.prompt_producer.produce_prompt(
                "extract_topics", documents
            )
            raw_result = self.agent.run_llm(prompt, system_prompt, max_tokens=500)
            result = self._postprocess_extract_topics(raw_result)
            return result
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            return []

    def _process_summarize_chronologically(
        self, documents: List[Document], topic: str
    ) -> Dict:
        """Process chronological topic summarization"""
        try:
            prompt, system_prompt = self.prompt_producer.produce_prompt(
                "summarize_topic_chronologically", documents, topic
            )
            raw_result = self.agent.run_llm(prompt, system_prompt, max_tokens=800)
            result = self._postprocess_summarize_topic_chronologically(raw_result)
            return result
        except Exception as e:
            self.logger.error(f"Error summarizing topic chronologically: {e}")
            return []

    def _process_total_summarize(self, documents: List[Document]) -> Dict[str, Any]:
        """Process total summarization"""
        try:
            prompt, system_prompt = self.prompt_producer.produce_prompt(
                "total_summarize", documents
            )
            raw_result = self.agent.run_llm(prompt, system_prompt, max_tokens=1500)
            result = self._postprocess_total_summarize(raw_result)
            return result
        except Exception as e:
            self.logger.error(f"Error in total summarization: {e}")
            return None

    # Postprocessing methods
    def _postprocess_classify(self, raw_result: Dict[str, Any]) -> List[str]:
        """Postprocess classification result"""
        if not raw_result:
            return ["Discussion"]

        categories = []
        valid_categories = self.prompt_producer.valid_categories

        for key, value in raw_result.items():
            if key not in valid_categories:
                continue
            try:
                confidence = float(value)
            except (ValueError, TypeError):
                continue
            if confidence < 0.7:
                break
            categories.append(key)

        return categories if categories else ["Discussion"]

    def _postprocess_sentiment(self, raw_result: Dict[str, Any]) -> str:
        """Postprocess sentiment result"""
        if not raw_result:
            return "Neutral"

        try:
            key = list(raw_result.keys())[0]
            value = raw_result[key]
            confidence = float(value)
            if confidence < 0.7:
                return "Neutral"
            return key
        except (IndexError, KeyError, ValueError, TypeError):
            return "Neutral"

    def _postprocess_libraries(self, raw_result: Dict[str, Any]) -> List[str]:
        """Postprocess libraries result"""
        if not raw_result:
            return []

        libraries = []
        known_libraries = self.agent.KNOWN_LIBRARIES

        for key, value in raw_result.items():
            try:
                confidence = float(value)
            except (ValueError, TypeError):
                continue

            if confidence > 0.9:
                libraries.append(key)
                continue

            if key not in known_libraries:
                continue

            if confidence < 0.7:
                continue

            libraries.append(key)

        return libraries

    def _postprocess_extract_all_meta(
        self, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Postprocess extract all metadata result"""
        if not raw_result:
            return {
                "categories": ["Discussion"],
                "sentiment": "Neutral",
                "libraries": ["General"],
            }

        categories = self._parse_categories_from_meta(raw_result)
        sentiment_label = self._parse_sentiment_from_meta(raw_result)
        libraries, library_confidence = self._parse_libraries_from_meta(raw_result)

        return {
            "categories": categories,
            "sentiment": sentiment_label,
            "libraries": libraries,
            "library_confidence": library_confidence,
        }

    def _parse_categories_from_meta(self, raw_result: Dict[str, Any]) -> List[str]:
        """Parse categories from raw metadata result"""
        raw_categories = raw_result.get("categories", [])
        if isinstance(raw_categories, dict):
            raw_categories = [raw_categories]
        if not isinstance(raw_categories, list):
            raw_categories = [raw_categories]

        category_confidence = {}
        valid_categories = self.prompt_producer.valid_categories

        for entry in raw_categories:
            if isinstance(entry, dict):
                label = entry.get("label") or entry.get("name") or entry.get("category")
                conf = entry.get("confidence", entry.get("score", 0.0))
            else:
                label = entry
                conf = 0.0

            if label is None:
                continue

            label_str = str(label).strip()
            if label_str not in valid_categories:
                continue

            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0

            conf = max(0.0, min(1.0, conf))
            if (
                label_str not in category_confidence
                or conf > category_confidence[label_str]
            ):
                category_confidence[label_str] = conf

        categories = sorted(
            category_confidence, key=category_confidence.get, reverse=True
        )[:3]
        return categories if categories else ["Discussion"]

    def _parse_sentiment_from_meta(self, raw_result: Dict[str, Any]) -> str:
        """Parse sentiment from raw metadata result"""
        sentiment_data = raw_result.get("sentiment", {})
        if isinstance(sentiment_data, dict):
            sentiment_label = sentiment_data.get("label", "Neutral")
            sentiment_confidence = sentiment_data.get(
                "confidence", sentiment_data.get("score", 0.0)
            )
        else:
            valid_sentiments = ["Positive", "Negative", "Neutral", "Urgent"]
            sentiment_label = (
                sentiment_data if sentiment_data in valid_sentiments else "Neutral"
            )
            sentiment_confidence = 0.0

        if sentiment_label not in ["Positive", "Negative", "Neutral", "Urgent"]:
            sentiment_label = "Neutral"

        try:
            sentiment_confidence = float(sentiment_confidence)
        except (TypeError, ValueError):
            sentiment_confidence = 0.0
        sentiment_confidence = max(0.0, min(1.0, sentiment_confidence))

        return sentiment_label

    def _parse_libraries_from_meta(
        self, raw_result: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Parse libraries from raw metadata result"""
        raw_libraries = raw_result.get("libraries", [])
        if isinstance(raw_libraries, dict):
            raw_libraries = [raw_libraries]
        if not isinstance(raw_libraries, list):
            raw_libraries = [raw_libraries]

        libraries = []
        library_confidence = {}
        known_libraries = self.agent.KNOWN_LIBRARIES

        for entry in raw_libraries:
            if isinstance(entry, dict):
                label = entry.get("label") or entry.get("name") or entry.get("library")
                conf = entry.get("confidence", entry.get("score", 0.0))
            else:
                label = entry
                conf = 0.0

            if label is None:
                continue

            label_str = str(label).strip()
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = 0.0

            conf = max(0.0, min(1.0, conf))

            if label_str == "General":
                library_confidence["General"] = max(
                    library_confidence.get("General", 0.0), conf
                )
                libraries.append("General")
                continue

            matched_name = self._match_library_name(label_str, known_libraries)
            if matched_name:
                library_confidence[matched_name] = max(
                    library_confidence.get(matched_name, 0.0), conf
                )
                libraries.append(matched_name)

        libraries = self._deduplicate_and_clean_libraries(libraries, library_confidence)
        return libraries, library_confidence

    def _match_library_name(
        self, label_str: str, known_libraries: List[str]
    ) -> Optional[str]:
        """Match a library name against known libraries (case-insensitive)"""
        if label_str in known_libraries:
            return label_str

        lower = label_str.lower()
        for known_lib in known_libraries:
            if known_lib.lower() == lower:
                return known_lib

        return None

    def _deduplicate_and_clean_libraries(
        self, libraries: List[str], library_confidence: Dict[str, float]
    ) -> List[str]:
        """Deduplicate libraries and remove 'General' if other libraries exist"""
        if not libraries:
            return ["General"]

        seen_libs = set()
        deduped = []
        for lib in libraries:
            if lib not in seen_libs:
                deduped.append(lib)
                seen_libs.add(lib)

        if "General" in deduped and len(deduped) > 1:
            deduped = [lib for lib in deduped if lib != "General"]
            library_confidence.pop("General", None)

        return deduped

    def _postprocess_extract_topics(self, raw_result: Dict[str, Any]) -> List[str]:
        """Postprocess extract topics result"""
        topics = raw_result.get("topics", [])
        return topics[:10] if topics else []

    def _postprocess_summarize_topic_chronologically(
        self, raw_result: Dict[str, Any]
    ) -> List:
        """Postprocess chronological summarization result"""
        summary = raw_result.get("chronological_summary", [])
        return summary

    def _postprocess_total_summarize(
        self, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Postprocess total summarization result"""
        if not raw_result:
            return None
        if "subject" not in raw_result:
            return None
        if raw_result["subject"] is None:
            return None
        if raw_result["subject"] == "":
            return None

        return raw_result
