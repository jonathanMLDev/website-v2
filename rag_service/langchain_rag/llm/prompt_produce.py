"""
Prompt Production Module

Generates prompts and system prompts for different LLM processing types.
"""

from typing import Any, List, Tuple

from langchain_core.documents import Document

from .base_agent import BaseAgent


class PromptProducer:
    """Produces prompts and system prompts for LLM processing"""

    def __init__(self, base_agent: BaseAgent):
        """
        Initialize prompt producer

        Args:
            base_agent: BaseAgent instance with category descriptions and known libraries
        """
        self.agent = base_agent
        self.category_descriptions_text = "\n".join([
            f"- {cat}: {desc}"
            for cat, desc in base_agent.CATEGORY_DESCRIPTIONS.items()
        ])
        self.valid_categories = list(base_agent.CATEGORY_DESCRIPTIONS.keys())
        self.known_libraries_text = ', '.join(base_agent.KNOWN_LIBRARIES)

    def produce_prompt(
        self,
        process_type: str,
        process_data: Any,
        topic: str = None,
    ) -> Tuple[str, str]:
        """
        Produce prompt and system prompt for given process type

        Args:
            process_type: Type of process (classify, sentiment, extract_libraries, etc.)
            process_data: Data to process (Document, List[Document], str, etc.)

        Returns:
            Tuple of (prompt, system_prompt)
        """
        if process_type == 'classify':
            return self._produce_classify_prompt(process_data)
        elif process_type == 'sentiment':
            return self._produce_sentiment_prompt(process_data)
        elif process_type in ['libraries', 'extract_libraries']:
            return self._produce_libraries_prompt(process_data)
        elif process_type == 'extract_all_meta':
            return self._produce_extract_all_meta_prompt(process_data)
        elif process_type == 'extract_topics':
            return self._produce_extract_topics_prompt(process_data)
        elif process_type == 'summarize_topic_chronologically':
            return self._produce_summarize_topic_chronologically_prompt(
                process_data, topic
            )
        elif process_type == 'total_summarize':
            return self._produce_total_summarize_prompt(process_data)
        else:
            raise ValueError(f"Unknown process type: {process_type}")

    def _produce_classify_prompt(self, document: Document) -> Tuple[str, str]:
        """Produce prompt for classification"""
        subject = document.metadata.get('subject', '')
        content = document.page_content
        parent_context = document.metadata.get('parent_content', '')
        text = f"Subject: {subject}\n\nContent: {content[:3000]}"
        if parent_context:
            text = f"{text}\n\nParent Context: {parent_context[:1000]}"

        prompt = f"""Analyze this Boost C++ library mailing list email and extract up to 3 categories with the highest confidence scores.

Task:
- Identify which categories best describe this email
- Categories must be from the following options only
- Assign confidence scores between 0 and 1 (higher = more confident)
- Return only the top 3 categories with highest confidence

Available categories:
{self.category_descriptions_text}

Email:
{text}

Instructions:
- Select categories that best match the email content
- Confidence scores should reflect how well the category applies (0.0 = not applicable, 1.0 = perfectly matches)
- Only include categories with confidence >= 0.5
- Return up to 3 categories, ordered by confidence (highest first)

Respond with JSON format:
{{
  "Bug": 0.9,
  "Documentation": 0.8,
  "Performance": 0.7
}}

If no categories match well (all confidences < 0.5), return an empty object {{}}."""

        system_prompt = (
            "You are a technical classifier for Boost C++ mailing list emails. "
            "Analyze emails and categorize them accurately. "
            "Respond only with valid JSON containing category names as keys "
            "and confidence scores (0-1) as values."
        )
        return prompt, system_prompt

    def _produce_sentiment_prompt(self, document: Document) -> Tuple[str, str]:
        """Produce prompt for sentiment extraction"""
        subject = document.metadata.get('subject', '')
        content = document.page_content
        parent_context = document.metadata.get('parent_content', '')
        text = f"Subject: {subject}\n\nContent: {content[:3000]}"
        if parent_context:
            text = f"{text}\n\nParent Context: {parent_context[:1000]}"

        prompt = f"""Analyze the sentiment of this Boost C++ mailing list email and classify it into ONE of the following categories.

Task:
- Determine the overall sentiment/tone of the email
- Choose exactly ONE category that best represents the sentiment
- Assign a confidence score between 0 and 1

Sentiment Categories:
- Urgent: Critical issues, security problems, data loss,
  time-sensitive problems requiring immediate attention
- Negative: Bugs, errors, complaints, frustration, criticism, problems
- Positive: Thanks, appreciation, solutions, praise, helpful responses,
  successful outcomes
- Neutral: Questions, discussions, neutral tone, informational content,
  general inquiries

Email:
{text}

Instructions:
- Analyze the tone, language, and context of the email
- Consider both explicit statements and implied sentiment
- Choose the category that best represents the overall sentiment
- Confidence score should reflect certainty (0.0 = uncertain, 1.0 = very certain)
- If truly neutral (no strong positive/negative/urgent tone), choose Neutral

Respond with JSON format:
{{
  "Positive": 0.85
}}

Or:
{{
  "Neutral": 0.7
}}

Or:
{{
  "Urgent": 0.95
}}"""

        system_prompt = (
            "You are a sentiment analyzer for Boost C++ mailing list emails. "
            "Classify the sentiment accurately based on tone, language, "
            "and context. Respond only with valid JSON containing one "
            "sentiment label as key and confidence score (0-1) as value."
        )
        return prompt, system_prompt

    def _produce_libraries_prompt(self, document: Document) -> Tuple[str, str]:
        """Produce prompt for library extraction"""
        subject = document.metadata.get('subject', '')
        content = document.page_content
        parent_context = document.metadata.get('parent_content', '')
        text = f"Subject: {subject}\n\nContent: {content[:3000]}"
        if parent_context:
            text = f"{text}\n\nParent Context: {parent_context[:1000]}"

        prompt = f"""Analyze this Boost C++ library mailing list email and identify which Boost C++ libraries are mentioned, discussed, or relevant to the content.

Task:
- Identify all Boost C++ libraries mentioned or discussed in the email
- Libraries must be from the known Boost libraries list only
- Assign confidence scores between 0 and 1 for each library
- Only include libraries that are actually mentioned or clearly relevant

Known Boost C++ Libraries:
{self.known_libraries_text}

Email:
{text}

Instructions:
- Look for explicit library names (e.g., "Boost.Asio", "Asio", "boost::asio")
- Consider context - if the email discusses features specific to a library, include it
- Confidence scores should reflect how clearly the library is mentioned:
  - 0.9-1.0: Explicitly named and central to discussion
  - 0.7-0.8: Explicitly named but peripheral
  - 0.5-0.6: Implied or strongly suggested by context
- Only include libraries with confidence >= 0.5
- If no libraries are mentioned or discussed, return an empty object {{}}

Respond with JSON format:
{{
    "Asio": 0.92,
    "Beast": 0.65,
    "unordered": 0.55
}}

If no libraries are mentioned:
{{}}"""

        system_prompt = (
            "You extract Boost C++ library names from mailing list emails. "
            "Identify libraries that are mentioned, discussed, or relevant "
            "to the email content. Respond only with valid JSON containing "
            "library names as keys and confidence scores (0-1) as values."
        )
        return prompt, system_prompt

    def _produce_extract_all_meta_prompt(self, document: Document) -> Tuple[str, str]:
        """Produce prompt for extracting all metadata"""
        subject = document.metadata.get('subject', '')
        content = document.page_content
        parent_context = document.metadata.get('parent_content', '')
        text = f"Subject: {subject}\n\nContent: {content[:3000]}"
        if parent_context:
            text = f"{text}\n\nParent Context: {parent_context[:1000]}"

        prompt = f"""Analyze this Boost C++ library mailing list email and extract all metadata with confidence scores.

Task:
Extract three types of metadata from the email:
1. Categories (1-3 categories)
2. Sentiment (exactly one)
3. Libraries (all mentioned libraries)

Email:
{text}

1. Categories:
Choose 1-3 categories from the following options that best describe the email:
{self.category_descriptions_text}

Instructions:
- Select categories that best match the email content
- Provide confidence scores between 0 and 1
- Only include categories with confidence >= 0.5
- Order by confidence (highest first)

2. Sentiment:
Choose exactly ONE sentiment category:
- Urgent: Critical issues, security problems, data loss
- Negative: Bugs, errors, complaints, frustration
- Positive: Thanks, appreciation, solutions, praise
- Neutral: Questions, discussions, neutral tone

Instructions:
- Determine the overall sentiment/tone
- Provide confidence score between 0 and 1

3. Libraries:
Identify Boost C++ libraries mentioned or discussed.
Known Boost libraries: {self.known_libraries_text}

Instructions:
- Include all libraries explicitly mentioned or clearly relevant
- Provide confidence scores between 0 and 1
- Only include libraries with confidence >= 0.5
- If no specific library is mentioned, return empty array []

Respond with JSON format:
{{
  "categories": [
    {{"Bug": 0.87}},
    {{"Documentation": 0.72}}
  ],
  "sentiment": {{"Neutral": 0.65}},
  "libraries": [
    {{"Asio": 0.9}},
    {{"Beast": 0.65}}
  ]
}}

Note: Each value in categories and libraries arrays should be a
dictionary with one key-value pair (category/library name as key,
confidence as value)."""

        system_prompt = (
            "You extract comprehensive metadata (categories, sentiment, "
            "libraries) from Boost C++ mailing list emails. Analyze the "
            "email thoroughly and provide accurate classifications with "
            "confidence scores. Respond only with valid JSON."
        )
        return prompt, system_prompt

    def _produce_extract_topics_prompt(self, documents: List[Document]) -> Tuple[str, str]:
        """Produce prompt for topic extraction"""
        # Limit documents and content length to avoid token limits
        max_docs = min(20, len(documents))
        combined_content = "\n\n---\n\n".join([
            f"Subject: {doc.metadata.get('subject', 'No Subject')}\n"
            f"Content: {doc.page_content[:800]}\n"
            f"URL: {doc.metadata.get('url', 'N/A')}"
            for doc in documents[:max_docs]
        ])

        prompt = f"""Analyze these Boost C++ mailing list discussions and identify the main topics being discussed.

Task:
- Identify 3-5 distinct main topics from the discussions
- Each topic should be specific and meaningful
- Extract key assertions/points for each topic
- Include reference URLs for each assertion

Discussions ({len(documents[:max_docs])} emails):
{combined_content}

Instructions:
- Topics should be specific (e.g., "Boost.Asio HTTP performance
  optimization" not just "performance")
- Each topic should represent a distinct discussion thread or theme
- Assertions should capture key points, questions, solutions, or
  decisions related to the topic
- Include URLs from the emails that discuss each assertion
- Group related discussions under the same topic
- Avoid generic topics - be specific to Boost C++ libraries and
  technical discussions

Return a JSON object with this structure:
{{
    "topics": [
        {{
            "subject": (
                "Specific topic subject "
                "(e.g., 'Boost.Asio async operation cancellation')"
            ),
            "assertions": [
                {{
                    "content": (
                        "Clear description of a key point, question, "
                        "solution, or decision related to this topic"
                    ),
                    "reference url": ["url1", "url2"]
                }},
                {{
                    "content": "Another key assertion for this topic",
                    "reference url": ["url3"]
                }}
            ]
        }},
        {{
            "subject": "Another distinct topic",
            "assertions": [
                {{
                    "content": "Key assertion for this topic",
                    "reference url": ["url4", "url5"]
                }}
            ]
        }}
    ]
}}

Important:
- Return 3-5 topics (fewer if there aren't enough distinct topics)
- Each topic should have at least 1 assertion
- Each assertion should have at least 1 reference URL
- Topics should be distinct and non-overlapping
- Be specific and technical in topic subjects"""

        system_prompt = (
            "You are a technical analyst extracting main topics from "
            "Boost C++ mailing list discussions. Identify distinct, "
            "specific topics and their key assertions with proper URL "
            "references. Respond only with valid JSON."
        )
        return prompt, system_prompt

    def _produce_summarize_topic_chronologically_prompt(
        self,
        documents: List[Document],
        topic: str,
    ) -> Tuple[str, str]:
        """Produce prompt for chronological topic summarization"""
        from datetime import datetime

        # Sort documents by date to ensure chronological order
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get('date', 0)
        )

        text = ""
        for doc in sorted_docs:
            metadata = doc.metadata
            date_timestamp = metadata.get('date', 0)
            if isinstance(date_timestamp, (int, float)) and date_timestamp > 0:
                current_date = datetime.fromtimestamp(date_timestamp)
                date_str = current_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = "Unknown date"

            subject = metadata.get('subject', 'No Subject')
            content = doc.page_content[:1000]  # Limit content length
            url = metadata.get('url', '')

            text += f"""
[{date_str}] {subject}
{content}
URL: {url}
---
"""

        prompt = f"""Summarize these Boost C++ mailing list discussions
about the topic: "{topic}"

The discussions are ordered chronologically. Provide a chronological
summary showing how the topic evolved over time.

Focus on:
- Initial question or issue raised
- Key responses and discussion points
- Solutions or workarounds proposed
- Resolution or current status
- Important decisions or conclusions reached

Group related discussions by time periods or key events. Each entry
should represent a significant point in the discussion timeline.

Discussions (in chronological order):
{text}

Provide a chronological summary in JSON format. The summary should be
a list of entries, each representing a significant point in the
discussion timeline.

Respond with JSON format:
{{
    "chronological_summary": [
        {{
            "Date": "YYYY-MM-DD",
            "summary": "A clear summary of what happened at this point in the discussion, including key points, questions, answers, or decisions.",
            "reference url": ["url1", "url2"]
        }},
        {{
            "Date": "YYYY-MM-DD",
            "summary": "Next significant point in the discussion timeline...",
            "reference url": ["url3"]
        }}
    ]
}}

Important:
- Use actual dates from the discussions (YYYY-MM-DD format)
- Each summary should be clear and concise (2-4 sentences)
- Include main relevant URLs from the discussions in that time period
- Order entries chronologically from earliest to latest
- Focus on the evolution and progression of the topic"""

        system_prompt = (
            "You are a technical writer summarizing Boost C++ mailing "
            "list discussions chronologically. Be factual, clear, and "
            "reference specific discussions with URLs. "
            "Respond only with valid JSON."
        )
        return prompt, system_prompt

    def _produce_total_summarize_prompt(self, documents: List[Document]) -> Tuple[str, str]:
        """Produce prompt for total summarization"""
        # Limit content length to avoid token limits
        text = ""
        for idx, doc in enumerate(documents[:15]):  # Limit to 15 documents
            subject = doc.metadata.get('subject', 'No Subject')
            content = doc.page_content[:1000]  # Limit content per document
            url = doc.metadata.get('url', '')
            text += f"""
Document {idx+1}:
Subject: {subject}
Content: {content}
URL: {url}
---
"""

        prompt = f"""Analyze these Boost C++ mailing list documents and create a comprehensive summary.

Task:
1. Extract a main subject that summarizes the overall topic of these discussions
2. Identify key assertions (main points, questions, solutions, decisions) related to the subject
3. Link each assertion to the relevant email URLs

Documents ({len(documents[:15])} emails):
{text}

Instructions:
- Main subject should be a clear, concise one-sentence summary of what these discussions are about
- Subject should be specific (e.g., "Discussion about Boost.Asio async operation cancellation mechanisms" not just "Asio discussion")
- Assertions should capture:
  * Key questions raised
  * Important points made
  * Solutions or workarounds proposed
  * Decisions reached
  * Technical details discussed
- Each assertion should be clear and self-contained
- Include all relevant URLs for each assertion (emails that discuss that point)
- Group related points logically
- Aim for 3-8 assertions depending on the complexity of the discussion

Return a JSON object with this structure:
{{
  "subject": "Clear one-sentence summary of the main topic discussed across all documents",
  "assertions": [
    {{
        "content": "Description of a key assertion, question, solution, or decision (2-3 sentences)",
        "reference url": ["url1", "url2"]
    }},
    {{
        "content": "Another key assertion with clear description",
        "reference url": ["url3", "url4"]
    }},
    {{
        "content": "Additional important point from the discussions",
        "reference url": ["url5"]
    }}
  ]
}}

Important:
- Subject must be specific and descriptive
- Each assertion should be meaningful and distinct
- Include all relevant URLs for each assertion
- Assertions should be ordered logically (chronologically or by importance)
- Be factual and accurate to the source material"""

        system_prompt = (
            "You are a technical summarizer for Boost C++ mailing list "
            "discussions. Extract the main subject and key assertions "
            "with proper URL references. Be accurate, clear, and "
            "comprehensive. Respond only with valid JSON."
        )
        return prompt, system_prompt
