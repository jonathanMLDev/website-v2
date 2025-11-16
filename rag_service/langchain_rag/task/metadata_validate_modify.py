"""
Metadata Validation and Modification Script

Updates existing emails in ChromaDB with new metadata fields:
- libraries (array)
- categories (array)
- sentiment (single value)

Uses parent email context for better accuracy.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from typing import Dict, List, Optional, Tuple
import structlog

logger = structlog.get_logger(__name__)
from tqdm import tqdm
from langchain_core.documents import Document
from dateutil import parser
# Import RAG components
from ..rag_pipeline import LangChainRAGPipeline
from ..preprocessor import BoostDataProcessor
from ..llm import LLMHelper


class MetadataValidateModify:
    """
    Validates and modifies metadata for emails in ChromaDB
    
    Loads source email data and ChromaDB vectors, then updates missing metadata fields
    using LLM helper and optional parent email context.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        source_data_path: Optional[str] = None,
        dry_run: bool = False,
        update_properties: Optional[List[str]] = None,
        empty_update: bool = False,
        need_source_data: bool = True,
        st_time: Optional[int] = 0,
    ):
        """
        Initialize metadata validator/modifier
        
        Args:
            config_path: Path to RAG config file
            source_data_path: Path to source email data (JSON format)
            dry_run: If True, only report what would be changed without updating
            update_properties: List of properties to update
            empty_update: If True, update empty fields
            need_source_data: If True, load source email data
            st_time: Start time(timestamp)
        """
        self.dry_run = dry_run
        self.source_data_path = source_data_path
        # Which properties to update (defaults to all supported)
        default_props = ['libraries', 'categories', 'sentiment', 'parent']
        # default_props = ['sentiment']
        
        self.update_properties = [p.strip().lower() for p in (update_properties or default_props)]
        self.empty_update = empty_update
        self.need_source_data = need_source_data
        self.st_time = st_time
        # Initialize logger
        self.logger = logger.bind(component="MetadataValidateModify")
        self.logger.info("Initializing MetadataValidateModify...")
        
        # Initialize LLM helper (for categories and sentiment)
        # Use OpenAI by default, can be configured via config
        llm_backend = "openai"
        self.llm_helper = LLMHelper(backend=llm_backend)
        self.logger.info(f"LLM helper initialized with backend: {llm_backend}")
        
        # Statistics
        self.stats = {
            'total_emails': 0,
            'already_complete': 0,
            'updated': 0,
            'errors': 0,
            'parent_context_used': 0
        }
        for prop in self.update_properties:
            self.stats[f'missing_{prop}'] = 0
        
        # Cache for source emails (by URL)
        self.source_by_url: Dict[str, Document] = {}
        
        self.docs_to_modify: List[Document] = []
        self.updated_metadata: Dict[str, Dict] = {}

    def load_source_data(self) -> Dict[str, Document]:
        """
        Load source email data via BoostDataProcessor.load_emails().
        Optionally also merge a provided JSON file (backward compatible).
        
        Returns:
            Dict mapping message_id (when available) to email data
        """
        processor = BoostDataProcessor()
        emails_docs = processor.load_emails()
        count_loaded = 0
        for doc in emails_docs:
            md = doc.metadata or {}
            url = md.get('url')
            if url:
                self.source_by_url[url] = doc
            
            count_loaded += 1
        self.logger.info(f"Loaded {count_loaded} emails from BoostDataProcessor.load_emails()")
        # Return a message_id map for compatibility with existing callers
        return self.source_by_url

    def get_all_chromadb_emails(self) -> List[Document]:
        """
        Retrieve all email documents from ChromaDB
        
        Returns:
            List of Document objects
        """
        # Initialize RAG pipeline (for ChromaDB access)
        self.logger.info("Loading RAG pipeline...")
        pipeline = LangChainRAGPipeline()
        
        # Get ChromaDB collection
        self.collection = pipeline.base_retrievers['mail'].vector_store._collection
        self.logger.info(f"Connected to ChromaDB collection: {self.collection.name}")
        
        try:
            # Get all documents
            all_docs = self.collection.get(
                where={"type": "mail"}  # Only get mail documents
            )
            
            count = len(all_docs['ids'])
            
            for i in range(count):
                doc = Document(
                    id=all_docs['ids'][i],
                    page_content=all_docs['documents'][i],
                    metadata=all_docs['metadatas'][i]
                )
                self.docs_to_modify.append(doc)
                self.updated_metadata[doc.metadata['url']] = doc.metadata
            self.logger.info(f"Retrieved {count} email documents from ChromaDB")
            return self.docs_to_modify
        except Exception as e:
            self.logger.error(f"Failed to fetch documents from ChromaDB: {e}")
            raise

    def get_parent_metadata(self, parent_url: str) -> Optional[Dict]:
        """
        Get metadata of parent email by URL
        
        Args:
            parent_url: URL of parent email
        
        Returns:
            Parent email metadata or None
        """
        if not parent_url:
            return None
        
        # Look up parent by URL
        parent_metadata = self.source_by_url.get(parent_url).metadata
        return parent_metadata

    def check_metadata_completeness(self, metadata: Dict) -> Tuple[bool, List[str]]:
        """
        Check if metadata has all required fields
        
        Args:
            metadata: Email metadata dict
        
        Returns:
            (is_complete, missing_fields)
        """
        missing = []
        
        for prop in self.update_properties:
            if prop not in metadata or not metadata.get(prop):
                missing.append(prop)
        
        is_complete = len(missing) == 0
        
        return is_complete, missing

    
    def modify_date(self, date: str) -> int:
        """
        Modify date
        """
        try:
            if isinstance(date, int):
                return date
            dt= parser.parse(date)
            return int(dt.timestamp())
        except Exception as e:
            self.logger.error(f"Failed to modify date {date}: {e}")
            return None
    
    def get_new_libraries(self, metadata: Dict) -> List[str]:
        """
        Modify libraries
        """
        current_url = metadata['url']
        subject = metadata['subject']
        content = subject + "\n" + self.source_by_url[current_url].page_content
        result = self.llm_helper.process_pipeline("extract_libraries", content)
        # process_pipeline returns a list (one result per document)
        return result[0] if result and len(result) > 0 else []
        # return self.classifier._extract_libraries(subject, content)
    
    def get_new_categories(self, metadata: Dict) -> List[str]:
        """
        Modify categories
        """
        current_url = metadata['url']
        subject = metadata['subject']
        content = subject + " " + self.source_by_url[current_url].page_content
        result = self.llm_helper.process_pipeline("classify", content)
        # process_pipeline returns a list (one result per document)
        return result[0] if result and len(result) > 0 else ['Discussion']
    
    def get_new_sentiment(self, metadata: Dict) -> str:
        """
        Modify sentiment
        """
        current_url = metadata['url']
        subject = metadata['subject']
        content = subject + " " + self.source_by_url[current_url].page_content
        result = self.llm_helper.process_pipeline("sentiment", content)
        # process_pipeline returns a list (one result per document)
        return result[0] if result and len(result) > 0 else "Neutral"
        
    def process_email(
        self,
        metadata: Dict,
        missing_fields: List[str]
    ) -> Optional[Dict]:
        """
        Process single email: check completeness and update if needed
        
        Args:
            metadata: Current metadata
            missing_fields: List of missing fields
        Returns:
            Updated metadata if changed, None otherwise
        """
        
        # Classify email with context
        try:
            new_metadata = metadata.copy()
            date_timestamp = self.modify_date(metadata['date'])
            
            if "date" in missing_fields:
                new_metadata['date'] = date_timestamp
            
            if "parent" in missing_fields:
                current_url = metadata['url']
                parent_url = self.source_by_url[current_url].metadata['parent']
                if parent_url:
                    new_metadata['parent'] = parent_url
                else:
                    new_metadata['parent'] = ""
            
            if date_timestamp < self.st_time:
                for prop in missing_fields:
                    if prop not in new_metadata:
                        new_metadata[prop] = ""
                return new_metadata
                
            # Extract each field using appropriate method
            if 'libraries' in missing_fields:
                # Use LLM helper for libraries extraction
                libraries = self.get_new_libraries(metadata)
                if libraries:
                    new_metadata['libraries'] = ",".join(libraries)
            
            if 'categories' in missing_fields:
                # Use LLM helper for categories
                categories = self.get_new_categories(metadata)
                if categories:
                    new_metadata['categories'] = ",".join(categories)
            
            if 'sentiment' in missing_fields:
                # Use LLM helper for sentiment
                sentiment = self.get_new_sentiment(metadata)
                if sentiment:
                    new_metadata['sentiment'] = sentiment
            
            self.stats['updated'] += 1
            
            return new_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to process email {metadata.get('url')}: {e}")
            self.stats['errors'] += 1
            return None

    def apply_updates_to_chromadb(
        self,
        updates: List[Dict]
    ):
        """
        Apply metadata updates to ChromaDB in batches
        
        Args:
            updates: List of dicts with 'id' and 'metadata'
        """
        if not updates:
            self.logger.info("No updates to apply")
            return
        
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would update {len(updates)} emails")
            # Show sample updates
            for i, update in enumerate(updates[:5]):
                self.logger.info(f"Sample {i+1}: {update['id']}")
                self.logger.info(f"  Libraries: {update['metadata'].get('libraries')}")
                self.logger.info(f"  Categories: {update['metadata'].get('categories')}")
                self.logger.info(f"  Sentiment: {update['metadata'].get('sentiment')}")
            return
        
        if len(updates) == 1:
            update = updates[0]
            self.collection.update(
                ids=[update['id']],
                metadatas=[update['metadata']]
            )
            return
        self.logger.info(f"Applying {len(updates)} updates to ChromaDB...")
        # Update in batches
        batch_size = 100
        for i in tqdm(range(0, len(updates), batch_size), desc="Updating ChromaDB"):
            batch = updates[i:i+batch_size]
            
            ids = [item['id'] for item in batch]
            metadatas = [item['metadata'] for item in batch]
            
            try:
                self.collection.update(
                    ids=ids,
                    metadatas=metadatas
                )
            except Exception as e:
                self.logger.error(f"Failed to update batch {i//batch_size + 1}: {e}")
                self.stats['errors'] += len(batch)
        
        self.logger.info("Updates applied successfully")

    def validate_and_modify_metadata(self) -> List[Document]:
        """
        Validate and modify metadata in the docs_to_modify list
        
        Returns:
            List of Document objects
        """
        # Process each email
        self.logger.info("Processing emails...")
        updates = []
        updated_urls = []
        self.updated_metadata = {}
        for doc in tqdm(self.docs_to_modify, desc="Analyzing emails"):
            doc_id = doc.id
            metadata = doc.metadata
            current_url = metadata['url']
            self.stats['total_emails'] += 1
            
            # Check if metadata is complete
            is_complete = False
            missing_fields = self.update_properties
            try:
                if self.empty_update:
                    is_complete, missing_fields = self.check_metadata_completeness(metadata)
                
                if is_complete:
                    self.stats['already_complete'] += 1
                    continue
                
                if current_url in updated_urls:
                    updates.append({'id': doc_id, 'metadata': self.updated_metadata[current_url]})
                    self.stats['updated'] += 1
                    # Update statistics
                    for prop in missing_fields:
                        self.stats[f'missing_{prop}'] += 1
                    
                    continue
                
                self.updated_metadata[current_url] = self.process_email(
                    metadata=metadata,
                    missing_fields=missing_fields
                )
                
                if self.updated_metadata[current_url]:
                    updated_urls.append(current_url)
                    update = {
                        'id': doc_id,
                        'metadata': self.updated_metadata[current_url]
                    }
                    if update['metadata'] != metadata:
                        updates.append(update)
                        # self.apply_updates_to_chromadb([update])
            except Exception as e:
                self.logger.error(f"Failed to process email {metadata.get('url')}: {e}")
                self.stats['errors'] += 1
                return None
        
        return updates
    
    def find_in_thread(self, parent_url: str, key: str) -> str:
        """
        Find value in thread
        """
        metadata = self.updated_metadata[parent_url]
        try:
            if metadata[key] != "":
                return metadata[key]
        except KeyError:
            pass
        p_url = metadata['parent']
        if p_url:
            return self.find_in_thread(p_url, key)
        return None
        
    
    def complete_updates(self, updates: List[Document]) -> List[Document]:
        """
        Complete updates with missing fields
        """
        repaired_updates = []
        for update in updates:
            metadata = update['metadata']
            is_repaired = False
            for key in self.update_properties:
                if key not in metadata:
                    metadata[key] = None
                elif metadata[key] == "":
                    metadata[key] = None
                
                if metadata[key] == None:
                    parent_url = metadata['parent']
                    if parent_url:
                        parent_value = self.find_in_thread(
                            parent_url = parent_url,
                            key = key
                        )
                        if parent_value:
                            metadata[key] = parent_value
                            is_repaired = True
            if is_repaired:
                repaired_updates.append(update)
        return repaired_updates
    
    def run(self):
        """
        Main execution: validate and modify all email metadata
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Metadata Validation and Modification")
        self.logger.info(f"Dry run: {self.dry_run}")
        self.logger.info("=" * 80)
        
        # Load source data (fills internal caches)
        if self.need_source_data:
            self.load_source_data()
        else:
            self.source_by_url = {}
        
        # Get all ChromaDB emails
        self.docs_to_modify = self.get_all_chromadb_emails()
        # updates = self.complete_updates(self.docs_to_modify)
        
        updates = self.validate_and_modify_metadata()
        
        # import json
        # with open("data.json", "w") as f:
        #     updates = json.load(f)
        
        # with open("data.json", "r") as f:
        #     updates = json.load(f)
        updates = self.complete_updates(updates)
        
        
        # Apply updates
        self.apply_updates_to_chromadb(updates)
        
        # Print statistics
        self.print_statistics()

    def print_statistics(self):
        """Print summary statistics"""
        self.logger.info("=" * 80)
        self.logger.info("STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Total emails processed:      {self.stats['total_emails']}")
        self.logger.info(f"Already complete:            {self.stats['already_complete']}")
        for prop in self.update_properties:
            self.logger.info(f"Missing {prop}:              {self.stats[f'missing_{prop}']}")
        self.logger.info(f"Successfully updated:        {self.stats['updated']}")
        self.logger.info(f"Parent context used:         {self.stats['parent_context_used']}")
        self.logger.info(f"Errors:                      {self.stats['errors']}")
        self.logger.info("=" * 80)
        
        # Calculate completion rate
        if self.stats['total_emails'] > 0:
            completion_rate = (self.stats['already_complete'] + self.stats['updated']) / self.stats['total_emails'] * 100
            self.logger.info(f"Metadata completion rate:    {completion_rate:.1f}%")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate and modify email metadata in ChromaDB"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to RAG config file"
    )
    parser.add_argument(
        '--source-data',
        type=str,
        default=None,
        help="Path to source email data JSON file"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Perform dry run (no actual updates)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Enable verbose logging"
    )
    parser.add_argument(
        '--properties',
        type=str,
        default='date',
        help="Comma-separated list of properties to update (libraries,categories,sentiment)"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="DEBUG"
        )
    
    # Run validator/modifier
    validator = MetadataValidateModify(
        config_path=args.config,
        source_data_path=args.source_data,
        dry_run=args.dry_run,
        update_properties=[p.strip().lower() for p in args.properties.split(',') if p.strip()]
    )
    
    validator.run()


if __name__ == "__main__":
    main()

