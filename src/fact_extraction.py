"""
Atomic Fact Extraction Module
Extracts atomic (individual) facts from LLM-generated text using LangChain + OpenAI.
Falls back to rule-based extraction if no API key is available.
"""

import os
import re
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class AtomicFactExtractor:
    """
    Extracts atomic facts from text using LLM-based or rule-based methods.
    
    An atomic fact is a single, self-contained claim that can be independently verified.
    Example: "Einstein was born in Germany" is one atomic fact.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the extractor.
        
        Args:
            model_name: OpenAI model to use for extraction.
        """
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = None

        if self.api_key and self.api_key != "your_openai_api_key_here":
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=0,
                    openai_api_key=self.api_key,
                )
                logger.info(f"AtomicFactExtractor initialized with LLM: {self.model_name}")
            except ImportError:
                logger.warning("langchain-openai not installed. Using rule-based extraction.")
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}. Using rule-based extraction.")
        else:
            logger.info("No OpenAI API key found. Using rule-based extraction.")

    def extract(self, text: str) -> List[str]:
        """
        Extract atomic facts from the given text.
        
        Args:
            text: Input text (typically an LLM response) to decompose.
            
        Returns:
            List of atomic fact strings.
        """
        if self.llm:
            return self._extract_with_llm(text)
        else:
            return self._extract_rule_based(text)

    def _extract_with_llm(self, text: str) -> List[str]:
        """Use LLM (via LangChain) to extract atomic facts."""
        prompt = f"""Break down the following text into a list of atomic facts.
Each atomic fact should be a single, self-contained claim that can be independently verified.
Return ONLY the list of facts, one per line, with no numbering or bullet points.

Text: \"{text}\"

Atomic Facts:"""

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            facts = [
                line.strip().lstrip("•-0123456789.) ")
                for line in content.split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            logger.info(f"Extracted {len(facts)} facts via LLM.")
            return facts
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}. Falling back to rule-based.")
            return self._extract_rule_based(text)

    def _extract_rule_based(self, text: str) -> List[str]:
        """
        Rule-based fallback: split text into sentences and treat each as a fact.
        Handles common abbreviations to avoid bad splits.
        """
        # Handle common abbreviations that contain periods
        abbreviations = ["Mr.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.", "vs.", "e.g.", "i.e.", "etc."]
        temp_text = text
        for abbr in abbreviations:
            temp_text = temp_text.replace(abbr, abbr.replace(".", "<DOT>"))

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', temp_text)

        facts = []
        for sentence in sentences:
            sentence = sentence.replace("<DOT>", ".").strip()
            # Filter out very short or non-informative sentences
            if len(sentence) > 10 and any(c.isalpha() for c in sentence):
                # Remove trailing punctuation for cleaner facts
                fact = sentence.rstrip(".!?").strip()
                if fact:
                    facts.append(fact)

        logger.info(f"Extracted {len(facts)} facts via rule-based method.")
        return facts

    def extract_with_confidence(self, text: str) -> List[dict]:
        """
        Extract facts with a confidence indicator.
        
        Returns:
            List of dicts: [{"fact": str, "method": str, "confidence": float}]
        """
        facts = self.extract(text)
        method = "llm" if self.llm else "rule_based"
        confidence = 0.85 if method == "llm" else 0.60

        return [
            {"fact": fact, "method": method, "confidence": confidence}
            for fact in facts
        ]
