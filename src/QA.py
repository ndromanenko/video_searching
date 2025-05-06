import re
import pandas as pd

import dspy
import streamlit as st
from langsmith import traceable

from src.retrieval import Retrieval
from src.core.logger import logger


class ParaphrasingQuery(dspy.Signature):
    "Перефразирование запроса на русском языке"
    query: str = dspy.InputField()
    paraphrase: str = dspy.OutputField(
        desc="Перефразирование не больше 10 слов"
    )

class AnswerWithContext(dspy.Signature):
    """Context ranking based on query without explanation"""
    context: str = dspy.InputField()
    query: str = dspy.InputField()
    ranked_context = str = dspy.OutputField(
        desc="List of numbers sorted by relevance, in the format:\n1. <number>\n2. <number>\n3. <number>",
        prefix="Ranked numbers:"
    )

class QA(dspy.Module):
    def __init__(self) -> None:
        """Initialize the QA module with paraphrase and ranker generators."""
        super().__init__()
        self.paraphrase_gen = dspy.ChainOfThought(ParaphrasingQuery)
        self.ranker_gen = dspy.ChainOfThought(AnswerWithContext)

    def _format_time(self, time_str: str) -> str:
        total_seconds = int(time_str.split(":")[0])
        minute = total_seconds // 60
        second = total_seconds % 60
        return f"{minute}:{second:02}"

    def _build_context(self, query: str, retrieval: Retrieval) -> tuple[list[str], dict[str, str], str]:
        similarity_content = retrieval.search(query)
        for_ranking = []
        mapping = {}
        context = ""

        for idx, doc in enumerate(similarity_content):
            time_str = doc.metadata.get("time", "0:00")
            timestamp = self._format_time(time_str)
            content = doc.page_content
            context += f"{idx + 1}. {content}\n"
            for_ranking.append(content)
            mapping[content] = f"Время: {timestamp}"

        return for_ranking, mapping, context

    def _parse_ranked_response(self, response: str, for_ranking: list[str], mapping: dict[str, str]) -> list[str]:
        if not response:
            return []

        correct_order = []
        for sentence in response.split("\n"):
            parts = sentence.split(". ")
            if len(parts) > 1 and parts[1].isdigit():
                index = int(parts[1]) - 1
                if 0 <= index < len(for_ranking):
                    content = for_ranking[index]
                    if content in mapping:
                        correct_order.append(mapping[content])
                        
        return correct_order[:3]

    # @traceable
    def forward(self, query: str) -> list[str]:
        """
        Process a query through the QA pipeline and return ranked responses.
        
        Args:
            query: The input query string to process
            
        Returns:
            A list of ranked responses with timestamps

        """
        paraphrased_query = self.paraphrase_gen.predict(query=query).paraphrase
        for_ranking, mapping, context = self._build_context(paraphrased_query, retrieval)

        max_attempts = 5
        attempt = 0

        response = self.ranker_gen.predict(context=context, query=paraphrased_query).ranked_context

        while (not response or not re.findall(r"[0-9]+", response)) and attempt < max_attempts:
            paraphrased_query = self.paraphrase_gen.predict(query=query).paraphrase
            for_ranking, mapping, context = self._build_context(paraphrased_query, retrieval)
            response = self.ranker_gen.predict(context=context, query=paraphrased_query).ranked_context
            attempt += 1

        return self._parse_ranked_response(response, for_ranking, mapping)


if __name__ == "__main__":
    retrieval = Retrieval()
    pipeline = QA()
    # print(pipeline.forward())