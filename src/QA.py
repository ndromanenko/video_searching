import dspy
from src.retrieval import Retrieval
from langsmith import traceable
import streamlit as st
import re

# class GenerateSearchQuery(dspy.Signature):
#     question: str = dspy.InputField()
#     search_query: str = dspy.OutputField()

class AnswerWithContext(dspy.Signature):
    """Context ranking based on query"""
    context: str = dspy.InputField()
    query: str = dspy.InputField()
    ranked_context: str = dspy.OutputField()

class QA(dspy.Module):
    def __init__(self) -> None:
        """
        Initialize the QA module with a context ranking generator.

        This method sets up the ranker generator used for processing queries
        and generating answers based on context.
        """
        super().__init__()
        self.ranker_gen = dspy.ChainOfThought(AnswerWithContext)

    @traceable
    def forward(self, query: str, retriever: Retrieval):
        """
        Process the input query and retrieve relevant documents.

        Args:
            query (str): The input query to search for relevant documents.
            retrieval: The retrieval object used to search for documents.

        Returns:
            The output from the ranker generator based on the context and query.

        """
        for_ranking, mapping, context = self.create_context_answer(query, retriever)

        response = self.ranker_gen(context=context, query=query).ranked_context

        st.write(response)
        # st.write(len(mapping))

        if not re.findall("[0-9]+", response):
            return "This information is not in this lecture, try another request"

        answer_index = int(response.split("\n")[0].split(". ")[0]) - 1
        return mapping[for_ranking[answer_index]]
    
    def create_context_answer(self, query, retriever):
        similarity_content = retriever.search(query)
    
        for_ranking = []
        mapping = dict()
        context = ""

        for number, doc in enumerate(similarity_content):

            time = int(doc.metadata["time"].split(":")[0])
            minute = time // 60
            second = time % 60
            context = context + f"{number + 1}. " + doc.page_content + "\n"

            if second < 10:
                for_ranking.append(doc.page_content)
                mapping[doc.page_content] = f"Время: {minute}:0{second}"
            else:
                for_ranking.append(doc.page_content)
                mapping[doc.page_content] = f"Время: {minute}:{second}"

        return for_ranking, mapping, context


if __name__ == "__main__":
    retrieval = Retrieval()
    pipeline = QA()
    # print(pipeline.forward())