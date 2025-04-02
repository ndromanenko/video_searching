import dspy
from src.retrieval import Retrieval

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

    def forward(self, query: str, retrieval: Retrieval):
        """
        Process the input query and retrieve relevant documents.

        Args:
            query (str): The input query to search for relevant documents.
            retrieval: The retrieval object used to search for documents.

        Returns:
            The output from the ranker generator based on the context and query.

        """
        relevant_docs = retrieval.search(query=query)

        for_ranking = [doc.page_content for doc in relevant_docs]

        # context = ""
        # for_ranking = []
        # for i, doc in enumerate(relevant_docs):
        #     context += f"{i + 1}. {doc.page_content}\n"
        #     for_ranking.append(doc.page_content)
        
        return self.ranker_gen(context=for_ranking,
                               query=query)


if __name__ == "__main__":
    retrieval = Retrieval()
    pipeline = QA()
    print(pipeline.forward())