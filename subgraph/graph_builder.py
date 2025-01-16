from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from typing import Any, Literal, TypedDict, cast
import os
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import Send


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import logging
from utils.utils import config

load_dotenv()

logger = logging.getLogger(__name__)

# Vector store configuration
VECTORSTORE_COLLECTION = config["retriever"]["collection_name"]
VECTORSTORE_DIRECTORY = config["retriever"]["directory"]
TOP_K = config["retriever"]["top_k"]
TOP_K_COMPRESSION = config["retriever"]["top_k_compression"]
ENSEMBLE_WEIGHTS = config["retriever"]["ensemble_weights"]
COHERE_RERANK_MODEL = config["retriever"]["cohere_rerank_model"]

def _setup_vectorstore() -> FAISS:
    """
    Set up and return the FAISS vector store instance.
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Load existing FAISS index if it exists
        save_path = os.path.join(VECTORSTORE_DIRECTORY, VECTORSTORE_COLLECTION)
        
        if os.path.exists(save_path):
            logger.info(f"Loading existing FAISS index from {save_path}")
            vectorstore = FAISS.load_local(
                save_path,
                embeddings,
                allow_dangerous_deserialization = True
            )
        else:
            logger.info("Creating new FAISS index")
            # Create an empty FAISS index
            vectorstore = FAISS.from_documents(
                [],  # Empty document list to initialize
                embeddings
            )
            
        logger.info(f"Successfully loaded vectorstore")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error setting up vectorstore: {e}")
        raise



def _load_documents(vectorstore: FAISS) -> list[Document]:
    """
    Load documents from FAISS store
    """
    try:
        if hasattr(vectorstore, 'docstore'):
            docs = list(vectorstore.docstore._dict.values())
            if not docs:
                logger.warning("No documents found in vectorstore")
            return docs
        else:
            logger.warning("Vectorstore has no docstore attribute")
            return []
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []




def _build_retrievers(documents: list[Document], vectorstore: FAISS) -> ContextualCompressionRetriever:
    """
    Build and return a compression retriever that includes
    an ensemble retriever and Cohere-based contextual compression.

    Args:
        documents (list[Document]): List of Document objects.
        vectorstore (FAISS): The vector store to use for building retrievers.

    Returns:
        ContextualCompressionRetriever: A compression retriever that can be used to fetch and re-rank documents.
    """
    # Create base retrievers
    retriever_bm25 = BM25Retriever.from_documents(documents, search_kwargs={"k": TOP_K})
    retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K})

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
        weights=ENSEMBLE_WEIGHTS,
    )

    # Set up Cohere re-ranking
    compressor = CohereRerank(top_n=TOP_K_COMPRESSION, model=COHERE_RERANK_MODEL)

    # Build compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever


vectorstore = _setup_vectorstore()
documents = _load_documents(vectorstore)

# Build the compression retriever (with Cohere inside)
compression_retriever = _build_retrievers(documents, vectorstore)


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the question (a step in the research plan).

    This function uses a language model to generate diverse search queries to help answer the question.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """

    class Response(TypedDict):
        queries: list[str]

    logger.info("---GENERATE QUERIES---")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.with_structured_output(Response).ainvoke(messages))
    queries = response["queries"]
    queries.append(state.question)
    logger.info(f"Queries: {queries}")
    return {"queries": response["queries"]}


async def retrieve_and_rerank_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query.

    This function uses a retriever to fetch relevant documents for a given query.

    Args:
        state (QueryState): The current state containing the query string.
        config (RunnableConfig): Configuration with the retriever used to fetch documents.

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """
    logger.info("---RETRIEVING DOCUMENTS---")
    logger.info(f"Query for the retrieval process: {state.query}")

    response = compression_retriever.invoke(state.query)

    return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query.

    This function prepares parallel document retrieval tasks for each query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the generated queries.

    Returns:
        Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

    Behavior:
        - Creates a Send object for each query in the state.
        - Each Send object targets the "retrieve_documents" node with the corresponding query.
    """
    return [
        Send("retrieve_and_rerank_documents", QueryState(query=query)) for query in state.queries
    ]



builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_and_rerank_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_and_rerank_documents"],
)
builder.add_edge("retrieve_and_rerank_documents", END)
researcher_graph = builder.compile()
