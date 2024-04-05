import os
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, PubMedAPIWrapper, ArxivAPIWrapper
from tenacity import retry, stop_after_attempt, wait_exponential
import chainlit as cl

def load_and_process_papers(search_query):
    loader = ArxivLoader(query=search_query, load_max_docs=10)
    files = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n---\n', chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(files)
    return docs

def create_vector_db(docs, openai_api_key):
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-ada-002")
    db = Chroma.from_documents(docs, embedding_model, persist_directory='./chroma_db')
    return db

def generate_answer(query, db, llm, domain, user_requirements):
    search = DuckDuckGoSearchAPIWrapper()
    pubmed_search = PubMedAPIWrapper()
    arxiv_search = ArxivAPIWrapper()

    rag_tool = Tool(
        name="RAG System",
        func=lambda q: RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever()).run(q),
        description="Useful for answering questions based on the retrieved papers."
    )

    tool_belt = [
        Tool(
            name="DuckDuckGo Search",
            func=search.run,
            description="Useful for searching for additional information on the internet."
        ),
        rag_tool,
        Tool(
            name="PubMed Search",
            func=pubmed_search.run,
            description="Useful for searching for biomedical literature on PubMed."
        ),
        Tool(
            name="arXiv Search",
            func=arxiv_search.run,
            description="Useful for searching for papers on arXiv."
        )
    ]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_answer_with_tools(query, user_requirements, domain):
        agent = initialize_agent(tool_belt, llm, agent="zero-shot-react-description", verbose=True)
        return agent.run(f"""
        The user has asked questions about {query} in the domain of {domain} and provided the following specific requirements:
        {user_requirements}
        
        Based on the user's questions and requirements, provide a comprehensive answer that addresses their specific concerns. Use the available tools to gather relevant information from the retrieved papers and ensure that the answer is accurate and relevant to the user's query.
        
        Give detailed answers of all the user's questions.

        If the available tools do not provide sufficient information to answer the user's questions, suggest alternative search queries or resources related to the topic.
        
        Remember to tailor your answer to the user's level of understanding and specific needs.
        """)

    try:
        answer = generate_answer_with_tools(query, user_requirements, domain)
    except Exception as e:
        print(f"Error generating answer: {str(e)}")
        answer = "I apologize, I encountered an error while generating the answer. I would recommend searching for more specific papers or resources related to the topic to get a better understanding."

    return answer

async def gather_user_requirements(llm, chat_memory, search_query, message):
    template = """
    You're an AI assistant and your task is to gather all details from a user who wants to understand a concept.

    At the beginning, shortly describe the purpose of this conversation.

    You should gather answers for the following questions:

    - What specific questions do you have about this topic?
    - What is your current level of understanding of this topic?
    - Are there any related concepts you'd like me to explain as well?

    Don't answer the question you are asking.

    Be patient and encouraging if the user doesn't know how to answer some questions, and help guide them.

    Ask one question at a time.
    

    Once you have gathered all the details, thank the user for their responses, summarize the relevant information that will help you provide the best explanation, and put '<END-OF-CONVERSATION>'

    Current conversation:
    {history}
    Human: {input}
    AI assistant:
    """

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(
        prompt=prompt,
        llm=llm,
        verbose=False,
        memory=chat_memory
    )

    current_input = f"I want to learn about {search_query}"
    end_seq = '<END-OF-CONVERSATION>'
    user_requirements = ''

    while True:
        ai_response = conversation.predict(input=current_input)
        await cl.Message(content=ai_response).send()

        if end_seq in ai_response:
            user_requirements = chat_memory.chat_memory.messages[-1].content.replace(end_seq, '')
            break

        response = await cl.AskUserMessage(content="User: ", timeout=60).send()
        user_input = response['content'].strip()
        current_input = user_input

    return user_requirements

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def recommend_similar_papers(user_requirements, db, llm, domain):
    similar_papers = db.similarity_search(user_requirements, k=3)

    if similar_papers:
        prompt_template = PromptTemplate(
            input_variables=["papers", "domain"],
            template="""
            Based on the user's requirements and the retrieved papers in the domain of {domain}, recommend 3 similar papers that could help the user better understand the topic.

            Retrieved Papers:
            {papers}
            """
        )
        similar_papers_str = "\n".join([f"- {paper.metadata.get('title', 'Title not available')}" for paper in similar_papers])
        recommend_prompt = prompt_template.format(papers=similar_papers_str, domain=domain)
        recommendation = llm.invoke(recommend_prompt)
    else:
        recommendation = "I apologize, I couldn't find any similar papers based on the user's requirements. The loaded papers do not seem to contain enough relevant information about the topic. I would suggest searching for more specific papers related to the user's questions to get a better understanding."

    return recommendation