import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from rag_utils import (
    load_and_process_papers,
    create_vector_db,
    generate_answer,
    gather_user_requirements,
    recommend_similar_papers,
)

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
chat_memory = ConversationBufferMemory(ai_prefix="AI Assistant")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.5, max_tokens=150)

@cl.on_chat_start
async def start_chat():
    await cl.Message(content="Welcome to the Research Paper QA Chatbot! Please provide the domain or topic you are interested in.").send()
    domain = (await cl.AskUserMessage(content="Enter the domain or topic: ", timeout=360).send())['content'].strip()
    cl.user_session.set("domain", domain)
    
    await cl.Message(content=f"Great! Let's search for papers related to {domain}.").send()
    await cl.Message(content="Enter the topic or keywords to search for papers:").send()
    response = await cl.AskUserMessage(content="Enter your search query: ", timeout=60).send()
    search_query = response['content'].strip()
    docs = load_and_process_papers(search_query)
    db = create_vector_db(docs, OPENAI_API_KEY)
    cl.user_session.set("search_query", search_query)
    cl.user_session.set("db", db)
    await cl.Message(content="Please provide your specific questions or requirements about the topic.").send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()

    if user_input.lower() == 'exit':
        await cl.Message(content="Conversation ended. Thank you for using the chatbot!").send()
        return

    if user_input.lower() == 'search again':
        await start_chat()
        return

    search_query = cl.user_session.get("search_query")
    db = cl.user_session.get("db")
    domain = cl.user_session.get("domain")

    user_requirements = await gather_user_requirements(llm, chat_memory, search_query, message)
    answer = generate_answer(user_requirements, db, llm, domain, user_requirements)
    recommendation = recommend_similar_papers(user_requirements, db, llm, domain)

    response = f"Answer: {answer}\n\nRecommended Papers: {recommendation}\n\nIf you want to search for papers on a different topic, type 'search again'. To exit the conversation, type 'exit'."
    await cl.Message(content=response).send()