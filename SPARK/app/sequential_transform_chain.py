import os
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import pinecone
import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate

pinecone.init(
    api_key="d5edf74b-d4c7-4160-9628-389de98a9e02",
    environment='gcp-starter'
)

from langchain.chains import LLMChain, TransformChain, SequentialChain

from chainlit import on_message, on_chat_start

index_name = "spark"

# Optional
namespace = None

embeddings = CohereEmbeddings(model='embed-english-light-v2.0',cohere_api_key="OdFQPnlX9Xv0OjKwZP5s5L1ODJl9e0wFTk3YIl5g")

llm = ChatOpenAI(temperature=0.7, verbose=True)

docsearch = Pinecone.from_existing_index(
        index_name=index_name, embedding=embeddings, namespace=namespace
    )

# welcome_message = "Welcome to the Chainlit Pinecone demo! Ask anything about documents you vectorized and stored in your Pinecone DB."
memory = ConversationBufferMemory(llm=llm, input_key='question',memory_key='chat_history',return_messages=True)
_template = """Below is a summary of the conversation so far, and a new question asked by the user that needs to be answered by searching in a knowledge base. Generate a search query based on the conversation and the new question.
Don't generate the search query if the user is conversing generally or engaging in small talk. In which case just return the original question.
Chat History:
{chat_history}

Question:
{question}

Remember - Don't change the search query from the user's question if user is engaging in small talk.
Search query:
"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

spark = """You are SPARK, a Prompt Engineering Assistant. SPARK stands for Smart Prompt Assistant and Resource Knowledgebase.

You are an AI-powered assistant that exudes a friendly and knowledgeable persona. You are designed to be a reliable and trustworthy guide in the
world of prompt engineering. With a passion for prompt optimization and a deep understanding of AI models, SPARK is committed to helping users navigate the field of prompt engineering and craft
high-performing prompts.

Personality:

Intelligent: SPARK is highly knowledgeable about prompt engineering concepts and practices. It possesses a vast array of information and resources to share with users, making it an expert in its field.

Patient: SPARK understands that prompt engineering can be complex and requires careful attention to detail. It patiently guides users through the intricacies of crafting prompts, offering support at every step.

Adaptable: SPARK recognizes that prompt engineering is a dynamic field with evolving best practices. It stays up to date with the latest trends and developments, adapting its knowledge and recommendations accordingly.

Interactions with SPARK:
Users can engage with SPARK by seeking advice on prompt design, exploring prompt engineering concepts, discussing challenges they encounter, and receiving recommendations for improving AI model performance. SPARK responds promptly, providing clear and concise explanations, examples, and actionable tips.

Important:
Answer with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. If asking a clarifying question to the user would help, ask the question. 

Sources:
---------------------
    {context}
---------------------
The sources above are NOT related to the conversation with the user. Ignore the sources if user is engaging in small talk.
"""
question_gen_prompt = PromptTemplate(template=_template, input_variables=["question", "chat_history"]  )

@on_chat_start
def init():
    memory.clear()
    
def transform_func(inputs: dict) -> dict:
    query = inputs["question"]
    qgen = LLMChain(
                llm=llm, prompt=question_gen_prompt, verbose=True, memory=memory, output_key='context')
            # Run the LLM Chain with the input variables. Note - Added additional format_instructions to parse the output as JSON
    search_query = qgen.predict(question=query)
    result = docsearch.similarity_search(search_query)
    context = [f"\n{source.page_content}\nSource:\n{source.metadata.get('title')} - {source.metadata.get('source')}" for source in result]
    return {"context": '\n'.join(context), "query":query}


@on_message
@cl.langchain_factory(use_async=True)
async def langchain_factory():
    retriever = docsearch.as_retriever(search_kwargs={"k":4}, search_type='mmr')
    messages = [SystemMessagePromptTemplate.from_template(spark)]
    messages.extend(memory.chat_memory.messages)
    messages.append(HumanMessagePromptTemplate.from_template("{query}"))

    chat_prompt = ChatPromptTemplate(messages=messages, input_variables=["context", "query"]  )
    answer_generator = LLMChain(
            llm=llm, prompt=chat_prompt, verbose=True, output_key='answer', memory=memory)

    transform_chain = TransformChain(
        input_variables=["question", ], output_variables=["context","query"], transform=transform_func
    )
        
    conversational_QA_chain = SequentialChain(
        chains=[transform_chain, answer_generator],
        input_variables=["chat_history", "question"],
        # Here we return multiple variables
        output_variables=["context", "answer"],
        verbose=True)

    return conversational_QA_chain

@cl.langchain_run
async def run(chain, input_str):
    res = chain._call({"question":input_str})
    await cl.Message(content=res["answer"]).send()
    
    
@cl.langchain_postprocess
async def process_response(res):
    print('res', res)
    answer = res["answer"]
    sources = res.get("sources", "").strip()  # Use the get method with a default value
    print('sources', sources)
    source_elements = []
    docs = res.get("source_documents", None)

    if docs:
        metadatas = [doc.metadata for doc in docs]
        # Get the source names from the metadata
        print('meta', metadatas)
        all_sources = [m["source"] for m in metadatas]
        print('all sources', all_sources)
        for i, source in enumerate(metadatas):
            source_elements.append(cl.Text(content=source.get('source'), name=source.get('title'), display='inline'))

    # Send the answer and the text elements to the UI
    await cl.Message(content=answer, elements=source_elements).send()