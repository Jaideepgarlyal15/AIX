import os
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
import pinecone
import chainlit as cl
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks import get_openai_callback
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from chainlit import user_session
from prompts import load_query_gen_prompt, load_spark_prompt
from chainlit import on_message, on_chat_start
import openai
from langchain.callbacks import ContextCallbackHandler
from promptwatch import PromptWatch
import wandb


index_name = "spark"
wandb.login(key="2f00e5f916a527f033a1d4466e07bf759c6f1dc1")

spark = load_spark_prompt()
query_gen_prompt = load_query_gen_prompt()
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(query_gen_prompt)
pinecone.init(
            api_key="d5edf74b-d4c7-4160-9628-389de98a9e02",
            environment='gcp-starter'
)

@on_chat_start
def init(): 
    token = "Help me with the spark documentation."
    context_callback = ContextCallbackHandler(token)
    os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
    os.environ["WANDB_PROJECT"] = "spark"
    llm = ChatOpenAI(temperature=0.7, verbose=True, openai_api_key = "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", streaming=True,
                     callbacks=[context_callback])
    memory = ConversationTokenBufferMemory(llm=llm,memory_key="chat_history", return_messages=True,input_key='question',max_token_limit=1000)
    embeddings = CohereEmbeddings(model='embed-english-light-v2.0',cohere_api_key="OdFQPnlX9Xv0OjKwZP5s5L1ODJl9e0wFTk3YIl5g")

    docsearch = Pinecone.from_existing_index(
    index_name=index_name, embedding=embeddings
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": 4})
    # compressor = CohereRerank()
    # reranker = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=retriever
    # )
    messages = [SystemMessagePromptTemplate.from_template(spark)]
    # print('mem', user_session.get('memory'))
    messages.append(HumanMessagePromptTemplate.from_template("{question}"))
    prompt = ChatPromptTemplate.from_messages(messages)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=True)
    doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", verbose=True,prompt=prompt)

    chain = ConversationalRetrievalChain(
            retriever=retriever,
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            verbose=True,
            memory=memory,
            rephrase_question=False,
            callbacks=[context_callback]
        )
    cl.user_session.set("conversation_chain", chain)

    
@on_message
async def main(message):
    with PromptWatch(api_key="dXZNRldjVTg1WlZLZUhFYmJXY2d6VjV2WU1kMjo2OThlMzFlMC1lMTk4LTU3ZmQtODMwYS1kYTNiMTNhZmI1ZDE=") as pw:
        token = "Help me please"
        context_callback = ContextCallbackHandler(token)
        chain = cl.user_session.get("conversation_chain")
        res = await chain.arun({"question": message.content},callbacks=[cl.AsyncLangchainCallbackHandler(),
                                                                context_callback])

        # Send the answer and the text elements to the UI
        await cl.Message(content=res).send()