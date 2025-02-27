import chainlit as cl
from dotenv import load_dotenv
import os
from pprint import pprint
import numpy as np
import json
import math

import pandas as pd

import re

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings, Settings

from IPython.display import Markdown

from PyPDF2 import PdfReader


import time

from tqdm import tqdm

import google.generativeai as genai

import ollama
from ollama import chat
from ollama import embeddings
from ollama import EmbeddingsResponse
from ollama import ChatResponse

from ddg import Duckduckgo

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=api_key)

local_embedding_model = "bge-m3"

gemini_model = genai.GenerativeModel()

def get_chroma_db_without_embedding_function(name):
    chroma_client = chromadb.PersistentClient(path="database/")
    return chroma_client.get_or_create_collection(name=name)

def get_relevant_passages_with_embeddings_and_articles(query, chat_model, embedding_model, db):

    embeddings_response = embeddings(
        model=embedding_model,
        prompt=query) 
    
    n_results = 1

    results = db.query(
        query_embeddings=embeddings_response.embedding,  
        n_results= n_results)
    
    return results

def convert_passages_to_list_updown(passages):
    context = ""

    a = passages["documents"][0]
    b = passages["ids"][0]

    b = list(map(str, b))
    x = [val for _, val in sorted(zip(b, a))]
    
    for passage in x:
        context += passage + "\n"
    return context

def make_prompt_legal(query, passage):

    query_oneline = query.replace("\n", "\n")

    passage_oneline = passage.replace("\n", "\n")

    ddg_api = Duckduckgo()
    results = ddg_api.search(f"find youtube videos corresponding to this query: {query}")

    list_of_youtube_videos = ""
    for result in results['data']:
        if ("https://www.youtube.com" in result['url']):
            list_of_youtube_videos += (result['url'] + "\n")
    if (list_of_youtube_videos != ""):
        list_of_youtube_videos = "List of similar recipes in youtube:\n" + list_of_youtube_videos

    # This prompt is where you can specify any guidance on tone, or what topics the model should stick to, or avoid.
    prompt = f"""You are a chatbot specialized in answering queries from users about food recipes based on the CONTEXT given bellow.
        Your answer must use exactly the sections given in the context: name, description, nutrition facts, list of ingredients, steps and the list of similar recipes in youtube.
        Your answer must be detailed and nicely formatted.
        In your answer, highlight the ingredients contained in the user query. 
    
    QUERY: 
    {query_oneline}
    
    CONTEXT: 
    {passage_oneline} 
    {list_of_youtube_videos}
    
    """
    
    return prompt

def answer_question_with_gemini(query, embedding_db, chat_model, embedding_model):

    passages = get_relevant_passages_with_embeddings_and_articles(query, chat_model, embedding_model, embedding_db)

    context = convert_passages_to_list_updown(passages)

    prompt = make_prompt_legal(query, context)

    answer = chat_model.generate_content(prompt)

    return answer.text

@cl.step(type="tool")
async def tool():
    # Fake tool
    await cl.sleep(2)
    return "Response from the tool!"


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    chroma_db = get_chroma_db_without_embedding_function("recipes")
    
    answer = answer_question_with_gemini(message.content, chroma_db, gemini_model, local_embedding_model)
    
    # Call the tool
    #tool_res = await tool()
    
    await cl.Message(content=answer).send()

    res = await cl.AskActionMessage(
        content="Tell us what you think about this recipe:",
        actions=[
            cl.Action(name="I like", payload={"value": "like"}, label="👍 I like"),
            cl.Action(name="I don't like", payload={"value": "notlike"}, label="👎 I don't like"),
        ],
    ).send()
    
