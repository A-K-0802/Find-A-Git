from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
import requests
import streamlit as st
import os
from langchain_community.tools import tool, StructuredTool
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
import json
load_dotenv()

client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",)

# prompt_template= PromptTemplate(
#     template="""You are a helpful AI assistant.
#     Your job is to help the user find a suitable project idea using GitHub repositories.
#     You will be provided by the user about the {topic}, the {language} of the project, and what they {need} you to provide.
#     The need can either be a project idea, example repository, or a list of repositories, or any open source project where the user can contribute.
#     You will use the tools provided to you to help the user.""",
#     input_variables=["topic", "need", "language"]
# )

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

@tool
def get_git_repo(topic:str, language:str, max_count:int=5):
    """This function fetches information about top 5 GitHub repositories based on the topic and language provided by the user.
    It returns a list of repositories that match the criteria."""
    headers={
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    params={
        "q": f"{topic} language:{language}",
        "sort": "stars",
        "order": "desc",
        "per_page": max_count
    }
    
    url="https://api.github.com/search/repositories"
    response = requests.get(url, headers=headers, params=params)
    repos= response.json().get("items", [])

    llm_data = []
    for repo in repos:
        llm_data.append({
            "name": repo["name"],
            "url": repo["html_url"],
            "description": repo["description"],
            "stars": repo["stargazers_count"],
            "language": repo["language"]
        })
    return llm_data

# print(get_git_repo.invoke({"topic": "pathfinding", "language": "python"}))

agent= client.bind_tools([get_git_repo])

messages=[HumanMessage("I want to contribute to a project on pathfinding using Python. Can you help me find a open source project after looking at existing repos?")]

response=agent.invoke(messages)
messages.append(response)

for tool_call in response.tool_calls:
    tool_response = get_git_repo.invoke(tool_call)
    messages.append(tool_response)

result=agent.invoke(messages)
print(result.content)