from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from importlib.metadata import version

load_dotenv()

def main():

    print(version("langchain_core"))
    print(version("langchain_openai"))
    print(version("langchain_anthropic"))
    print(version("langgraph"))
    # llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    # response = llm_openai.invoke("What is at the center of Andromeda galaxy?")
    # print(response.content)

    # llm_anthopic = ChatAnthropic(model_name="gpt-4o-mini")
    # response = llm_anthopic.invoke("Hi")
    # print(response.content)

if __name__ == "__main__":
    main()