from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from importlib.metadata import version
from langchain_core.messages import HumanMessage

load_dotenv()

def main():

    print(version("langchain_core"))
    print(version("langchain_openai"))
    print(version("langchain_anthropic"))
    print(version("langgraph"))
    # stream_usage=True makes OpenAI send token counts in the last streamed chunk
    # llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, stream_usage=True)
    # # response = llm_openai.invoke("What is at the center of Andromeda galaxy?")
    # # print(response.content)
    # # print(response.response_metadata)
    #
    # # Create a message
    # msg = HumanMessage(content="What is the biggest object in the universe?", name="Nash")
    # full = None  # None | AIMessageChunk
    # for chunk in llm_openai.stream([msg]):
    #     print(chunk.text, end="", flush=True)  # print each piece as it arrives
    #     full = chunk if full is None else full + chunk  # adding chunks merges text and metadata
    # print()
    #
    # # Metadata is spread across chunks; it's complete once all chunks are merged
    # print("Response metadata:", full.response_metadata)  # model name, finish reason, etc.
    # print("Usage metadata:", full.usage_metadata)  # input/output/total token counts
    #
    # print()
    # msg2 = "Who made first smart phone?"
    # llm_openai.invoke([msg2])
    # print(response.response_metadata)
    # print(response.content)
    # llm_anthopic = ChatAnthropic(model_name="gpt-4o-mini")
    # response = llm_anthopic.invoke("Hi")
    # print(response.content)


    from langchain_tavily import TavilySearch  # updated at 1.0

    tavily_search = TavilySearch(max_results=3)

    data = tavily_search.invoke({"query": "What is LangGraph?"})
    search_docs = data.get("results", data)
    print(search_docs)

if __name__ == "__main__":
    main()