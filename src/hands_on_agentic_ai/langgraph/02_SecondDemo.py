from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(verbose=True)

def basic_chain():
    prompt = ChatPromptTemplate.from_template("You are a Medical AI and answer questions in one line {question}")

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    chain = prompt | model | parser

    product = "HHKB Keyboard"
    result = chain.invoke({"question": "What is a Topre Keyboard"})
    print(result)

    return chain

if __name__ == "__main__":
    chain = basic_chain()