import os
from typing import Optional, TypedDict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, StateGraph

from utils.utils import *

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# 폰트를 matplotlib에 등록
font_manager.fontManager.addfont("./utils/fonts/NanumBarunGothic.ttf")

plt.rcParams["axes.unicode_minus"] = False
plt.rc("font", family="NanumBarunGothic")


class State(TypedDict):
    question: str
    generation: str
    data: str
    code: str


class ExcelPDFChatbot:
    def __init__(
        self,
        df_data: Optional[pd.DataFrame] = None,
        df_description: Optional[str] = None,
        pdf_path: Optional[str] = None,
        pdf_description: Optional[str] = None,
    ) -> None:
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="openai/gpt-5-nano",
            base_url="https://mlapi.run/daef5150-72ef-48ff-8861-df80052ea7ac/v1",
            temperature=1,
            streaming=False,
        )
        self.route_llm = ChatOpenAI(
            openai_api_key=api_key,
            model="openai/gpt-5-nano",
            base_url="https://mlapi.run/daef5150-72ef-48ff-8861-df80052ea7ac/v1",
            temperature=1,
        )
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="openai/text-embedding-3-small",
            base_url="https://mlapi.run/b54ff33e-6d14-42df-93f9-0f1132160ee8/v1"
        )

        self.df_data = df_data
        self.pdf_path = pdf_path

        if df_data is not None:
            self.df_data = df_data
            self.df_description = df_description
            self.df_columns = ", ".join(self.df_data.columns.tolist())
            if self.df_description is None:
                raise ValueError("Please provide a description for the Excel data.")

        if pdf_path is not None:
            self.pdf_path = pdf_path
            self.pdf_description = pdf_description
            if self.pdf_description is None:
                raise ValueError("Please provide a description for the PDF data.")

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(docs)

            self.vectorstore = FAISS.from_documents(
                split_docs, embedding=self.embeddings
            )
            self.db_retriever = self.vectorstore.as_retriever()

        self.graph = StateGraph(State)

        self.graph.add_node("init_answer", self.route_question)
        self.graph.add_node("excel_data", self.query)
        self.graph.add_node("rag", self.retrieval)
        self.graph.add_node("excel_plot", self.plot_graph)
        self.graph.add_node("answer_with_data", self.answer_with_data)
        self.graph.add_node("plain_answer", self.answer)
        self.graph.add_node("answer_with_retrieval", self.answer_with_retrieved_data)

        self.graph.set_entry_point("init_answer")

        self.graph.add_edge("plain_answer", END)
        self.graph.add_edge("answer_with_data", END)
        self.graph.add_edge("answer_with_retrieval", END)
        self.graph.add_edge("excel_plot", END)
        self.graph.add_edge("excel_data", "answer_with_data")
        self.graph.add_edge("rag", "answer_with_retrieval")

        self.graph.add_conditional_edges(
            "init_answer",
            self._extract_route,
            {
                "excel_data": "excel_data",
                "rag": "rag",
                "excel_plot": "excel_plot",
                "plain_answer": "plain_answer",
            },
        )

        self.graph = self.graph.compile()

    def invoke(self, question) -> str:
        return self.graph.invoke({"question": question})

    def query(self, state: State):
        print("---데이터 쿼리---")
        question = state["question"]

        if self.df_data is None:
            raise ValueError("Please provide Excel data to query while initializing the chatbot.")

        system_message = f"당신은 주어진 {self.df_description} 데이터를 분석하는 데이터 분석가입니다.\n"
        system_message += f"{self.df_description} 데이터가 저장된 df DataFrame에서 데이터를 출력하여 주어진 질문에 답할 수 있는 파이썬 코드를 작성하세요. "
        system_message += f"`df DataFrame에는 다음과 같은 열이 있습니다: {self.df_columns}\n"
        system_message += "데이터는 이미 로드되어 있으므로 데이터 로드 코드를 생략해야 합니다."

        prompt_with_data_info = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{question}"),
        ])

        code_generate_chain = (
            {"question": RunnablePassthrough()}
            | prompt_with_data_info
            | self.llm
            | StrOutputParser()
            | python_code_parser
        )
        code = code_generate_chain.invoke(question)
        data = run_code(code, df=self.df_data)
        return {"question": question, "code": code, "data": data, "generation": code}

    def answer_with_data(self, state: State):
        print("---데이터 기반 답변 생성---")
        question = state["question"]
        data = state["data"]

        reasoning_with_data_chain = (
            ChatPromptTemplate.from_messages([
                ("system", "당신은 데이터를 바탕으로 질문에 답하는 데이터 분석가입니다.\n사용자가 입력한 데이터를 바탕으로, 질문에 대답하세요."),
                ("human", "데이터: {data}\n{question}"),
            ])
            | self.llm
            | StrOutputParser()
        )

        generation = reasoning_with_data_chain.invoke({"data": data, "question": question})
        return {"question": question, "code": state["code"], "data": data, "generation": generation}

    def answer(self, state: State):
        print("---답변 생성---")
        question = state["question"]
        return {"question": question, "generation": self.llm.invoke(question).content}

    def plot_graph(self, state: State):
        def change_plot_to_save(code: str) -> str:
            code = code.split("plt.plot()")[0]
            code += "plt.plot()\nplt.savefig('plot.png')"
            return code

        print("---그래프 시각화---")
        question = state["question"]

        system_message = (
            f"당신은 주어진 {self.df_description} 데이터를 분석하는 데이터 분석가입니다.\n"
            f"{self.df_description} 데이터가 저장된 df DataFrame에서 데이터를 추출하여 "
            "사용자의 질문에 답할 수 있는 그래프를 그리는 plt.plot()으로 끝나는 코드를 작성하세요. "
            f"`df` DataFrame에는 다음과 같은 열이 있습니다: {self.df_columns}\n"
            "데이터는 이미 로드되어 있으므로 데이터 로드 코드를 생략해야 합니다."
        )

        code_generate_chain = (
            {"question": RunnablePassthrough()}
            | ChatPromptTemplate.from_messages([("system", system_message), ("human", "{question}")])
            | self.llm
            | StrOutputParser()
            | python_code_parser
            | change_plot_to_save
        )
        code = code_generate_chain.invoke(question)
        answer = run_code(code, df=self.df_data)
        data = "plot.png"
        if "Error" in answer:
            data = None
        return {"question": question, "code": code, "data": data, "generation": answer}

    def retrieval(self, state: State):
        def get_retrieved_text(docs):
            return "\n".join([doc.page_content for doc in docs])

        print("---데이터 검색---")
        question = state["question"]
        data = (self.db_retriever | get_retrieved_text).invoke(question)
        return {"question": question, "data": data}

    def answer_with_retrieved_data(self, state: State):
        print("---검색된 데이터를 바탕으로 답변 생성---")
        question = state["question"]
        data = state["data"]

        qa_chain = (
            ChatPromptTemplate.from_messages([
                ("system", """
                당신은 사용자의 질문에 자세하게 답변하는 QA 챗봇입니다. 
                사용자가 입력하는 정보를 바탕으로 질문에 답하세요.
                답변을 할 때엔 근거가 된 문장을 알려주어야 합니다.
                해당하는 내용이 문서에 없을 때엔 '해당 질문에 대한 내용은 문서에 존재하지 않습니다.' 라는 답변을 출력하세요.
                """),
                ("human", "정보: {context}.\n{question}."),
            ])
            | self.llm
            | StrOutputParser()
        )

        generation = qa_chain.invoke({"context": data, "question": question})
        return {"question": question, "data": data, "generation": generation}

    def _extract_route(self, state: State) -> str:
        return state["generation"]

    def route_question(self, state: State):
        print("---질문 라우팅---")
        route_system_message = "당신은 사용자의 질문에 RAG, 엑셀 데이터 중 어떤 것을 활용할 수 있는지 결정하는 전문가입니다."
        usable_tools_list = ["`plain_answer`"]

        if self.df_data is not None:
            route_system_message += f"{self.df_description} 과 관련된 질문이라면 excel_data를 활용하세요. \n"
            route_system_message += "그래프를 그리는 질문이라면 excel_plot을 활용하세요. \n"
            usable_tools_list.extend(["`excel_data`", "`excel_plot`"])

        if self.pdf_path is not None:
            route_system_message += f"{self.pdf_description} 과 관련된 질문이라면 RAG를 활용하세요. \n"
            usable_tools_list.append("`rag`")

        route_system_message += "그 외의 질문이라면 plain_answer로 충분합니다. \n"
        usable_tools_text = ", ".join(usable_tools_list)
        route_system_message += f"주어진 질문에 맞춰 {usable_tools_text} 중 하나를 선택하세요. \n"
        route_system_message += "답변은 `route` key 하나만 있는 JSON으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."

        router_chain = (
            ChatPromptTemplate.from_messages([
                ("system", route_system_message),
                ("human", "{question}"),
            ])
            | self.route_llm
            | JsonOutputParser()
        )
        route = router_chain.invoke({"question": state["question"]})["route"]
        return {"question": state["question"], "generation": route.lower().strip()}
