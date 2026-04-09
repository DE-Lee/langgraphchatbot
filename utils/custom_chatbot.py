import os
from typing import Optional, TypedDict

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    # 그래프 상태의 속성을 정의합니다.
    # 질문, LLM이 생성한 텍스트, 데이터, 코드를 저장합니다.
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
        """
        Chatbot을 초기화합니다.

        Args:
            df_data (Optional[pd.DataFrame], optional): 엑셀 데이터 프레임. Defaults to None.
            df_description (Optional[str], optional): 엑셀 데이터 프레임 설명. df_data가 None이 아닐 경우 설명을 반드시 입력해야 합니다. Defaults to None.
            pdf_path (Optional[str], optional): PDF 파일 경로 리스트. Defaults to None.
            pdf_description (Optional[str], optional): PDF 파일 설명 리스트. pdf_path가 None이 아닐 경우, 설명을 반드시 입력해야 합니다. Defaults to None.
        """
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o-mini",
            temperature=0,
            streaming=True,
        )
        self.route_llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o-mini",
            temperature=0,
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        self.df_data = df_data
        self.pdf_path = pdf_path

        # 엑셀 데이터를 불러옵니다.
        if df_data is not None:
            self.df_data = df_data
            self.df_description = df_description
            self.df_columns = ", ".join(self.df_data.columns.tolist())
            if self.df_description is None:
                raise ValueError("Please provide a description for the Excel data.")

        # PDF 데이터를 불러옵니다.
        if pdf_path is not None:
            self.pdf_path = pdf_path
            self.pdf_description = pdf_description
            if self.pdf_description is None:
                raise ValueError("Please provide a description for the PDF data.")

            # PDF 문서를 불러옵니다.
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # 텍스트 스플리터로 텍스트를 나눕니다.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=100
            )
            split_docs = text_splitter.split_documents(docs)

            # FAISS에 청크된 문서를 추가합니다.
            self.vectorstore = FAISS.from_documents(
                split_docs, embedding=self.embeddings
            )
            self.db_retriever = self.vectorstore.as_retriever()

        # 그래프를 초기화합니다.
        self.graph = StateGraph(State)

        ## 그래프 구성

        # 앞서 정의한 Node를 모두 추가합니다.
        self.graph.add_node("init_answer", self.route_question)

        self.graph.add_node("excel_data", self.query)
        self.graph.add_node("rag", self.retrieval)

        self.graph.add_node("excel_plot", self.plot_graph)
        self.graph.add_node("answer_with_data", self.answer_with_data)
        self.graph.add_node("plain_answer", self.answer)
        self.graph.add_node("answer_with_retrieval", self.answer_with_retrieved_data)

        # 시작지점을 정의합니다.
        self.graph.set_entry_point("init_answer")

        # 간선을 정의합니다.
        # END는 종결 지점을 의미합니다.
        self.graph.add_edge(
            "plain_answer", END
        )  # self.graph.set_finish_point("answer")와 동일합니다.
        self.graph.add_edge("answer_with_data", END)
        self.graph.add_edge("answer_with_retrieval", END)
        self.graph.add_edge("excel_plot", END)  # 그래프를 그리고 종결합니다.
        self.graph.add_edge("excel_data", "answer_with_data")
        self.graph.add_edge("rag", "answer_with_retrieval")

        # 조건부 간선을 정의합니다.
        # init_answer 노드의 답변을 바탕으로 decide_query 함수에서 query 또는 answer로 분기합니다.
        self.graph.add_conditional_edges(
            "init_answer",
            self._extract_route,
            # 어떤 노드로 이동할지 mapping합니다. 없어도 무방하지만, Graph의 가독성을 높일 수 있습니다.
            {
                "excel_data": "excel_data",
                "rag": "rag",
                "excel_plot": "excel_plot",
                "plain_answer": "plain_answer",
            },
        )

        self.graph = self.graph.compile()

    def invoke(self, question) -> str:
        answer = self.graph.invoke({"question": question})

        return answer

    def query(self, state: State):
        """
        데이터를 쿼리하는 코드를 생성하고, 실행하고, 그 결과를 포함한 State를 반환합니다.
        위 과정은 앞서 정의한 `find_data` 함수를 활용합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): 쿼리한 데이터를 포함한 새로운 State
        """
        print("---데이터 쿼리---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]

        if self.df_data is None:
            raise ValueError(
                "Please provide Excel data to query while initializing the chatbot."
            )

        # Retrieval
        # 이전 실습에서 `find_data` 함수를 사용했지만, 여기서는 query 함수에 해당 로직을 포함시켰습니다.
        system_message = f"당신은 주어진 {self.df_description} 데이터를 분석하는 데이터 분석가입니다.\n"
        system_message += f"{self.df_description} 데이터가 저장된 df DataFrame에서 데이터를 출력하여 주어진 질문에 답할 수 있는 파이썬 코드를 작성하세요. "
        # system_message += f"df DataFrame에 액세스할 수 있습니다.\n"
        system_message += (
            f"`df DataFrame에는 다음과 같은 열이 있습니다: {self.df_columns}\n"
        )
        system_message += (
            "데이터는 이미 로드되어 있으므로 데이터 로드 코드를 생략해야 합니다."
        )

        message_with_data_info = [
            ("system", system_message),
            ("human", "{question}"),
        ]

        prompt_with_data_info = ChatPromptTemplate.from_messages(message_with_data_info)

        # 체인을 구성합니다.
        code_generate_chain = (
            {"question": RunnablePassthrough()}
            | prompt_with_data_info
            | self.llm
            | StrOutputParser()
            | python_code_parser
        )
        code = code_generate_chain.invoke(question)
        data = run_code(code, df=self.df_data)
        return {
            "question": question,
            "code": code,
            "data": data,
            "generation": code,
        }

    def answer_with_data(self, state: State):
        """
        쿼리한 데이터를 바탕으로 답변을 생성합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): LLM의 답변을 포함한 새로운 State
        """
        print("---데이터 기반 답변 생성---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]
        data = state["data"]

        # 데이터를 바탕으로 질문에 대답하는 코드를 생성합니다.
        reasoning_system_message = (
            "당신은 데이터를 바탕으로 질문에 답하는 데이터 분석가입니다.\n"
        )
        reasoning_system_message += (
            f"사용자가 입력한 데이터를 바탕으로, 질문에 대답하세요."
        )

        reasoning_user_message = "데이터: {data}\n{question}"

        reasoning_with_data = [
            ("system", reasoning_system_message),
            ("human", reasoning_user_message),
        ]
        reasoning_with_data_chain = (
            ChatPromptTemplate.from_messages(reasoning_with_data)
            | self.llm
            | StrOutputParser()
        )

        # 대답 생성
        generation = reasoning_with_data_chain.invoke(
            {"data": data, "question": question}
        )
        return {
            "question": question,
            "code": state["code"],
            "data": data,
            "generation": generation,
        }

    def answer(self, state: State):
        """
        데이터를 쿼리하지 않고 답변을 바로 생성합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): LLM의 답변을 포함한 새로운 State
        """
        print("---답변 생성---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]

        return {
            "question": question,
            "generation": self.llm.invoke(question).content,
        }

    def plot_graph(self, state: State):
        """
        현재 그래프 상태를 시각화합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            None
        """

        def change_plot_to_save(code: str) -> str:

            # plt.plot() 뒤에 plt.savefig('plot.png')을 추가합니다.
            code = code.split("plt.plot()")[0]
            code += "plt.plot()\nplt.savefig('plot.png')"
            return code

        print("---그래프 시각화---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]

        # 챗봇이 이미지를 안정적으로 불러올 수 있도록 프롬프트를 개선했습니다.
        # 그래프를 그릴 경우, 반드시 `plt.plot()` 으로 코드를 마무리해야 합니다.
        system_message = (
            f"당신은 주어진 {self.df_description} 데이터를 분석하는 데이터 분석가입니다.\n"
            f"{self.df_description} 데이터가 저장된 df DataFrame에서 데이터를 추출하여 "
            "사용자의 질문에 답할 수 있는 그래프를 그리는 plt.plot()으로 끝나는 코드를 작성하세요. "
            f"`df` DataFrame에는 다음과 같은 열이 있습니다: {self.df_columns}\n"
            "데이터는 이미 로드되어 있으므로 데이터 로드 코드를 생략해야 합니다."
        )

        message_with_data_info = [
            ("system", system_message),
            ("human", "{question}"),
        ]

        prompt_with_data_info = ChatPromptTemplate.from_messages(message_with_data_info)

        # 체인을 구성합니다.
        code_generate_chain = (
            {"question": RunnablePassthrough()}
            | prompt_with_data_info
            | self.llm
            | StrOutputParser()
            | python_code_parser
            | change_plot_to_save  # plt.plot()를 plt.savefig('plot.png')로 변경합니다.
        )
        code = code_generate_chain.invoke(question)
        # 코드를 실행하고, 출력값 혹은 에러 메시지를 반환합니다.
        answer = run_code(code, df=self.df_data)
        # 챗봇이 `plot.png` 파일을 불러오도록 설정합니다.
        data = "plot.png"

        # 에러가 발생했을 경우, data를 None으로 설정합니다.
        if "Error" in answer:
            data = None
        return {
            "question": question,
            "code": code,
            "data": data,
            "generation": answer,
        }

    def retrieval(self, state: State):
        """
        데이터 검색을 수행합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): 검색된 데이터를 포함한 새로운 State
        """

        def get_retrieved_text(docs):
            result = "\n".join([doc.page_content for doc in docs])
            return result

        print("---데이터 검색---")  # 현재 상태를 확인하기 위한 Print문
        question = state["question"]

        # Retrieval Chain
        retrieval_chain = self.db_retriever | get_retrieved_text

        data = retrieval_chain.invoke(question)

        return {"question": question, "data": data}

    def answer_with_retrieved_data(self, state: State):
        """
        검색된 데이터를 바탕으로 답변을 생성합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): LLM의 답변을 포함한 새로운 State
        """
        # role에는 "AI 어시스턴트"가, question에는 "당신을 소개해주세요."가 들어갈 수 있습니다.

        print(
            "---검색된 데이터를 바탕으로 답변 생성---"
        )  # 현재 상태를 확인하기 위한 Print문

        question = state["question"]
        data = state["data"]

        # 2챕터의 프롬프트와 체인을 활용합니다.
        messages_with_contexts = [
            (
                "system",
                """
                당신은 사용자의 질문에 자세하게 답변하는 QA 챗봇입니다. 
                사용자가 입력하는 정보를 바탕으로 질문에 답하세요.
                답변을 할 때엔 근거가 된 문장을 알려주어야 합니다.
                해당하는 내용이 문서에 없을 때엔 '해당 질문에 대한 내용은 문서에 존재하지 않습니다.' 라는 답변을 출력하세요.
                """,
            ),
            ("human", "정보: {context}.\n{question}."),
        ]
        prompt_with_context = ChatPromptTemplate.from_messages(messages_with_contexts)

        # 체인 구성
        qa_chain = prompt_with_context | self.llm | StrOutputParser()

        generation = qa_chain.invoke({"context": data, "question": question})
        return {"question": question, "data": data, "generation": generation}

    def _extract_route(self, state: State) -> str:
        """
        라우팅된 질문을 추출합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            str: 라우팅된 질문
        """
        return state["generation"]

    def route_question(self, state: State):
        """
        질문을 라우팅합니다.

        Args:
            state (dict): 현재 그래프 상태

        Returns:
            state (dict): 라우팅된 질문을 포함한 새로운 State
        """
        print("---질문 라우팅---")
        # 시스템 메시지에 사용 가능한 툴과 각 툴을 사용할 상황을 명시합니다.
        # 수월한 선택을 위해 JSON 형식으로 출력하도록 프롬프트에 지정합니다.
        route_system_message = "당신은 사용자의 질문에 RAG, 엑셀 데이터 중 어떤 것을 활용할 수 있는지 결정하는 전문가입니다."

        usable_tools_list = ["`plain_answer`"]

        if self.df_data is not None:
            route_system_message += f"{self.df_description} 과 관련된 질문이라면 excel_data를 활용하세요. \n"
            route_system_message += (
                f"그래프를 그리는 질문이라면 excel_plot을 활용하세요. \n"
            )
            usable_tools_list.extend(["`excel_data`", "`excel_plot`"])

        if self.pdf_path is not None:
            route_system_message += (
                f"{self.pdf_description} 과 관련된 질문이라면 RAG를 활용하세요. \n"
            )
            usable_tools_list.append("`rag`")

        route_system_message += "그 외의 질문이라면 plain_answer로 충분합니다. \n"

        usable_tools_text = ", ".join(usable_tools_list)

        route_system_message += (
            f"주어진 질문에 맞춰 {usable_tools_text} 중 하나를 선택하세요. \n"
        )
        route_system_message += "답변은 `route` key 하나만 있는 JSON으로 답변하고, 다른 텍스트나 설명을 생성하지 마세요."
        route_user_message = "{question}"
        route_prompt = ChatPromptTemplate.from_messages(
            [("system", route_system_message), ("human", route_user_message)]
        )
        # 로직 선택용 LLM 객체를 생성합니다.
        # 출력 양식을 json으로 명시하고, 같은 질문에 같은 로직을 적용하기 위해 temperature를 0으로 설정합니다.
        router_chain = route_prompt | self.route_llm | JsonOutputParser()
        route = router_chain.invoke({"question": state["question"]})["route"]
        return {
            "question": state["question"],
            "generation": route.lower().strip(),
        }
