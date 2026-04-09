import os

import pandas as pd
import streamlit as st
from utils.custom_chatbot import ExcelPDFChatbot

# page title
st.set_page_config(page_title="🦜 엑셀 데이터와 PDF 문서 기반 챗봇")
st.title("🦜 엑셀 데이터와 PDF 문서를 모두 활용하는 챗봇")

file_path = "./pages/data/RE177_2023년 국내외 인공지능 산업 동향 연구.pdf"
file_description = "인공지능 산업 동향"
data_path = "./pages/data/도로교통공단_사고유형별_교통사고_통계_20231231.csv"
data_description = "교통사고 통계"
df = pd.read_csv(data_path, encoding="cp949")


@st.cache_resource
def init_chatbot():
    chatbot = ExcelPDFChatbot(
        df,
        data_description,
        file_path,
        file_description,
    )
    return chatbot


# Streamlit app은 app code를 계속 처음부터 재실행하는 방식으로 페이지를 갱신합니다.
# Chatbot을 state에 포함시키지 않으면 매 질문마다 chatbot을 다시 초기화 합니다.
if "chatbot" not in st.session_state:
    with st.spinner("챗봇을 초기화 중입니다."):
        chatbot = init_chatbot()
        st.session_state.chatbot = chatbot
    st.write("챗봇 초기화를 완료했습니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    **인공지능 산업 동향 연구보고서**와 **교통사고 통계 데이터**를 기반으로 답변하는 챗봇을 이용해보세요!
    * 인공지능 산업 동향 연구보고서 출처 - 소프트웨어정책연구소, [링크](https://spri.kr/posts/view/23728?code=data_all&study_type=research#none)
    * 교통사고 통계 데이터 출처 - 교통사고분석시스템, [링크](http://taas.koroad.or.kr)
    ---
    """
)


st.markdown(
    """
- 예시 질문 (인공지능 산업 동향 연구보고서): 인공지능에 투자를 많이 하고 있는 기업들을 알려줘
- 예시 질문 (교통사고 현황 데이터 활용): 사고유형대분류가 차대차인 사고건수를 알려줘
- 예시 질문 (교통사고 현황 데이터 시각화): 사고유형 대분류에 따른 중상자수를 파이차트로 그려줘
- 예시 질문 (데이터 무관): 저녁 메뉴 추천해줘
"""
)

# 데이터프레임을 접을 수 있는 버튼으로 확인할 수 있게 설정
with st.expander("📊 교통사고 통계 데이터 보기"):
    st.write("아래 표는 교통사고 통계 데이터입니다:")
    st.dataframe(df)

for conversation in st.session_state.messages:
    with st.chat_message(conversation["role"]):
        st.write(conversation["content"])

if prompt := st.chat_input("질문을 입력하면 챗봇이 답변을 제공합니다."):
    # 메시지를 채팅UI로 표현
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )  # 채팅 기록 추가

if prompt is not None:
    response = st.session_state.chatbot.invoke(prompt)  # 답변 생성
    generation = response["generation"]
    with st.chat_message("assistant"):
        st.markdown(generation)
        if "data" in response.keys() and response["data"] == "plot.png":
            st.image("plot.png")
            os.remove("plot.png")  # 이미지를 보여준 다음 삭제

    st.session_state.messages.append({"role": "assistant", "content": generation})
