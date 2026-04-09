import streamlit as st

st.set_page_config(page_title="Home", page_icon=":house:")

st.title("LangGraph ChatBot")

st.markdown(
    """
    이번 실습에선 엑셀 데이터와 PDF 파일을 모두 활용하여 답변하는 챗봇을 구현해보도록 하겠습니다.  

    `LangGraph`를 활용하면 다양한 기능을 지원하는 유연한 챗봇을 구현할 수 있습니다. 
    
    이론강의에서 확인해 보았던 `LangCraph` 챗봇 모델을 실제로 구현해보겠습니다. 그래프의 구성 요소와 역할을 기억해보세요.
    """
)
