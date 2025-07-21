import streamlit as st
import datetime
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, RetrievalQA

# --- HyperCLOVA Wrapper ---
class HyperCLOVALLM:
    def __init__(self, api_key_id, api_key, model="hyperclova-82b"):
        self.headers = {
            "Content-Type": "application/json",
            "X-NCP-APIGW-API-KEY-ID": api_key_id,
            "X-NCP-APIGW-API-KEY": api_key
        }
        self.url = "https://naveropenapi.apigw.ntruss.com/llm/v1/t2t"
        self.model = model

    def generate(self, prompt, max_tokens=512, temperature=0.7):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "maxTokens": max_tokens,
            "temperature": temperature
        }
        resp = requests.post(self.url, headers=self.headers, json=payload)
        resp.raise_for_status()
        return resp.json()["completion"]

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)


# --- Resources Loader ---
@st.cache_resource
def load_resources(report_url):
    loader = UnstructuredURLLoader(urls=[report_url])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return retriever, docs


# --- Prompt Template for Q&A ---
question_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""아래는 선택된 보고서 내용입니다:
\"\"\"{context}\"\"\"

질문: \"{question}\"
위 정보를 바탕으로 자세히 답변하세요.
"""
)


# --- Search & Recommend (NewsAPI) ---
def search_recommendations(profile, news_api_key, top_n=3):
    client = NewsApiClient(api_key=news_api_key)
    articles = client.get_top_headlines(q=profile, language='ko', page_size=10)
    titles = [a["title"] for a in articles.get("articles", [])]
    return titles[:top_n]


# --- Streamlit UI ---
st.set_page_config(page_title="🤖 HyperCLOVA 투자 챗봇", layout="wide")
st.title("🤖 HyperCLOVA 투자 챗봇 (채팅형 UI)")

# --- Sidebar Settings ---
st.sidebar.header("설정")
api_key_id    = st.sidebar.text_input("HyperCLOVA API Key ID", type="password")
api_key       = st.sidebar.text_input("HyperCLOVA API Key",    type="password")
news_api_key  = st.sidebar.text_input("NewsAPI Key",           type="password")
report_list_url = st.sidebar.text_input("보고서 목록 페이지 URL 입력")

# --- Report Selection from List Page ---
report_url = None
if report_list_url:
    try:
        res  = requests.get(report_list_url)
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.select('a[href$=".pdf"]')
        options = {
            (link.get_text(strip=True) or link["href"]):
            (link["href"] if link["href"].startswith("http")
             else requests.compat.urljoin(report_list_url, link["href"]))
            for link in links
        }
        if options:
            selected = st.sidebar.selectbox("보고서 선택", list(options.keys()))
            report_url = options[selected]
            st.sidebar.markdown(f"**선택된 보고서**: [{selected}]({report_url})")
        else:
            st.sidebar.warning("PDF 보고서를 찾을 수 없습니다.")
    except Exception as e:
        st.sidebar.error(f"목록 페이지 로드 실패: {e}")

# --- Initialize Resources ---
if report_url and api_key_id and api_key:
    retriever, docs = load_resources(report_url)
    hc_llm = HyperCLOVALLM(api_key_id, api_key)
else:
    retriever = None
    hc_llm = None

# --- Chat History Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Chat Messages ---
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- User Input Handling ---
user_input = st.chat_input("질문이나 요청을 입력하세요…")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 투자유형 기반 인터넷 종목 추천
    if user_input.startswith("투자유형:") and news_api_key:
        profile = user_input.split("투자유형:", 1)[1].strip()
        recs = search_recommendations(profile, news_api_key)
        rec_text = "\n".join([f"- {t}" for t in recs])
        answer = (
            f"**{profile} 투자유형 추천 종목 (뉴스API 기반 Top{len(recs)}):**\n{rec_text}\n\n"
            "보고서 기반 Q&A를 원하시면 자유롭게 질문하세요!"
        )

    # 보고서 기반 Q&A
    elif retriever and hc_llm:
        with st.spinner("응답 생성 중…"):
            qa_chain = RetrievalQA.from_chain_type(
                llm=hc_llm,
                chain_type="map_reduce",
                retriever=retriever,
                return_source_documents=False,
                combine_prompt=question_prompt
            )
            answer = qa_chain.run(question=user_input)

    else:
        answer = "먼저 사이드바에 보고서 목록 URL, HyperCLOVA API Key, NewsAPI Key를 모두 입력해주세요."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.experimental_rerun()
