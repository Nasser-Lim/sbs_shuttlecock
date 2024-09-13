import streamlit as st
import json
import requests
import urllib.parse
import re
import anthropic
            
from langchain_anthropic.chat_models import ChatAnthropic as ChatAnthropicModel
from langchain.schema import HumanMessage
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from custom_json_loader import JSONLoader

def clean_text(text):
    # <!HS>와 <!HE> 태그 제거
    text = re.sub(r"<!HS>|<!HE>", "", text)
    # <h4> 태그와 그 내용을 제거
    text = re.sub(r'<h4[^>]*>.*?</h4>', '', text, flags=re.DOTALL)
    # 연속된 개행 문자를 하나의 개행으로 축소
    text = re.sub(r'\n+', '\n', text)
    return text

def fetch_news_data(query, limit):
    # 쿼리 인코딩
    # encoded_query = urllib.parse.quote(query)
    # API URL 구성
    url = f"https://searchapi.news.sbs.co.kr/search/news?query={query}&collection=news_sbs&offset=0&limit={limit}"

    # API 요청
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()  # JSON 데이터 파싱
        articles = data.get('news_sbs', [])
        
        # 필요한 정보 추출
        extracted_data = []
        for article in articles:
            title = clean_text(article.get('TITLE', ''))
            article_text = clean_text(article.get('REDUCE_CONTENTS', ''))
            article_date = clean_text(article.get('DATE', ''))
            link = f"https://news.sbs.co.kr/news/endPage.do?news_id={article.get('DOCID')}"
            
            if title:  # 제목이 있는 경우만 추가
                extracted_data.append({
                    'title': title,
                    'article': article_text,
                    'date': article_date,
                    'link': link
                })
        return extracted_data
    else:
        return f"Error fetching data: {response.status_code}"

def format_response(response):
    # 응답에서 쉼표, 점, 그리고 다른 구분자들을 제거
    cleaned_response = re.sub(r'[^\w\s]', '', response)
    words = cleaned_response.split()
    filtered_words = [word for word in words if word not in ('키워드', '검색어')]
    # 첫 두 개의 단어만 선택
    if len(filtered_words) >= 2:
        return '%20'.join(filtered_words[:2])
    elif len(filtered_words) == 1:
        return filtered_words[0]
    return ""

def load_and_split_documents(file_path):
    # JSONLoader를 사용하여 JSON 파일 로드
    loader = JSONLoader(file_path=file_path, text_content=False)
    docs = loader.load()
    
    # 각 Document 객체의 metadata에 source 정보 추가
    for doc in docs:
        content = json.loads(doc.page_content)
        doc.metadata['source'] = content['link']
    
    # RecursiveCharacterTextSplitter를 사용하여 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=['\n\n', '\n', ' ', '']
    )
    
    split_docs = []
    for doc in docs:
        content = json.loads(doc.page_content)
        splits = text_splitter.split_text(content['article'])
        
        for i, split in enumerate(splits):
            metadata = {
                'source': doc.metadata['source'],
                'seq_num': doc.metadata['seq_num'],
                'split_idx': i,
                'title': content['title']
            }
            split_docs.append(Document(page_content=split, metadata=metadata))
    
    return split_docs

def create_vector_store(documents):
    # OpenAI 임베딩 생성
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"], model="text-embedding-3-small")
    
    if not documents:
        return None
    
    # FAISS를 사용하여 벡터 저장소 생성
    vector_store = FAISS.from_documents(documents, embeddings)
    
    return vector_store

def retrieve_relevant_documents(query, vector_store, k=5):
    # 쿼리와 관련된 문서 검색
    relevant_docs = vector_store.similarity_search(query, k=k)
    
    return relevant_docs

#########################################################################################

# 모바일 디바이스 여부 확인
is_mobile = "mobile" in st.query_params

# 디바이스에 따라 다른 타이틀 출력
if is_mobile:
    st.title("🏸셔틀콕")
else:
    st.title("🏸셔틀콕 ― 새로운 AI 지식검색")

# 사용자와 챗봇 간의 대화 메시지를 저장하기 위해 session state를 확인하고 초기화합니다.
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "안녕하세요? '셔틀콕'은 믿을 수 있는 SBS 뉴스를 바탕으로 사용자 질문 맥락을 콕~ 짚어 이해하고 답변하는 AI 챗봇입니다. 생성한 문장마다 출처를 표시해 신뢰도를 높였습니다.\n\n제가 무엇을 도와드릴까요?"}
    ]

# session state에 저장된 모든 메시지를 반복하여 화면에 표시합니다.
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자로부터 새로운 입력을 받는 필드를 생성합니다. 여기서는 예시 질문을 placeholder로 제시합니다.
if prompt := st.chat_input(placeholder="최근 삼성전자 실적을 알려줘."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.spinner('sbs 뉴스를 검색합니다..'):
        anthropic_api_key = st.secrets["CLAUDE_API_KEY"]
        llm = ChatAnthropicModel(model="claude-3-haiku-20240307", temperature=0.0, max_tokens=64, anthropic_api_key=anthropic_api_key)

        search_prompt = f"""
        사용자 질문: {prompt}

        위 질문에 대한 SBS 뉴스 검색 키워드를 추출하세요:
        1. 질문의 핵심 주제와 관련된 1-2개의 키워드 선정
        2. 가능한 고유명사 위주로 선별, 없으면 중요 일반명사 선택
        3. 키워드 2개면 공백으로 구분

        출력 형식: 키워드1 키워드2
        """.strip()

        formatted_prompt = llm.invoke([HumanMessage(content=search_prompt)])
        search_keyword = format_response(formatted_prompt.content)

        # st.write("후처리 키워드: " + search_keyword)
        
        fetched_news_data = fetch_news_data(search_keyword, limit=40)

        # fetched_news_data를 직접 문서 객체로 변환
        docs = [Document(page_content=article['article'], metadata={'title': article['title'], 'date': article['date'], 'source': article['link']}) for article in fetched_news_data]
        
        # 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        # 벡터 저장소 생성
        vector_store = create_vector_store(split_docs)
        
        if vector_store is None:
            empty_answer = "질문하신 내용으로 검색된 sbs 뉴스가 없습니다."
            st.session_state.messages.append({"role": "assistant", "content": empty_answer})
            st.write(empty_answer)
            skip_answer = True
        else:
            # 관련 문서 검색 및 통합
            relevant_docs = retrieve_relevant_documents(prompt, vector_store)

            # 관련 문서에 번호 붙이기
            numbered_docs = [f"{doc.page_content} (기사 작성날짜: {doc.metadata['date']}) ({i+1})" for i, doc in enumerate(relevant_docs)]
            
            # 관련 문서 합치기
            combined_docs = "\n".join(numbered_docs)
            skip_answer = False

    if not skip_answer:
        # 챗봇으로부터의 응답을 처리하기 위한 Streamlit 컨테이너를 생성합니다.
        with st.chat_message("assistant"):
                
            # 최종 답변 생성
            answer_prompt = f"""
            사용자 질문: {prompt}

            관련 뉴스 정보:
            {combined_docs}

            지시사항:
            1. 위 정보를 바탕으로 사용자 질문에 대한 답변을 상세하게 작성하세요.
            2. 답변은 핵심 내용을 먼저 제시하는 두괄식으로 작성하세요.
            3. 정보가 부족하면 "관련 정보가 충분하지 않습니다"라고 답하세요.
            4. '오늘', '어제' 등의 상대적 날짜는 구체적인 날짜로 변환하세요.
            5. 각 문장 끝에 출처를 [1], [2] 형식으로 표시하세요. 여러 출처는 [3][4] 형식으로 표시하세요.

            답변:
            """

            client = anthropic.Anthropic(api_key=anthropic_api_key)

            def stream_response():
                with client.messages.stream(
                    max_tokens=2048,
                    messages=[{"role": "user", "content": answer_prompt}],
                    system="You are a helpful assistant who answers in Korean.",
                    model="claude-3-haiku-20240307",
                    temperature=0.0,
                ) as stream:
                    for text in stream.text_stream:
                        yield text

            response = st.write_stream(stream_response())
            st.session_state.messages.append({"role": "assistant", "content": response})

            annotations = re.findall(r'(\[\d+\])', response)
            if annotations:
                source = "뉴스 출처:\n\n"
                unique_annotations = list(set(annotations))  # 주석 중복 제거
                for annotation in unique_annotations:
                    doc_index = int(annotation[1:-1]) - 1  # 대괄호 제거 후 숫자 추출
                    if 0 <= doc_index < len(relevant_docs):
                        doc = relevant_docs[doc_index]
                        source += f"{annotation} {doc.metadata['title']}\n\n{doc.metadata['source']}\n\n"
                st.markdown(source)
                st.session_state.messages.append({"role": "assistant", "content": re.sub(r'<[^>]+>', '', source)})
