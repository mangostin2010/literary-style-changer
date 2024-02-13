from transformers import pipeline
import pandas as pd
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os

with open('style.css') as f: 
    st.markdown(f"<style>{f.read()}<style>", unsafe_allow_html=True)

style_map = {
    'formal': '문어체',
    'informal': '구어체',
    'android': '안드로이드',
    'azae': '아재',
    'chat': '채팅',
    'choding': '초등학생',
    'emoticon': '이모티콘',
    'enfp': 'enfp',
    'gentle': '신사',
    'halbae': '할아버지',
    'halmae': '할머니',
    'joongding': '중학생',
    'king': '왕',
    'naruto': '나루토',
    'seonbi': '선비',
    'sosim': '소심한',
    'translator': '번역기'
}

df = pd.read_csv("smilestyle_dataset.tsv", sep="\t")

model_path = 'text-transfer-smilegate-bart-eos/'
tokenizer_name = 'gogamza/kobart-base-v2'


def generate_text(pipe, text, target_style, num_return_sequences=5, max_length=60):
    target_style_name = style_map[target_style]
    text = f"{target_style_name} 말투로 변환:{text}"
    out = pipe(text, num_return_sequences=num_return_sequences, max_length=max_length)
    return [x['generated_text'] for x in out]

def load_model():
    if not os.path.exists("/text-transfer-smilegate-bart-eos"): 
        os.system('git clone https://huggingface.co/mangostin2010/text-transfer-smilegate-bart-eos')
    if 'loaded' not in st.session_state:
        st.session_state.pipeline = pipeline('text2text-generation', model=model_path, tokenizer=tokenizer_name)


# 원하는 스타일 리스트
target_styles = ['enfp']
target_styles = df.columns


st.title('말투 변환 AI')
st.markdown('말투 변환 AI는 **smilegate-ai**님의 **korean_smile_style_dataset**을 사용하여 제작되었습니다. 해당 데이터셋을 **gogamza**님의 **kobart-base-v2** 모델에 미세조정(파인튜닝)시켜 말투 변환 AI를 만들었습니다. 학습 라이브러리는 **Huggingface**의 **Transformers**에 내장된 **Seq2SeqTrainer** 라이브러리를 사용했습니다.')
st.divider()
with st.spinner('모델 로드중'):
    if st.button('변환 모델 로드하기'):
        load_model()
        st.success('모델 로드 완료')

src_text = st.text_input('변환될 텍스트')

if st.button('변환하기'):
    try:
        for style in target_styles:
            result =  generate_text(st.session_state.pipeline, src_text, style, num_return_sequences=1, max_length=60)[0]
            with stylable_container(
                key='result_fsdjklajsdkf', 
                css_styles='''{ 
                    p {
                        margin: 0px 0px 0px; 
                    }
                    }'''
                ):
                    st.write(f"**<{style}>**", result)
            

    except Exception as e: 
        st.error('변환 모델을 로드해 주세요')