# literary-style-changer(말투 변환 AI)
말투 변환 AI는 smilegate-ai님의 korean_smile_style_dataset을 사용하여 제작되었습니다. 해당 데이터셋을 gogamza님의 kobart-base-v2 모델에 미세조정(파인튜닝)시켜 말투 변환 AI를 만들었습니다. 학습 라이브러리는 Huggingface의 Transformers에 내장된 Seq2SeqTrainer 라이브러리를 사용했습니다.  
말투 변환 AI는 Streamlit Cloud를 지원하지 않습니다

# Installation
`git clone https://github.com/mangostin2010/literary-style-changer`  
`cd literary-style-changer`  
`pip install -r requirements.txt`  
`streamlit run convert-streamlit.py`  
