import os
import pandas as pd
from PIL import Image
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
# import supporting funcitons
from utils import *

openai.api_key = st.secrets["OPENAI_KEY"]
# For Langchain
os.environ["OPENAI_API_KEY"] = openai.api_key


# ==== Section 1: Streamlit Settings ======
with st.sidebar:
    st.markdown("# About 🙌")
    st.markdown(
        "chatOutside allows you to talk to version of chatGPT \n"
        "that has access to latest Outside content!  \n"
        )
    st.markdown(
        "Unlike chatGPT, chatOutside can't make stuff up\n"
        "and will answer from Outside knowledge base. \n"
    )
    st.markdown("👩‍🏫 Developer: Wen Yang")
    st.markdown("---")
    st.markdown("# Under The Hood 🎩 🐇")
    st.markdown("How to Prevent Large Language Model (LLM) hallucination?")
    st.markdown("Pinecone: vector database for Outside knowledge")
    st.markdown("Langchain: to remember the context of the conversation")


st.title("chatOutside: Outside + ChatGPT")

image = Image.open('/Users/wen/Downloads/VideoBkg_08.jpg')

st.image(image, caption='Get Outside!')

st.header("chatGPT 🤖")


# ====== Section 2: ChatGPT only ======
def chatgpt(prompt):
    res = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system",
             "content": "You are a friendly and helpful assistant. "
                        "Answer the question as truthfully as possible. "
                        "If unsure, say you don't know."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )["choices"][0]["message"]["content"]

    return res


input_gpt = st.text_input(label='Chat here! 💬')
output_gpt = st.text_area(label="Answered by chatGPT:",
                          value=chatgpt(input_gpt), height=200)

# ========= End of Section 2 ===========

# ========== Section 3: chatOutside ============================

st.header("chatOutside 🏕️")

# -------3.1 Langchain to remember conversation context --------


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = OpenAI(temperature=0)
    chain = ConversationChain(llm=llm)
    return chain


chain = load_chain()
# ---------------------------------------

# -------3.2 chatOutside function -------------------
COMPLETIONS_API_PARAMS = {
        "temperature": 0.2,
        "max_tokens": 1000,
        "model": 'gpt-3.5-turbo',
    }


def chatoutside(query):
    # start chat with chatOutside
    source_text, source_urls = construct_prompt_pinecone(query)
    prompt = source_text + "\n\n Q: " + query + "\n A:"

    try:
        response = openai.ChatCompletion.create(
            messages=[{"role": "system",
                       "content": "You are a friendly and helpful assistant. "
                                  "You are an expert on Outdoor activities "
                                  "such as hiking, climbing, cycling, yoga "
                                  "and you are aware of latest trend in "
                                  "outdoor gears, clothing and shoes."},
                      {"role": "user", "content": str(prompt)}],
            **COMPLETIONS_API_PARAMS
        )
    except Exception as e:
        print("I'm afraid your question failed! This is the error: ")
        print(e)
        return None

    choices = response.get("choices", [])
    if len(choices) > 0:
        return choices[0]["message"]["content"]

    else:
        return None

# ============================================================


# ========== 4. Display ChatOutside in chatbot style ===========
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'source' not in st.session_state:
    st.session_state['source'] = []


def clear_text():
    st.session_state["input"] = ""


# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input('Chat here! 💬', key="input")
    return input_text


user_input = get_text()

if user_input:
    output = chatoutside(user_input)
    # source contain urls from Outside
    _, source = construct_prompt_pinecone(user_input)

    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    st.session_state.source.append(source)

    # Use df to display source urls
    link1 = source[0]
    link2 = source[1]
    link3 = source[2]
    df = pd.DataFrame(
        {
            "source_url": [
                f'<a target="_blank" href="{link1}">{link1}</a>',
                f'<a target="_blank" href="{link2}">{link2}</a>',
                f'<a target="_blank" href="{link3}">{link3}</a>'
            ]
        }
    )

    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i],  key=str(i))  # seed=bott_av
        message(st.session_state['past'][i], is_user=True,
                avatar_style="big-ears",  key=str(i) + '_user')  # seed=user_av



