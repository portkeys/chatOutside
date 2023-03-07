import pinecone
import streamlit as st
import openai
import tiktoken
from streamlit_chat import message
import random

openai.api_key = "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"

pinecone_api_key = "5bd7c6ef-c6f1-426d-acb6-b1168078bd9d"

pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")

# ==== ChatOutside + Pinecone =============================
# ====== 1. Support function


def randomize_array(arr):
    sampled_arr = []
    while arr:
        elem = random.choice(arr)
        sampled_arr.append(elem)
        arr.remove(elem)
    return sampled_arr


def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens



MAX_SECTION_LEN = 2500 #in tokens
SEPARATOR = "\n"
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

dimension = 1536
# ====== 2. Pinecone function ====
EMBEDDING_MODEL = "text-embedding-ada-002"
index_name = "outside-chatgpt"
pineconeindex = pinecone.Index(index_name)

def get_embedding(text, model):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def construct_prompt_pinecone(question):
    """
    Fetch relevant information from pinecone DB
    """
    xq = get_embedding(question, EMBEDDING_MODEL)

    # print(xq)

    res = pineconeindex.query([xq], top_k=3,
                              include_metadata=True)  # , namespace="movies"

    # print(res)
    # print(most_relevant_document_sections[:2])

    chosen_sections = []
    chosen_sections_length = 0

    for match in res['matches']:
        # print(f"{match['score']:.2f}: {match['metadata']['text']}")
        if chosen_sections_length <= MAX_SECTION_LEN:
            document_section = match['metadata']['text']

            #   document_section = str(_[0] + _[1])
            chosen_sections.append(SEPARATOR + document_section)

            chosen_sections_length += num_tokens_from_string(
                str(document_section), "gpt2")

    # Useful diagnostic information
    # print(f"Selected {len(chosen_sections)} document sections:")

    header = """
    Sources:\n
    """
    return header + "".join(chosen_sections)


# ======3. chatOutside function
COMPLETIONS_API_PARAMS = {
        "temperature": 0.2,
        "max_tokens": 1000,
        "model": 'gpt-3.5-turbo',
    }

def chatoutside(query):
    # start chat with chatOutside
    source = construct_prompt_pinecone(query)
    prompt = source + "\n\n Q: " + query + "\n A:"

    try:
        response = openai.ChatCompletion.create(
            messages=[{"role": "system",
                       "content": "You are a friendly and helpful assistant and you are an expert with Outdoor activities."},
                      {"role": "user", "content": str(prompt)}],
            **COMPLETIONS_API_PARAMS
        )
    except Exception as e:
        print("I'm afraid your question failed! This is the error: ")
        print(e)
        return None

    choices = response.get("choices", [])
    if len(choices) > 0:
        #print(source)
        return choices[0]["message"]["content"]

    else:
        return None

# ============================================================

# ==== Section 1: Streamlit Settings ======
with st.sidebar:
    st.markdown("# About 🙌")
    st.markdown(
        "chatOutside allows you to talk to version of chatGPT \n"
        "that has access to latest Outside content about 6000 articles!  \n"
        )
    st.markdown(
        "Unlike chatGPT, chatOutside can't make stuff up\n"
        "and will only answer from injected knowledge 👩‍🏫 \n"
    )
    st.markdown("---")
    st.markdown("Prevent Large Language Model (LLM) hallucination - Wen Y.")

st.title("chatOutside: Outside + ChatGPT")

st.header("chatGPT 🤖")

# ====== Section 2: ChatGPT only ======
def summarize(prompt):
    st.session_state["summary"] = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system",
             "content": "You are a friendly and helpful assistant. Answer the question as truthfully as possible. If unsure, say you don't know."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )["choices"][0]["message"]["content"]

if "summary" not in st.session_state:
    st.session_state["summary"] = ""

input_text = st.text_input(label='Chat here! 💬')

st.button(
    "Submit",
    on_click=summarize,
    kwargs={"prompt": input_text},
)

output_text = st.text_area(label="Answered by chatGPT:",
                           value=st.session_state["summary"], height=250)

# ========= End of Section 2 ===========

# ========== Section 3: chatOutside ============================

st.header("chatOutside 🏕️")

# Storing the chat
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
    source = construct_prompt_pinecone(user_input)

    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    st.session_state.source.append(source)


# random user picture
user_av = random.randint(0, 100)

# random bott picture
bott_av = random.randint(0, 100)


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i],  key=str(i))  # seed=bott_av
        message(st.session_state["source"][i])
        message(st.session_state['past'][i], is_user=True,
                avatar_style="female",  key=str(i) + '_user') # seed=user_av

