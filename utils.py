import random
import tiktoken
import pinecone
import openai
import streamlit as st

# Support Functions
# --- for OpenAI ------
MAX_SECTION_LEN = 2500  # in tokens
SEPARATOR = "\n"
ENCODING = "cl100k_base"  # encoding for text-embedding-ada-002

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

dimension = 1536
EMBEDDING_MODEL = "text-embedding-ada-002"

# --- for Pinecone ------
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
index_name = "outside-chatgpt"
pineconeindex = pinecone.Index(index_name)


def get_embedding(text, model):
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def num_tokens_from_string(string, encoding_name):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def randomize_array(arr):
    sampled_arr = []
    while arr:
        elem = random.choice(arr)
        sampled_arr.append(elem)
        arr.remove(elem)
    return sampled_arr


# ====== Pinecone function ====
def construct_prompt_pinecone(question):
    """
    Fetch relevant information from pinecone DB
    """
    xq = get_embedding(question, EMBEDDING_MODEL)

    res = pineconeindex.query([xq], top_k=3,
                              include_metadata=True)  # , namespace="movies"

    chosen_sections = []
    chosen_sections_length = 0

    for match in res['matches']:
        # select relevant section
        if chosen_sections_length <= MAX_SECTION_LEN:
            document_section = match['metadata']['text']

            #   document_section = str(_[0] + _[1])
            chosen_sections.append(SEPARATOR + document_section)

            chosen_sections_length += num_tokens_from_string(
                str(document_section), "gpt2")

    # Add urls as sources
    sources = [
        x['metadata']['url'] for x in res['matches']
    ]

    header = f"""
    Sources:\n 
    """
    return header + "".join(chosen_sections), sources