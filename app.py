import streamlit as st
import os
from self_discover import SelfDiscover

st.set_page_config(
    page_title="SELF-DISCOVER",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("SELF-DISCOVER")

# Add a toggle to choose between local LLM and OpenAI
use_local_llm = st.checkbox("Use Local LLM (Gemma3)", value=False)

if not use_local_llm:
    api_key = st.text_input("Enter OpenAI API key (optional if using local LLM)")

task = st.text_area("Enter the task example you want to generate a reasoning structure for ")

if st.button("Generate Reasoning Structure"):
    if not use_local_llm and not api_key:
        st.error("Please provide an OpenAI API key or enable the local LLM option.")
    else:
        if not use_local_llm:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            os.environ["USE_LOCAL_LLM"] = "true"  # Set the environment variable for local LLM

        result = SelfDiscover(task)
        result()
        tab1, tab2, tab3 = st.tabs(["SELECTED_MODULES", "ADAPTED_MODULES", "REASONING_STRUCTURE"])
        with tab1:
            st.header("SELECTED_MODULES")
            st.write(result.selected_modules)

        with tab2:
            st.header("ADAPTED_MODULES") 
            st.write(result.adapted_modules)

        with tab3:
            st.header("REASONING_STRUCTURE") 
            st.write(result.reasoning_structure)
else:
    st.info("Please provide a task example and optionally an API key or enable the local LLM option.")