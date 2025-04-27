import autogen
from datetime import datetime
import streamlit as st

llm_config = [{
    "model": "llama-3.3-70b-versatile",
    "api_key": "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",
    "api_type": "groq"
}]

date_str = datetime.now().strftime("%Y-%m-%d")

#######################################
##### Financial and Research Task #####
#######################################

financial_agent = autogen.AssistantAgent(
    name="Financial Assistant",
    llm_config={"config_list": llm_config}
)

research_agent = autogen.AssistantAgent(
    name="Research Assistant",
    llm_config={"config_list": llm_config}
)

########################
##### Writing Task #####
########################

writing_task = [
    """Develop an engaging financial report using all information provided, include the normalized.png figure and other figures if provided.
    Mainly rely on the information provided.
    Create a table of all the fundamental ratios and data.
    Provided a summary of the resent news for the stocks.
    Provide connections between the news headlines and the performance of the stocks.
    Provide an analysis of possible future scenarios."""
]

writer_system_message = """You are a professional writer known for your insightful and engaging financial reports.
You transform complex concepts into compelling narratives.
Include all metrics provided to you as context in your analysis.
Only return your final work without any additional comments."""

writing_agent = autogen.AssistantAgent(
    name="Writer",
    llm_config={"config_list": llm_config},
    system_message=writer_system_message
)

#################################
##### Refining the Blogpost #####
#################################

critic_system_message = """You are a critic. You review the work of the writer and provide constructive feedback to help improve the 
quality of the content."""

critic_agent = autogen.AssistantAgent(
    name="Critic",
    llm_config={"config_list": llm_config},
    system_message=critic_system_message
)

legal_rviewer_message = """You are a legal reviewer known for your ability to ensure that content is legally compliant and 
free from any potential legal issues. Make sure your suggestion is concise, concrete and to the point. Begin the review by
stating your role."""

legal_rviewer = autogen.AssistantAgent(
    name="Legal Reviewer",
    llm_config={"config_list": llm_config},
    system_message=legal_rviewer_message
)

textAllignment_reviewer_message = """You are a text data allignment reviewer known for your ability o ensure tha the meaning of the 
written content is alligned with the numbers written in the text. You must ensure that the text clearly describes the numbers provided 
in the text without contradiction. Make sure your suggestion is concise, concrete and to the point. Begin the review by stating your role."""
textAllignment_reviewer = autogen.AssistantAgent(
    name="Text Allignment Reviewer",
    llm_config={"config_list": llm_config},
    system_message=textAllignment_reviewer_message
)

completion_reviewer_message = """You are a content reviewer known for your ability to check that financial reports contain all the required 
elements. Always verify that the report contains a news report about each asset, a description of different ratios and prices,
a description of possible future scenarios, a table containing fundamental ratios and at least one figure. Make sure your suggestion is 
concise, concrete and to the point. Begin the review by stating your role.
"""
completion_reviewer = autogen.AssistantAgent(
    name="Completion Reviewer",
    llm_config={"config_list": llm_config},
    system_message=completion_reviewer_message
)

meta_reviewer_message = """You are a meta reviewer, you aggregate and review the work of the work of each reviewers and give a final
suggestion on the content."""

meta_reviewer = autogen.AssistantAgent(
    name="Meta Reviewer",
    llm_config={"config_list": llm_config},
    system_message=meta_reviewer_message
)

##############################################
##### Exporting the blogpost in markdown #####
##############################################

exporting_task = ["""Save the blogpost to a .md file using a python script."""]

export_agent = autogen.AssistantAgent(
    name="Exporter",
    llm_config={"config_list": llm_config}
)

user_proxy_agent = autogen.UserProxyAgent(
    name="User Proxy Agent",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").strip().endswith("TERMINATE"),
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "coding",
        "use_docker": False
    }
)

########################
##### Agentic Flow #####
########################

### Nested Chat Flow ###

def reflection_message(recipient, messages, sender, config):
    return f'''Review the following content.
    \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}'''

review_chats = [
    {"recipient": legal_rviewer, 
     "message": reflection_message,
     "summary_methods": "reflection_with_llm",
     "summary_args": {"summary_prompt": "Return review into a JSON object only: {'Reviewer':'', 'Review': ''}"},
     "max_turns": 1},
    {"recipient": textAllignment_reviewer, 
     "message": reflection_message,
     "summary_methods": "reflection_with_llm",
     "summary_args": {"summary_prompt": "Return review into a JSON object only: {'Reviewer':'', 'Review': ''}"},
     "max_turns": 1},
    {"recipient": completion_reviewer, 
     "message": reflection_message,
     "summary_methods": "reflection_with_llm",
     "summary_args": {"summary_prompt": "Return review into a JSON object only: {'Reviewer':'', 'Review': ''}"},
     "max_turns": 1},
    {"recipient": meta_reviewer, 
     "message": "Aggrgate feedback from all reviewers and give final suggestions on the writing.",
     "max_turns": 1},
]

critic_agent.register_nested_chats(
    review_chats,
    trigger=writing_agent
)

### Main Chat Flow ###

stocks = st.text_input("Which stock you want to analyze?")
hit_button = st.button("Start analysis")

if hit_button:
    financial_tasks = [
        f"""Today is the {date_str}.
        What are the current stock prices of {stocks}, how how the stock has been performing over the past 6 months in terms of percentage changes?
        Start by retrieving the full name of the stocks and use it for all future requests.
        Prepare a figure of the normalized price of these stocks and save it to a file named normalized.png.
        Retrieve all the fundamental ratios for the stocks.""",

        """Investigate the possible reasons of the stock performance leveraging market news from Bing News or Google Search.
        Retrieve news headlines using python and return them."""
    ]

    with st.spinner("Agents woring on the analysis..."):
        chat_result = autogen.initiate_chats(
            [
                {
                "sender": user_proxy_agent,
                "recipient": financial_agent,
                "message": financial_tasks[0],
                "summary_method": "reflection_with_llm",
                "summary_args":{"summary_prompt": "Return the stock prices of the stock, their performance and all the fundamental ratios"
                                "into a JSON object."},
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done"
                },
                {
                "sender": user_proxy_agent,
                "recipient": research_agent,
                "message": financial_tasks[1],
                "summary_method": "reflection_with_llm",
                "summary_args":{"summary_prompt": "Provide the news headlines for the stocks, be precide but do not consider news event that are vague,"
                                "return the result in a JSON object"},
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done"
                },
                {
                "sender": critic_agent,
                "recipient": writing_agent,
                "message": writing_task[0],
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done",
                "max_turns": 2,
                "summary_method": "last_msg"
                },
                {
                "sender": user_proxy_agent,
                "recipient": export_agent,
                "message": exporting_task[0],
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done",
                },
            ]
        )
    st.image("./codnig/normalized.png")
    st.markdown(chat_result[-1].chat_history[-1]["content"])

