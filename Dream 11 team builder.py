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
import autogen
from datetime import datetime
import streamlit as st

llm_config = [{
    "model": "deepseek-r1-distill-llama-70b",
    "api_key": "",
    "api_type": "groq"
}]

date_str = datetime.now().strftime("%Y-%m-%d")


#######################################
##### Financial and Research Task #####
#######################################

web_search_tasks = f"""Today is the {date_str}.
    IPL(Indian Premier League) season is going on. What are all the cricket matches to be held today in IPL?
    Retrive name of all the players in the teams playing in the matches.
    Also retrieve the recent performance of all the players in a particular team, whether they are in a good form or not.
    Retrve their playing stats in all IPL seasons held till now.
    See in which venue the matches are going to be held, weather there, pitch condition and all other playing conditions.
    Retrieve all possible news of a particular match leveraging news from Bing News or Google Search."""

research_tasks = """Prepare the best Dream 11 fantasy team using players from both the team of a particular match,
    While preparing the team consider all possible factors like a player's recent performance, the playing stats
    of that player over the IPL seasons, the weather condition in the venue of the match etc.
    Make one player captain whom you think have the highest probability of good batting or bowling.
    Also ake one player vice-captain whom you think have the second highest probability of good batting or bowling.
    Provide the reason of chosing the players you kept in the team and also chosing captain and vice-captain"""

web_search_agent = autogen.AssistantAgent(
    name="Web Search Assistant",
    llm_config={"config_list": llm_config}
)

research_agent = autogen.AssistantAgent(
    name="Research Assistant",
    llm_config={"config_list": llm_config}
)


########################
##### Writing Task #####
########################

writing_task = """Develop an engaging report using all information provided.
    Provide the reason of chosing the players you kept in the team and also chosing captain and vice-captain"""

writer_system_message = """You are a professional writer known for your insightful and engaging reports.
You transform complex concepts into compelling narratives.
Only return your final work without any additional comments."""

writing_agent = autogen.AssistantAgent(
    name="Writer",
    llm_config={"config_list": llm_config},
    system_message=writer_system_message
)


#################################
##### Refining the content #####
#################################

critic_task = """Review the work of the writer, check whether all the informaton is correct
    in the present condition and provide constructive feedback to help improve the quality of the content."""

critic_system_message = """You are a critic. You review the work of the writer, check whether all the informaton is correct
in the present condition and provide constructive feedback to help improve the quality of the content."""

critic_agent = autogen.AssistantAgent(
    name="Critic",
    llm_config={"config_list": llm_config},
    system_message=critic_system_message
)


##############################################
##### Exporting the content in markdown #####
##############################################

exporting_task = """Save the blogpost to a .md file using a python script."""

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

# review_chats = [
#     {"recipient": legal_rviewer, 
#      "message": reflection_message,
#      "summary_methods": "reflection_with_llm",
#      "summary_args": {"summary_prompt": "Return review into a JSON object only: {'Reviewer':'', 'Review': ''}"},
#      "max_turns": 1},
#     {"recipient": textAllignment_reviewer, 
#      "message": reflection_message,
#      "summary_methods": "reflection_with_llm",
#      "summary_args": {"summary_prompt": "Return review into a JSON object only: {'Reviewer':'', 'Review': ''}"},
#      "max_turns": 1},
#     {"recipient": completion_reviewer, 
#      "message": reflection_message,
#      "summary_methods": "reflection_with_llm",
#      "summary_args": {"summary_prompt": "Return review into a JSON object only: {'Reviewer':'', 'Review': ''}"},
#      "max_turns": 1},
#     {"recipient": meta_reviewer, 
#      "message": "Aggrgate feedback from all reviewers and give final suggestions on the writing.",
#      "max_turns": 1},
# ]

# critic_agent.register_nested_chats(
#     review_chats,
#     trigger=writing_agent
# )

### Main Chat Flow ###

hit_button = st.button("Start analysis")

if hit_button:

    with st.spinner("Agents woring on the analysis..."):
        chat_result = autogen.initiate_chats(
            [
                {
                "sender": user_proxy_agent,
                "recipient": web_search_agent,
                "message": web_search_tasks,
                "summary_method": "reflection_with_llm",
                "summary_args":{"summary_prompt": "Provide all matches today, teams, players, their performance, weather conditions"
                                "into a JSON object."},
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done"
                },
                {
                "sender": user_proxy_agent,
                "recipient": research_agent,
                "message": research_tasks,
                "summary_method": "reflection_with_llm",
                "summary_args":{"summary_prompt": "Provide the reason of chosing the players you kept in the team and also chosing captain and vice-captain,"
                                "return the result in a JSON object"},
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done"
                },
                {
                "sender": user_proxy_agent,
                "recipient": writing_agent,
                "message": writing_task,
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done",
                "max_turns": 2,
                "summary_method": "last_msg"
                },
                {
                "sender": user_proxy_agent,
                "recipient": critic_agent,
                "message": critic_task,
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done",
                "max_turns": 2,
                "summary_method": "last_msg"
                },
                {
                "sender": user_proxy_agent,
                "recipient": export_agent,
                "message": exporting_task,
                "carryover": "Wait for the confirmation of code execution before terminatig the conversation. Reply TERMINATE in the end when everuthing is done",
                },
            ]
        )
    st.markdown(chat_result[-1].chat_history[-1]["content"])
