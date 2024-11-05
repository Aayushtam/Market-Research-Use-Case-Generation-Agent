import streamlit as st
import warnings
from crewai import Agent, Task, Crew, LLM
from IPython.display import Markdown

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up Ollama model configuration for local API use
ollama_base_url = "http://localhost:11434"

# Define agents
market_research_agent = Agent(
    role="Market Research Agent",
    goal="Conduct detailed and focused market research on {industry} for {company_name}, using factual data from credible sources.",
    backstory=(
        "You are a skilled market research analyst specializing in the {industry} sector, tasked with uncovering insights specific to {company_name}. "
        "Your role includes identifying key trends, competitors, and challenges in the industry, and analyzing potential AI/GenAI applications. "
        "Retain key findings on industry trends and competitors for sharing with other agents. "
        "You have access to tools like a Web Browser for real-time research. Stay objective, avoid assumptions, and ensure insights are actionable. "
        "Coordinate with the Use Case Generation Agent to provide foundational information, and rely on up-to-date, verified sources only."
    ),
    allow_delegation=False,
    verbose=True,
    llm=LLM(model="ollama/phi3:14b", base_url=ollama_base_url)
)

use_case_agent = Agent(
    role="Use Case Generator",
    goal="Develop relevant AI and GenAI use cases based on Market Research findings.",
    backstory=(
        "As an AI Use Case Specialist, you analyze market research and propose use cases in line with {company_name}'s objectives in the {industry} sector. "
        "Your task is to create practical use cases for AI and GenAI that address {company_name}'s operational and customer-focused needs, using insights shared by the Market Research Agent. "
        "You specialize in AI and ML applications and use GenAI frameworks to conceptualize industry-relevant solutions. "
        "Your output should focus on realistic, beneficial use cases directly aligned with {company_name}'s strategic goals. "
        "Work closely with the Resource Asset Collector, ensuring each use case is supported by relevant datasets and resources. Avoid speculative ideas and ensure feasibility."
    ),
    allow_delegation=False,
    verbose=True,
    llm=LLM(model="ollama/phi3:14b", base_url=ollama_base_url)
)

resource_collection_agent = Agent(
    role="Resource Asset Collector",
    goal="Gather relevant datasets, pre-trained models, and resources for use cases in {industry}.",
    backstory=(
        "You specialize in finding resources to support AI/GenAI projects, including datasets, pre-trained models, and code repositories. "
        "Your focus is to collect resources that directly support the AI/GenAI use cases developed by the Use Case Generation Agent. "
        "You search trusted platforms like Kaggle, HuggingFace, and GitHub, ensuring resources are well-documented and accessible. "
        "Coordinate with the Use Case Generator for any additional resource needs and verify that links are accurate. "
        "Prioritize high-quality, reputable resources, and validate each link for relevance and accessibility to ensure the best support for {company_name}'s projects."
    ),
    allow_delegation=False,
    verbose=True,
    llm=LLM(model="ollama/phi3:14b", base_url=ollama_base_url)
)

final_proposal_agent = Agent(
    role="Final Proposal Agent",
    goal="Compile a cohesive proposal listing AI/GenAI use cases and supporting resources aligned with {company_name}'s goals.",
    backstory=(
        "As a Proposal Specialist, you synthesize the work of all agents into a professional and cohesive proposal tailored for {company_name}. "
        "Ensure all use cases and resources are presented in an organized and actionable format."
    ),
    allow_delegation=False,
    verbose=True,
    llm=LLM(model="ollama/phi3:14b", base_url=ollama_base_url)
)

# Define tasks
market_research_task = Task(
    description=(
        "1. Research the latest trends, competitors, and challenges in the {industry} industry.\n"
        "2. Identify {company_name}'s key offerings and strategic focus areas like operations, supply chain, and customer experience.\n"
        "3. Summarize insights relevant to AI applications."
    ),
    expected_output="Comprehensive report on industry trends, competitors, and AI opportunities for {industry}.",
    agent=market_research_agent,
)

use_case_task = Task(
    description=(
        "1. Analyze findings from Market Research.\n"
        "2. Generate a list of AI/GenAI use cases, each tailored to address industry trends and {company_name}'s goals.\n"
        "3. Include details on each use case’s objective, potential AI/GenAI application, and expected impact."
    ),
    expected_output="Detailed list of AI/GenAI use cases, focusing on enhancing {company_name}'s operations and customer satisfaction.",
    agent=use_case_agent,
)

resource_collection_task = Task(
    description=(
        "1. Search for relevant datasets on Kaggle, HuggingFace, and GitHub for each use case.\n"
        "2. Compile resources in Markdown format as follows:\n"
        "Thought: [Current reasoning]\n"
        "Final Answer: Here is the structured list:\n"
        "- [Resource name](URL)\n"
        "Ensure all links are clickable and correct."
    ),
    expected_output="Final Answer: Here is the structured list of datasets and resources:\n - [Dataset](URL) - [Model](URL) etc.",
    agent=resource_collection_agent,
)

final_proposal_task = Task(
    description=(
        "Compile a cohesive, structured proposal that aligns with {company_name}'s goals. "
        "Organize the proposal with the following sections:\n\n"
        "1. **Introduction**: Briefly summarize the purpose and scope of the proposal.\n"
        "2. **Market Research Insights**: Summarize findings from the Market Research Agent on industry trends and competitors.\n"
        "3. **Use Case Suggestions**: List AI/GenAI use cases identified by the Use Case Generation Agent, including each use case’s objective, AI application, and potential impact.\n"
        "4. **Resource Links**: Provide clickable links to datasets, models, and resources from the Resource Asset Collector, formatted as follows:\n"
        "   - [Resource Name](URL)\n\n"
        "Ensure each section is clear, concise, and actionable, following this exact format:\n"
        "Thought: [Current reasoning]\n"
        "Final Answer: [Complete, structured proposal in markdown]"
    ),
    expected_output="Final Answer: [Formatted proposal in markdown with Introduction, Market Research Insights, Use Case Suggestions, Resource Links]",
    agent=final_proposal_agent,
)

# Assemble the crew
research_crew = Crew(
    agents=[market_research_agent, use_case_agent, resource_collection_agent, final_proposal_agent],
    tasks=[market_research_task, use_case_task, resource_collection_task, final_proposal_task],
    verbose=True
)

# Streamlit Interface
st.title("AI/GenAI Use Case Proposal Generator")

# Inputs for industry and company name
industry = st.text_input("Enter Industry:")

company_name = st.text_input("Enter Company Name:")

if st.button("Generate Proposal"):
    # Run the crew with specific inputs
    with st.spinner("Generating Proposal..."):
        result = research_crew.kickoff(inputs={"industry": industry, "company_name": company_name})
        markdown_text = result.raw  # Extract the raw text content

    # Display the final proposal in markdown format if available
    if markdown_text:
        st.markdown(markdown_text)
    else:
        st.write("No proposal generated. Please try again.")

