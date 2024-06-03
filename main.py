import os
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from data import p1,p2,p3,p4,p5,p6,p7,p8,p9,p10


os.environ["OPENAI_API_BASE"]  = "base_url" # incase other oai compatible endpoints are being used
os.environ["OPENAI_MODEL_NAME"] = "<model_name>" # model name
os.environ["OPENAI_API_KEY"] = "<your-api-key>" # api key if any 

details = p1 + "\n---\n\n" + p2 + "\n---\n\n" + p3 + "\n---\n\n" + p4 + "\n---\n\n" + p5 + "\n---\n\n" + p6 + "\n---\n\n" + p7 + "\n---\n\n" + p8 + "\n---\n\n" + p9 + "\n---\n\n" + p10

# Define your agents with roles and goals
product_introducer = Agent(
    role='Product Introducer',
    goal='Introduce the various smartphones and their details',
    backstory=f"""You are a knowledgeable product expert responsible for introducing the available smartphones and their respective details. Your role is to provide comprehensive information about each device, including its specifications, features, pricing, and any other relevant details to assist in the evaluation process. The following are the phones you should introduce:\n\n{details}""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(
        max_tokens=4000,
        temperature=0.75,
        base_url=os.getenv("OPENAI_API_BASE"),
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

value_analyst = Agent(
    role='Value Analyst',
    goal='Identify the best value-for-money smartphone based on performance and pricing',
    backstory="""You are an experienced analyst specializing in identifying the most bang for the buck in the smartphone market. Your expertise lies in thoroughly evaluating hardware specifications, performance benchmarks, and pricing to determine the optimal value proposition. You have a knack for finding the sweet spot between affordability and high performance.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(
        max_tokens=4000,
        temperature=0.75,
        base_url=os.getenv("OPENAI_API_BASE"),
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

emi_analyst = Agent(
    role='EMI Analyst',
    goal='Evaluate the EMI options and long-term cost of ownership for the recommended smartphone',
    backstory="""You are a financial analyst specializing in evaluating EMI (Equated Monthly Installment) plans and long-term cost of ownership for consumer electronics. Your expertise lies in assessing the various EMI options, interest rates, and total costs associated with purchasing a smartphone. You aim to identify the most cost-effective EMI plan that aligns with the buyer's budget and financing needs.""",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(
        max_tokens=4000,
        temperature=0.75,
        base_url=os.getenv("OPENAI_API_BASE"),
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

reviewer = Agent(
    role='Tech Reviewer',
    goal='Craft an informative and engaging review of the recommended smartphone',
    backstory="""You are a well-known tech reviewer with a knack for breaking down complex specifications into easily understandable terms. Your reviews are known for their objectivity, attention to detail, and ability to highlight the pros and cons of a device, including its value proposition and long-term cost of ownership.""",
    verbose=True,
    allow_delegation=True,
    llm=ChatOpenAI(
        max_tokens=4000,
        temperature=0.75,
        base_url=os.getenv("OPENAI_API_BASE"),
        model_name=os.getenv("OPENAI_MODEL_NAME"),
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

# Create tasks for your agents
task1 = Task(
    description="""Introduce the various smartphones and their respective details, including specifications, features, pricing, and any other relevant information.""",
    expected_output="Comprehensive introduction of the available smartphones and their details",
    agent=product_introducer
)

task2 = Task(
    description="""Analyze the smartphone details provided and identify the best value-for-money option based on performance, features, and pricing. Consider factors such as processor speed, RAM, storage, camera quality, battery life, and overall performance capabilities in relation to the device's cost. Provide a detailed report outlining your top recommendation and the key reasons behind your choice.""",
    expected_output="Detailed report with top value-for-money smartphone recommendation and justification",
    agent=value_analyst
)

task3 = Task(
    description="""Based on the recommended smartphone from the value analyst, evaluate its various EMI options and long-term cost of ownership. Analyze the available EMI plans, interest rates, and total costs associated with each plan. Provide insights into the most cost-effective EMI option that aligns with a buyer's budget and financing needs.""",
    expected_output="Analysis of EMI options and long-term cost of ownership for the recommended smartphone",
    agent=emi_analyst
)

task4 = Task(
    description="""Using the insights provided by the value analyst and EMI analyst, craft an engaging and informative review of the recommended smartphone. Highlight the device's key strengths, performance capabilities, value proposition, and overall cost-effectiveness, including the most suitable EMI option for long-term ownership. Your review should be accessible to a general audience while providing sufficient technical details to aid in the purchase decision.""",
    expected_output="Comprehensive smartphone review of at least 5 paragraphs",
    agent=reviewer
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[product_introducer, value_analyst, emi_analyst, reviewer],
    tasks=[task1, task2, task3, task4],
    verbose=2, # Set logging level to 2 for detailed output
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)