import os


openai_base_url = "https://api.gptsapi.net/v1"
claude_base_url = "https://api.gptsapi.net"
from dotenv import load_dotenv

_ = load_dotenv(".env")

# def _set_env(var: str):
#     value = os.environ.get(var)
#     if not value :
#         os.environ[var] = ""
#     else:
#         os.environ[var] = value


# _set_env("ANTHROPIC_API_KEY")
# _set_env("TAVILY_API_KEY")

print("*********")
print("env is loaded")
print(os.environ)
print("*********")
# print(os.environ.get("ANTHROPIC_API_KEY"))
# print(os.environ.get("TAVILY_API_KEY"))

tutorial_questions = [
    "Hi there, what time is my flight?",
    "Am i allowed to update my flight to something sooner? I want to leave later today.",
    "Update my flight to sometime next week then",
    "The next available option is great",
    "what about lodging and transportation?",
    "Yeah i think i'd like an affordable hotel for my week-long stay (7 days). And I'll want to rent a car.",
    "OK could you place a reservation for your recommended hotel? It sounds nice.",
    "yes go ahead and book anything that's moderate expense and has availability.",
    "Now for a car, what are my options?",
    "Awesome let's just get the cheapest option. Go ahead and book for 7 days",
    "Cool so now what recommendations do you have on excursions?",
    "Are they available while I'm there?",
    "interesting - i like the museums, what options are there? ",
    "OK great pick one and book it for my second day there.",
]