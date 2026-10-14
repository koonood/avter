from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

response = client.responses.create(
    model="gpt-6-luna",
    input="In one short sentence, explain what a relationship timeline is."
)

print(response.output_text)
