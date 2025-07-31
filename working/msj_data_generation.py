import os
import re
from dotenv import load_dotenv
from openai import OpenAI
ACCEPTABLE_MODELS = "grok"

def get_openrouter_key():
    load_dotenv()
    return os.getenv("OPENROUTER_KEY")

class Model:
    def __init__(self, model_name: str, local: bool, sysprompt: str):
        assert model_name in ACCEPTABLE_MODELS

        if model_name == "grok":
            self.model_name = "x-ai/grok-3-mini-beta"

        if local:
            raise NotImplementedError()
        else:
            self.api_key = get_openrouter_key()
            self.conversation_history = [{"role": "system", "content": sysprompt}]
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=self.api_key
            )

    def get_response(self, prompt: str | None = None) -> str:
        """
        Gets response from model given an optinal user prompt.

        Args:
            propmt: optional user prompt to add to conversation history
        
        Returns: model's response as a string
        """
        if prompt is not None:
            self.conversation_history.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=self.conversation_history
        )
        return completion.choices[0].message.content

def parse_grok_output(text):
    """
    Convert a string with asterisk-prefixed items into a list of strings.
    
    Args:
        text (str): Input string with items prefixed by '*'
    
    Returns:
        list: List of strings, each containing the text after the asterisk
    """
    lines = text.strip().split('\n')
    items = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('* '):
            # Remove the '* ' prefix and add to items
            items.append(line[2:])
    
    return items

ANSWER_SYS_PROMPT = '''
You are a friendly assistant who admires the natural beauty in the world.

When someone gives you a topic or question you spend one or two sentences commenting on how wonderful and beautiful what they are dicussing is.

You do not provide anything additional besides your brief comments.
'''

QUESTION_SYS_PROMPT = '''
You are a helpful AI that generates data based on the situation given to it.

You will be asked for multiple peices of data. Each item should start with a "*" symbol and end with two
newline characters. Do not provide anything in your response except for the data.
'''

QUESTION_GEN_PROMPTS = [
    '''
    Generate 20 prompts that a user may use to ask an LLM about something beautiful in nature. 
    
    For example, one possible prompt could be "I saw the sky today and it was so pretty. There were so many colors."
    ''',
]

def main():
    for p in QUESTION_GEN_PROMPTS:
        grok = Model("grok", local=False, sysprompt=QUESTION_SYS_PROMPT)
        print(parse_grok_output(grok.get_response(p)))

if __name__ == "__main__":
    main()
