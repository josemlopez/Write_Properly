# %% 
import openai
from typing import List, Tuple
from wp_secrets.secrets_ import OPENAI_API_KEY
import gradio as gr
from enum import Enum

# Set the API key
openai.api_key = OPENAI_API_KEY


# Enum for the different functions
class Functions(Enum):
    WRITE_PROPERLY = "Write Properly"
    WRITE_THE_SAME_GRAMMAR_FIXED = "Write the same with Grammar Fixed"
    ANSWER_THE_EMAIL = "Answer the Email"
    ANSWER_A_QUESTION = "Answer a question"
    REMOVE_PASSIVE_VOICE = "Remove Passive Voice"
    SUMMARIZE = "Summarize"

    @staticmethod
    def get_list_of_functions() -> List[str]:
        return [function.value for function in Functions]


class ProperlyWriten:
    def __init__(self,
                 max_tokens: int = 300,
                 temperature: float = 0.7,
                 top_p: float = 1,
                 frequency_penalty: float = 0,
                 presence_penalty: float = 0,
                 stop: str = "",
                 ):
        """ Initialize the ProperlyWriten class.
            Args:
                max_tokens (int): The maximum number of tokens to generate
                temperature (float): The temperature of the model
                top_p (float): The cumulative probability threshold for top-p sampling
                # top_k (int): The number of the highest probability vocabulary tokens to keep for top-k sampling
                frequency_penalty (float): The frequency penalty
                presence_penalty (float): The presence penalty
                stop (str): The stop sequence
        """

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.prompts: List[str] = []

    def query_gpt3_generator(self, prompt: str) -> List[str]:
        """
        Query GPT-3 using the OpenAI API with a prompt

        Args:
            prompt (str): The prompt 

        Returns:
            List[str]: properly writen prompts
        """

        # query GPT-3 using the OpenAI API with the prompt
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                stop=self.stop
            )
        except openai.error.InvalidRequestError as e:
            if e.error == 400:
                return []
            else:
                raise e
        choices = [choice["text"] for choice in response["choices"]]
        print(f"Prompt to send \n {prompt}. \n Choices: {choices}")
        return choices


# define the function that will be called when the user clicks the button
def write_properly(text: str, pw: ProperlyWriten) -> Tuple[str, str]:
    prompt = "Write this properly:  \n" + text
    return pw.query_gpt3_generator(prompt)[0].strip(), prompt


def write_the_same_grammar_fixed(text: str, pw: ProperlyWriten) -> Tuple[str, str]:
    prompt = "Write the same, fixing the grammar:  \n" + text
    return pw.query_gpt3_generator(prompt)[0].strip(), prompt


def remove_passive_voice(text: str, pw: ProperlyWriten) -> Tuple[str, str]:
    prompt = "Remove the passive voice in the next text: \n {" + text + "}"
    return pw.query_gpt3_generator(prompt)[0].strip(), prompt


def summarize(text: str, pw: ProperlyWriten) -> Tuple[str, str]:
    prompt = "Summarize the text:  \n" + text
    return pw.query_gpt3_generator(prompt)[0].strip(), prompt


def answer_the_email(text: str,
                     post_fix: str,
                     pw: ProperlyWriten,
                     previous_text: str = None,
                     contine_: bool = False) -> Tuple[str, str]:
    prompt = ("Answer the email:  \n" + text +
              ".\n Using these main ideas: \n" + post_fix +
              "\n Write it properly and as a human English native speaker would do it.")

    if contine_:
        prompt = prompt + "\n" + previous_text
        res = pw.query_gpt3_generator(prompt)[0].strip()
    else:
        res = pw.query_gpt3_generator(prompt)[0].strip()
    return res, prompt


def answer_the_question(text: str, post_fix: str, pw: ProperlyWriten) -> str:
    if post_fix == "":
        return pw.query_gpt3_generator("Answer the question:  \n" + text +
                                       "\n Answer factually and thinking step by step.")[0].strip()
    else:
        prompt = ("Answer the question:  \n" + text +
                  ".\n Using these main ideas: \n" + post_fix +
                  "\n Answer factually and thinking step by step.")
        return pw.query_gpt3_generator(prompt)[0].strip()


def selector(function: str,
             text: str,
             post_fix: str,
             pw: ProperlyWriten,
             previous_text: str = None,
             mood: str = "normal",
             verbosity: str = "normal",
             choice=None) -> str:
    if mood == "Factual":
        print("Factual Temperature: 0.01 selected")
        pw.temperature = 0.01
    elif mood == "Creative":
        print("Creative Temperature: 0.7 selected")
        pw.temperature = 0.7
    if verbosity == "Verbose":
        print("Verbose max_tokens: 400 selected")
        pw.max_tokens = 400
        post_fix = post_fix + "\n write at least 400 tokens"
    elif verbosity == "Normal":
        print("Normal max_tokens: 150 selected")
        pw.max_tokens = 150
    if function == Functions.WRITE_PROPERLY.value:
        res = write_properly(text,
                             pw)
    elif function == Functions.WRITE_THE_SAME_GRAMMAR_FIXED.value:
        res = write_the_same_grammar_fixed(text,
                                           pw)
    elif function == Functions.ANSWER_THE_EMAIL.value:
        if choice == "Continue":
            res = answer_the_email(text,
                                   post_fix,
                                   pw,
                                   previous_text,
                                   True)
        res = answer_the_email(text,
                               post_fix,
                               pw)
    elif function == Functions.ANSWER_A_QUESTION.value:
        res = answer_the_question(text,
                                  post_fix,
                                  pw)
    elif function == Functions.REMOVE_PASSIVE_VOICE.value:
        res = remove_passive_voice(text,
                                   pw)
    elif function == Functions.SUMMARIZE.value:
        res = summarize(text,
                        pw)
    else:
        raise ValueError(f"function {function} not recognized")

    return res[0]


examples = [
    ["Write Properly",
     "correct this sentence so it can be explained correctly and everyone can understand what is said"],
    ["Write the same with Grammar Fixed", "the grammar were wrong in this sentence"]]

"""
iface = gr.Interface(selector,
                     [gr.components.Dropdown(["Write Properly",
                                              "Write the same with Grammar Fixed",
                                              "Answer the Email"]),
                      gr.components.Textbox(lines=5,
                                            label="Text"),
                      gr.components.Textbox(lines=5,
                                            label="Include this points",
                                            placeholder="Points to include in "
                                                        "the email")],
                     gr.components.Textbox(),
                     examples=examples,
                     title="Write Properly",
                     description="Write a text properly using GPT-3"
                     )

# set a title
iface.launch(share=False,
             server_port=3737)

gr.close_all()
"""


# %%
class WriteProperlyUI:
    """
    Write Properly UI
    """

    def __init__(self, pw: ProperlyWriten, port: int = 3737):
        self.pw = pw
        self.port = port
        self.selector = gr.components.Dropdown(Functions.get_list_of_functions(),
                                               label="Function",
                                               render=False)

        self.mood = gr.components.Dropdown(["Factual", "Creative"],
                                           label="Mood",
                                           render=False)
        self.verbosity = gr.components.Dropdown(["Succinct", "Verbose"],
                                                label="Verbosity",
                                                render=False)

        self.text = gr.components.Textbox(lines=5,
                                          label="Text",
                                          render=False)

        self.post_fix = gr.components.Textbox(lines=5,
                                              label="Include this points",
                                              placeholder="Points to include in the email",
                                              render=False)

        self.output = gr.components.Textbox(label="Output",
                                            render=False)

        self.run_button = gr.components.Button("Run",
                                               render=False)

        self.wp_block = gr.Blocks()

    def run(self,
            function: str,
            text: str,
            post_fix: str,
            previous_text: str = None,
            mood: str = "normal",
            verbosity: str = "normal",
            choice=None) -> str:
        return selector(function,
                        text,
                        post_fix,
                        self.pw,
                        previous_text,
                        mood,
                        verbosity,
                        choice)

    def launch(self):
        """
        Launch the interface
        """
        with self.wp_block:
            with gr.Row():
                with gr.Column():
                    self.selector.render()
                    self.mood.render()
                    self.verbosity.render()
                    self.text.render()
                    self.post_fix.render()
                    self.run_button.render()
                    radio = gr.components.Radio(["New", "Continue"],
                                                label="Should continue?",
                                                render=True)
                    radio.value = "New"
                    self.run_button.click(fn=self.run,
                                          inputs=[self.selector, self.text, self.post_fix, radio,
                                                  self.mood, self.verbosity],
                                          outputs=self.output)

                with gr.Column():
                    self.output.render()
        self.wp_block.launch(server_port=self.port)

    def close(self):
        """
        Close the interface
        """
        self.wp_block.close()
        gr.close_all()


pw_obj = ProperlyWriten()
wpui = WriteProperlyUI(pw_obj)
wpui.launch()
