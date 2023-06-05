# LongChian Integration with LLaMA-Adapter

[LangChain](https://python.langchain.com/en/latest/index.html) is a library that facilitates the development of applications by leveraging large language models (LLMs) and enabling their composition with other sources of computation or knowledge. With the integration of LangChain, LLaMA-Adapter can be used to compose LLMs with other sources of computation or knowledge. And could be used to build a simple ChatBot with lines of codes.

To use LangChain, you need to first install it with:
```shell
pip install langchain
```

## Singel-Modal LLAMA-Adapter V1

To integrate LangChain with LLaMA-Adapter, we need to create a new class that inherits from the `LLM` class. The new class should implement the `_call` method. 

```python
class LLaMALangChain(LLM):

    generator: LLaMA
    temperature: float = 0.1
    top_p: float = 0.75
    max_seq_len: int = 512
    max_gen_len: int = 64

    @property
    def _llm_type(self) -> str:
        return "LLaMA Adapter"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        full_result = self.generator.generate([prompt], max_gen_len=self.max_gen_len, temperature=self.temperature, top_p=self.top_p)[0]
        pure_response = full_result[full_result.rfind("Response:")+9:]
        # to avoid it automatically adding a self-proposed question
        if "### Input:" in pure_response:
            print(f"### Input: found in response {pure_response}, removing it")
            pure_response = pure_response[:pure_response.find("### Input:")]
            print(f"New response: {pure_response}")
        return pure_response


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_seq_len": self.max_seq_len,
        }
```

Then we can use the new class to build a simple ChatBot along with the `ConversationChain` to add memory funcationality to the LLaMA Adapter.

```python
from langchain.prompts.prompt import PromptTemplate

template = """The following is a conversation between a human and an AI. Write a response that completes the request.

Current conversation:
{history}
### Input: {input}
### Response:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=template
)

conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm, 
        verbose=True, 
        memory=ConversationBufferMemory(ai_prefix="### Response", human_prefix="### Input")
)
```

Finally, enjoy the chatBot with `conversation.predict(input=xxx)`.


## Multi-Modal LLaMA-Adapter V2

Since `LangChain` is orignally designed for text generation while LLaMA-Adapter V2 needs the visual modality input for visual reasoning, we need to add the image path information inside the prompt and let the model conduct visual processing during the call of LLM. You can choose other position indicator as you like.

```python
prompt = img_path + "<ImgPathEnd>" + prompt
```

Besides, we add the history management module similar to LangChain in V2, since the original LangChain got crashed when the input with some special characters such as the position indicator.

 We provide a full demo for the integration of LLaMA-Adapter V2 and LangChain in [HERE](../docs/langchain_LLaMA_AdapterV2_demo.ipynb). You can just have a try!