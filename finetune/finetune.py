# handle dependencies
from dataclasses import dataclass
import vertexai
from vertexai.tuning import sft

#dataclass
@dataclass
class model_params:




#sentence generation

def system_convo(sample):
    return {"systemInstruction": {
                "role": "system",
                "parts": [
                { "text": "Imagine you are Vaishnavi, the recipient of this text message. What would you say in response to this?" },
                { "text": "Use very similar tone, emoji usage, etc to this response." }
                ]
            },
            "contents": [
                {
                "role": "user",
                "parts": [
                    { "text": sample['insert'] }
                ]
                },
                {
                "role": "model",
                "parts": [
                    { "text": sample["response"] }
                ]
                }
            ]
    }
#model loading

