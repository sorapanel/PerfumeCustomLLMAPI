import json
import os
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline, 
    TrainingArguments
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
import torch

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class CustomProcessingView(APIView):
    http_method_names = ['get', 'head', 'options']

    def get(self, request, format=None):
        instruction = request.query_params.get('instruction')

        response = ''

        try:
            response = generateInfo(instruction)
        except (TypeError, ValueError):
            return Response({"error": "Invalid parameters"}, status=status.HTTP_400_BAD_REQUEST)

        message = '生成結果の取得に失敗しました。再度やり直してください。'

        if response != '' or response is not None:
            message = '生成成功。'

        return Response({"result": response, "message": message})

def generateInfo(instruction):
    model_name = "cyberagent/open-calm-3b"

    local_peft_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fine_tuned_open_calm")

    print(f"Attempting to load PEFT model from: {local_peft_model_path}")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.backends.mps.is_available() else torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, local_peft_model_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    input_instruction = '### 質問:\n' + instruction + '\n\n### 回答:\n'

    result = pipe(
        input_instruction,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = result[0]['generated_text']
    response_only = generated_text[len(input_instruction):].split(tokenizer.eos_token)[0].strip()

    return response_only

