import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class KoLLM():

    def __init__(self, model_name, bnbConfig):
        
        self.model =  AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnbConfig,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        self.instruction = '당신은 뛰어난 학생입니다, 주어지는 지문을 읽고 문제의 답을 고르세요.'


    def Inference(self, data):
        outputs = {'answer' : []}

        with torch.no_grad():
            for i in range(3):
                inputs = self.tokenizer(self.instruction+"\n지문 : "+data["paragraphs"][i]+"\n문제 : "+data["problems"][i]+"\n 답 : ", return_tensors="pt").to(self.model.device)

                output = self.model.generate(
                        **inputs,
                        max_new_tokens=128,
                        top_p=0.9,
                        top_k=50,
                        temperature=0.3,
                        do_sample=True,
                        eos_token_id=self.tokenizer.eos_token_id
                        )

                output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                outputs['answer'].append(output)
        
        return outputs

        
