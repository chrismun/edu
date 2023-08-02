import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from util import extract_key_points, generate_question, evaluate_answer

# LLM 
model_path = 'psmathur/orca_mini_v2_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)

model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

# Data 
transcript = '/psyc350/transcript_psyc350-010-20181012-090501.mp4'

key_points = extract_key_points(model, tokenizer, transcript, 5)

for point in key_points:
    question = generate_question(model, tokenizer, point)
    print(question)
    user_answer = input("Your answer: ")
    feedback = evaluate_answer(model, tokenizer, user_answer, question, point)
    print(feedback)