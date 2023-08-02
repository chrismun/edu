import torch
#generate text function
def generate_text(tokenizer, model, system, instruction, input=None):
    
    if input:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    else:
        prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    
    tokens = tokenizer.encode(prompt)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to('cuda')

    instance = {'input_ids': tokens,'top_p': 1.0, 'temperature':0.7, 'generate_len': 1024, 'top_k': 50}

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens, 
            max_length=length+instance['generate_len'], 
            use_cache=True, 
            do_sample=True, 
            top_p=instance['top_p'],
            temperature=instance['temperature'],
            top_k=instance['top_k']
        )    
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    return f'[!] Response: {string}'


def extract_key_points(model, tokenizer, content, num_points=5):

    prompt = f"Summarize the following text into {num_points} key points:\n{content}"

    system = "You are an intelligent AI model. You will assist in creating educational materials for students."
    
    response = generate_text(tokenizer, model, system, prompt, input=None)
    
    key_points = response.split('\n')  # Assume the model returns key points separated by newlines
    return key_points

def generate_question(model, tokenizer, point):

    prompt = f"Generate a question about this fact: {point}"

    system = "You are an intelligent AI model. You will assist in creating educational materials for students."
    
    question = generate_text(tokenizer, model, system, prompt, input=None)
    return question

def evaluate_answer(model, tokenizer, user_answer, question, point):
   
    correct_answer = point  

    prompt = f"The user was asked this question: {question}\n" \
             f"The user's answer was: {user_answer}\n" \
             f"The correct answer is: {correct_answer}\n" \
             f"Please provide feedback on the user's answer."

    system = "You are an intelligent AI model. You will assist in creating educational materials for students."


    feedback = generate_text(tokenizer, model, system, prompt, input=None)
    return feedback
