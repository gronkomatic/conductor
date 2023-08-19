import argparse

from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="C:/Users/sam/OneDrive/Documents/ai-stuff/llm-models/ggml/wizardlm-1.0-uncensored-llama2-13b.ggmlv3.q4_K_S.bin")
parser.add_argument("-t", "--text", type=str, default="Hello world!")
args = parser.parse_args()

llm = Llama(model_path=args.model, embedding=True)

embedding = llm.create_embedding(args.text)
# embedding.data.save("embedding.pt")
print(llm.create_embedding(args.text))
