#pip install ollama chromadb langchain langchain-community fastembed

import ollama
 
messages = []
 
def send(chat):
  messages.append(
    {
      'role': 'user',
      'content': chat,
    }
  )
  stream = ollama.chat(model='tinyllama', 
    messages=messages,
    stream=True,
  )
 
  response = ""
  for chunk in stream:
    part = chunk['message']['content']
    print(part, end='', flush=True)
    response = response + part
 
  messages.append(
    {
      'role': 'assistant',
      'content': response,
    }
  )
 
  print("")
 
if __name__ in "__main__":
    while True:
        chat = input(">>> ")
    
        if chat == "/exit":
            break
        elif len(chat) > 0:
            send(chat)