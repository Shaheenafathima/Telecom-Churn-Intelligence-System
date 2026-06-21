import google.generativeai as genai

genai.configure(api_key="AIzaSyCBv-N1ximef9rrwJkSc9WtIyzbptcsZ90")

for model in genai.list_models():
    print(model.name)