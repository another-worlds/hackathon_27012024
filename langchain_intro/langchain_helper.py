from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

def generate_pet_name(animal_type, pet_color):
    llm = OpenAI(temperature=0.7)

    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'],
        template="I have a {animal_type} pet. Suggest me 5 cool names for it. It is {pet_color} in color."
    )
    
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)
    response  = name_chain.invoke({'animal_type' : animal_type, 'pet_color': pet_color})
    return response


if __name__ == '__main__':
    print(generate_pet_name("Cow"))