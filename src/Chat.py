from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
import os

os.environ["OPENAI_API_KEY"] = "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"
chat = ChatOpenAI(model="gpt-4o", temperature=0)   

class ChatBot:
    
    def __init__(self, temperature=0):
        self.chat = ChatOpenAI(model="gpt-4o", temperature=temperature)        

    def ranker_function(self, query, context):
        prompt = '''
        Проранжируй следующие отрывки по степени их сходства с заданным запросом. Учитывай как семантическое значение, так и контекстную значимость. Укажи рейтинги в числовом порядке (1 для наиболее похожих, затем 2, 3 и т. д.) без каких-либо объяснений. Если значимого сходства нет, укажи «Нет». Сделай вывод таким, как в примере:

        Пример:
        Вход:
        Запрос: Упражнение на подсчёт совместной энтропии

        Информация для ранжирования:

        1. Сегодня всем спасибо, если есть вопросы, задавайте сейчас, пишите в чат что угодно, спасибо большое, до свидания, пожалуйста.
        2. Континуум значений и идеально точно угадать мы не сможем, поэтому интерпретация энтропии для случайных величин с функцией плотности хуже, но она тоже есть, с функцией плотности хуже, но давайте попробуем её тоже вытащить.
        3. А со степенями двойки легко интерпретировать: в рамках отдельного эксперимента можно интерпретировать, как энтропия показывает, сколько процентов при идеальном сжатии займёт длинное сообщение.
        4. Совместная энтропия — одна вероятность, но случайные величины разные. В кросс-энтропии и дивергенции Кульба-Клайблера — разные видения мира, что объясняет их различия.

        Выход:

        1. 3
        2. 1
        3. 4

        Задача:
        Проранжируй следующую информацию на основе ее сходства с предоставленными запросом.

        Запрос: {question}

        Информация для ранжирования:
        {context}
        '''

        prompt_template = PromptTemplate(
                    input_variables=['question', 'context'],
                    template=prompt
                )
                
        human_message = HumanMessage(content=prompt_template.format(question=query, context=context))

        messages = [human_message]
        response = self.chat.invoke(messages)

        return response.content
    
    