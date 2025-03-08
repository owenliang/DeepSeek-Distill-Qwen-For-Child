from openai import OpenAI
from modelscope.msdatasets import MsDataset
import threading
import time 
import json 
import os 

API_KEY=os.getenv("API_KEY")
BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME='deepseek-r1'

PROMPT='''
# 角色
我是爸爸，会耐心解答女儿提出的问题。

# 注意事项
- 女儿目前读小学一年级，解答时请考虑她的理解水平。
- 如果女儿提出的问题太难，你可以适当简化问题，让她能够理解。
- 请不要使用专业术语和概念，尽量用通俗易懂的语言解答。

# 爸爸的风格
- 喜欢循序渐进的讲解，逐步引导女儿理解问题
- 喜欢用生活中的例子来解释抽象的概念
- 理性思维，喜欢用逻辑推理的方式解答问题
- 经常会叫女儿的名字“赛西”，以便保证她的注意力
- 反复确认女儿有没有听懂，通过提问和重复解答的方式，确保她理解了问题
- 偶尔抛出反问或有趣的问题，引发女儿思考

# 来自女儿的提问
{question}
'''

THREAD=30
SAMPLES=1000

class R1Generator:
    def __init__(self,threads,dataset,samples):
        self.client=OpenAI(api_key=API_KEY,base_url=BASE_URL)
        self.idx=0
        self.threads=threads
        self.dataset=dataset
        self.samples=samples
        self.mutex=threading.Lock()

    def generate(self,question):
        completion=self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {'role': 'user', 'content': PROMPT.format(question=question)},
            ]
        )
        return completion.choices[0].message.reasoning_content,completion.choices[0].message.content

    def begin(self):
        self.idx=0
        self.progress=0
        self.result=[None]*self.samples
        self.thread_handlers=[]
        for i in range(self.threads):
            t=threading.Thread(target=self.thread_main)
            t.start()
            self.thread_handlers.append(t)

    def join(self):
        while True:
            with self.mutex:
                print(f'Progress: {self.progress}/{self.samples}',end='\r')
                if self.progress>=self.samples:
                    break
            time.sleep(1)
        for t in self.thread_handlers:
            t.join()
        return [res for res in self.result if res is not None]
    
    def thread_main(self):
        while True:
            with self.mutex:
                if self.idx>=self.samples:
                    break
                cur_idx=self.idx
                self.idx+=1
            try:
                question=self.dataset[cur_idx]['question']
                reasoning,answer=self.generate(question)
                self.result[cur_idx]=(question,reasoning,answer)
            except:
                pass
            with self.mutex:
                self.progress+=1

if __name__=='__main__':
    gsm8k=MsDataset.load('modelscope/gsm8k',subset_name='main',split='train')
    r1=R1Generator(threads=THREAD,dataset=gsm8k,samples=SAMPLES)
    r1.begin()
    result=r1.join()
    
    with open('r1_distill.txt','w') as f:
        for res in result:
            question,reasoning,answer=res
            f.write(json.dumps({'question':question,'reasoning':reasoning,'answer':answer})+'\n')