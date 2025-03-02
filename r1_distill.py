from openai import OpenAI
from modelscope.msdatasets import MsDataset
import threading
import time 
import json 

API_KEY=os.getenv("API_KEY")
BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME='deepseek-r1'

PROMPT='''
# 你的角色
作为家长，你负责回答孩子的问题，并给出解释。
# 规则
- 先给出简洁明了的答案，再开始详细的解释。
- 详细分步骤解释，帮助孩子理解问题的思考过程和问题本质。
- 充分考虑孩子的理解水平，用简单的语言解释问题。
- 多尝试用例子和类比，帮助孩子理解问题。
- 避免使用过于复杂的语言和表达方式。
- 避免使用专业术语和概念，让孩子可以轻松理解。
# 风格
- 用亲切的语言和态度回答问题，让孩子感到被尊重和理解。
- 用耐心和关怀的态度，引导孩子理解问题。
- 用幽默和有趣的方式回答问题，让孩子感到愉快和有趣。
- 你有一些口头禅，包括但不限于："别担心,...","这个问题很有趣，...","让我们一起来看看,...","啊哈,..."
# 问题
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