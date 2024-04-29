import torch
from torch import bfloat16
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import Dataset,DataLoader
import functools
from transformers import AutoTokenizer,DefaultDataCollator,GenerationConfig,PreTrainedModel,AutoModelForSeq2SeqLM,AutoModelForCausalLM,AutoConfig,DataCollatorWithPadding
import logging
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from tqdm import tqdm

from loguru import logger




class DefaultDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data=tokenizer(data,return_tensors='pt',padding=True)

    
    def __getitem__(self,idx):
        return {'input_ids':self.data['input_ids'][idx]}
    
    def __len__(self):
        return self.data['input_ids'].size(0)




class NiuInference:
    def __init__(self,model_dir,data,dtype=bfloat16,dataset=None,data_collator=None,output_path='niuinference.out',auto_batch_size=True,batch_size=1,generation_config=None):
        self.model_dir=model_dir
        self.dtype=dtype
        self.data=data
        self.dataset=dataset
        self.data_collator=data_collator
        self.output_path=output_path
        self.batch_size=batch_size
        self.auto_batch_size=auto_batch_size
        
        try:
            self.generation_config=AutoConfig.from_pretrained(f'{model_dir}/generation_config.json')
        except:
            self.generation_config=None
        
        
    def _load_model_and_tokenizer(self,device):
        config=AutoConfig.from_pretrained(self.model_dir)
        tokenizer=AutoTokenizer.from_pretrained(self.model_dir)
        if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            model=AutoModelForCausalLM.from_pretrained(self.model_dir,torch_dtype=self.dtype)
            tokenizer.padding_side='left'
        else:
            model=AutoModelForSeq2SeqLM.from_pretrained(self.model_dir,torch_dtype=self.dtype)
        model.to(device)
        
        return model,tokenizer


        
    
    def get_pred(self,rank,out_path,data,dict):

        batch_size=1
        
        try:
            device = torch.device(f'cuda:{rank}')
            model, tokenizer = self._load_model_and_tokenizer(device)
            if self.dataset is not None:
                dataset=self.dataset(data=data,tokenizer=tokenizer)
            else:
                dataset=DefaultDataset(data=data,tokenizer=tokenizer)

            if self.data_collator is not None:
                collator=self.data_collator(tokenizer,model=model,padding=True)
            else:
                collator= DataCollatorWithPadding(tokenizer)
            dataloader=DataLoader(dataset,batch_size,collate_fn=collator,pin_memory=True,num_workers=0)
            result=[]
            for input in tqdm(dataloader):
                input.to(device)

                output = model.generate(
                            input_ids=input['input_ids'],
                            attention_mask=input['attention_mask'],
                            num_beams=5,
                            do_sample=False,
                            temperature=1.0,
                            max_new_tokens=512,
                        )
                pred = tokenizer.batch_decode(output,skip_special_tokens=True)
                print(pred)
                result+=pred
            dict[f'{rank}']=result
        except Exception as e:
            logger.error(f'{device}推断出错了')
            raise
        
        
            
            
    
    def split_list(self,lst, n):
        avg = len(lst) / float(n)
        return [lst[int(avg * i):int(avg * (i + 1))] for i in range(n)]


    @logger.catch
    def run(self,):
    
        world_size = min(torch.cuda.device_count(),len(self.data)) # 数据量就几条就不用那么多卡，corner case
        
        data_subsets = self.split_list(self.data,world_size)
        processes = []
        manager = mp.Manager()
        record_dict = manager.dict()
        for rank in range(world_size):

            p = mp.Process(target=self.get_pred, args=(rank,self.output_path,data_subsets[rank],record_dict))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        with open(self.output_path, "w", encoding="utf-8") as f:
            for rank in range(world_size):
                for r in record_dict[f'{rank}']:
                    f.write(r.replace('\n','\\n')+'\n')



if __name__=='__main__':
    mp.set_start_method('spawn')
    i=NiuInference(model_dir='/data/ruanjh/best_training_method/gemma-2b',data=['111','1','1',])
    i.run()
