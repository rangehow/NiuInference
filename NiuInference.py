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
# from accelerate import find_executable_batch_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mp.set_start_method('spawn', force=True)

class DefaultDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data=tokenizer(data,return_tensors='pt',padding=True)

    
    def __getitem__(self,idx):
        return {'input_ids':self.data['input_ids'][idx]}
    
    def __len__(self):
        return self.data['input_ids'].size(0)


def should_reduce_batch_size(exception: Exception) -> bool:
    """
    Checks if `exception` relates to CUDA out-of-memory, CUDNN not supported, or CPU out-of-memory

    Args:
        exception (`Exception`):
            An exception
    """
    _statements = [
        "CUDA out of memory.",  # CUDA OOM
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED.",  # CUDNN SNAFU
        "DefaultCPUAllocator: can't allocate memory",  # CPU OOM
    ]
    if isinstance(exception, RuntimeError) and len(exception.args) == 1:
        return any(err in exception.args[0] for err in _statements)
    return False


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
        self.generation_config=generation_config
        
        
    def _load_model_and_tokenizer(self,device):
        print(self.dtype)
        config=AutoConfig.from_pretrained(self.model_dir)
        if config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            model=AutoModelForCausalLM.from_pretrained(self.model_dir,torch_dtype=self.dtype)
        else:
            model=AutoModelForSeq2SeqLM.from_pretrained(self.model_dir,torch_dtype=self.dtype)
        model.to(device)
        tokenizer=AutoTokenizer.from_pretrained(self.model_dir)
        return model,tokenizer

    # @find_executable_batch_size(starting_batch_size=1)
    # def auto_get_pred(batch_size):
        

    def get_pred(self,rank,out_path,data,dict):
        batch_size=16
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
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
                dataloader=DataLoader(dataset,batch_size,collate_fn=collator,pin_memory=True,num_workers=4)
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
                    result+=pred
                dict[f'{rank}']=result
            except Exception as e:
                print(e)
                if should_reduce_batch_size(e):
                    gc.collect()
                    if is_xpu_available():
                        torch.xpu.empty_cache()
                    elif is_mlu_available():
                        torch.mlu.empty_cache()
                    elif is_npu_available():
                        torch.npu.empty_cache()
                    else:
                        torch.cuda.empty_cache()
                    batch_size //= 2
                else:
                    raise
        
        
        # except Exception as e:
        #     logger.error(f'rank {rank}的推理过程解析出错了')
        #     logger.error(e)
            
            
    
    def split_list(self,lst, n):
        avg = len(lst) / float(n)
        return [lst[int(avg * i):int(avg * (i + 1))] for i in range(n)]

    def run(self,):
    
        world_size = torch.cuda.device_count()
        data_subsets = self.split_list(self.data,world_size)
        print(data_subsets)
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

    
    def _get_batch_size(
        self,
        max_batch_size: int = None,
        num_iterations: int = 5,
    ) -> int:
        world_size = torch.cuda.device_count()
        model,tokenizer=self._load_model_and_tokenizer(torch.device('cuda:0'))
        if self.dataset is not None:
            dataset=self.dataset(data=self.data,tokenizer=tokenizer)
        else:
            dataset=DefaultDataset(data=self.data,tokenizer=tokenizer)
        if self.generation_config is not None:
            if generation_config.max_new_tokens is not None:
                
                length=dataset[0]['input_ids'].size(1)+generation_config.max_new_tokens
            elif generation_config.max_length is not None:
                length=generation_config.max_length
        else:
            try:
                model_max_len=model.config.max_position_embeddings  
            except:
                model_max_len=tokenizer.config.model_max_length
            length=min(dataset[0]['input_ids'].size()[0]*3,model_max_len)
        
   
        batch_size = 2
        while True:
            if max_batch_size is not None and batch_size >= max_batch_size:
                batch_size = max_batch_size
                break
            if batch_size >= len(dataset):
                batch_size = batch_size // 2
                break
            try:
                for _ in range(num_iterations):
                    # dummy inputs and targets
                    inputs = torch.randint(low=4,high=6,size=(batch_size,length), device=torch.device('cuda:0'))
  
                    model(input_ids=inputs)
                batch_size *= 2
            except RuntimeError:
                batch_size //= 2
                break
        del model,tokenizer
        torch.cuda.empty_cache()
        import pdb
        pdb.set_trace()
        return batch_size
    

if __name__=='__main__':
    i=NiuInference(model_dir='/data/ruanjh/best_training_method/T5_V3_4gram',data=['你好','我爱你','我爱你','我爱你','我爱你','我爱你','我爱你','啊啊啊啊啊啊啊啊啊啊啊啊啊啊'])
    i.run()
