import json,time,traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
root=Path('../ckpts/yxzhang2024--StarWM').resolve()
out=Path('tmp/starwm-gpu-20260924')
prompt=('Current StarCraft II observation at t=0:\n'
'Info: Minerals 150; Gas 0; Supply 12/15; no alerts.\n'
'Queue: Barracks construction 80 percent complete.\n'
'My Units: SCV #1 at (20,20), HP 100 percent, idle; Marine #2 at (24,20), HP 100 percent.\n'
'My Structures: Command Center at (16,16), HP 100 percent.\n'
'Visible Hostiles: Enemy Zergling #3 at (30,20), HP 100 percent, approaching Marine #2.\n'
'Actions over the next 5 seconds: SCV #1 continues Barracks construction; Marine #2 attacks Zergling #3.\n'
'Predict the observation after 5 seconds in the same five modules.\n\n/no_think')
result={'repo_id':'yxzhang2024/StarWM','checkpoint':str(root),'prompt':prompt,'device':'cuda:0','status':'running'}
(out/'result.json').write_text(json.dumps(result,indent=2))
try:
 t=time.time();tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
 print('tokenizer_loaded',time.time()-t,flush=True)
 t=time.time();model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,torch_dtype=torch.bfloat16,device_map='cuda:0',low_cpu_mem_usage=True).eval()
 print('model_loaded',time.time()-t,'memory',torch.cuda.memory_allocated(),flush=True)
 messages=[{'role':'user','content':prompt}]
 text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 inp=tok(text,return_tensors='pt').to('cuda:0')
 t=time.time()
 with torch.inference_mode(): output=model.generate(**inp,max_new_tokens=192,do_sample=False,pad_token_id=tok.eos_token_id)
 response=tok.decode(output[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
 result.update({'status':'generated','response':response,'input_tokens':int(inp['input_ids'].shape[1]),'output_tokens':int(output.shape[1]-inp['input_ids'].shape[1]),'generation_seconds':time.time()-t,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
 print('generated',result['output_tokens'],'tokens',flush=True)
except Exception as exc:
 result.update({'status':'error','error':repr(exc),'traceback':traceback.format_exc()})
 print('error',repr(exc),flush=True)
finally:
 (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
