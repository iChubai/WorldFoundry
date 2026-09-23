import json,time,traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from worldfoundry.synthesis.visual_generation.world_model.starwm.offline_infer.run_inference_client import process_prompt,parse_output
root=Path('../ckpts/yxzhang2024--StarWM').resolve()
out=Path('tmp/starwm-gpu-20260924')
item=json.loads((out/'official-fixture.json').read_text())[0]
prompt=process_prompt(next(m['content'] for m in item['messages'] if m['role']=='user'),'nothink')
label=next(m['content'] for m in item['messages'] if m['role']=='assistant')
result={'repo_id':'yxzhang2024/StarWM','fixture_source':'https://github.com/yxzzhang/StarWM/blob/main/data/wm_test_horizon5_1traj.json','fixture_index':0,'prompt':prompt,'label':label,'status':'running'}
(out/'official-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
try:
 tok=AutoTokenizer.from_pretrained(root,local_files_only=True)
 t=time.time();model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',low_cpu_mem_usage=True).eval()
 print('loaded',time.time()-t,flush=True)
 text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
 inp=tok(text,return_tensors='pt').to('cuda:0');result['input_tokens']=int(inp['input_ids'].shape[1])
 t=time.time();torch.manual_seed(0)
 with torch.inference_mode(): output=model.generate(**inp,max_new_tokens=1000,do_sample=True,temperature=0.6,top_p=0.95,top_k=20,pad_token_id=tok.eos_token_id)
 raw=tok.decode(output[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
 think,pred=parse_output(raw)
 result.update({'status':'generated','raw_response':raw,'prediction':pred,'thought':think,'output_tokens':int(output.shape[1]-inp['input_ids'].shape[1]),'generation_seconds':time.time()-t,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
 print('generated',result['output_tokens'],raw[:500],flush=True)
except Exception as exc:
 result.update({'status':'error','error':repr(exc),'traceback':traceback.format_exc()});print('error',repr(exc),flush=True)
finally:
 (out/'official-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
