import json,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from safetensors.torch import load_file
from transformers import ViTConfig,ViTModel
from worldfoundry.synthesis.visual_generation.world_model.le_wm.jepa import JEPA
from worldfoundry.synthesis.visual_generation.world_model.le_wm.module import ARPredictor,Embedder,MLP
root=Path('../ckpts/eren23--lewm-models/slim_96d_4e_4p').resolve()
out=Path('tmp/leworldmodel-gpu-20260924')
image_path=Path('worldfoundry/data/benchmarks/assets/wrbench/natural25/first_frames/bedroom_cat_bed_jump.png')
result={'model_id':'leworldmodel','checkpoint':str(root/'lejepa_weights.safetensors'),'input_image':str(image_path),'status':'running'}
(out/'result.json').write_text(json.dumps(result,indent=2))
try:
 cfg=json.loads((root/'config.json').read_text())
 encoder=ViTModel(ViTConfig(image_size=224,patch_size=14,num_channels=3,hidden_size=192,num_hidden_layers=4,num_attention_heads=3,intermediate_size=768,qkv_bias=True),add_pooling_layer=False)
 predictor=ARPredictor(num_frames=3,depth=4,heads=16,mlp_dim=2048,input_dim=96,hidden_dim=192,output_dim=192,dim_head=64)
 model=JEPA(encoder,predictor,Embedder(10,10,96,4),projector=MLP(192,1024,96,norm_fn=nn.BatchNorm1d),pred_proj=MLP(192,1024,96,norm_fn=nn.BatchNorm1d))
 state=load_file(str(root/'lejepa_weights.safetensors'))
 report=model.load_state_dict(state,strict=False)
 result['weight_load']={'missing':report.missing_keys,'unexpected':report.unexpected_keys,'tensor_count':len(state)}
 if report.missing_keys or report.unexpected_keys: raise RuntimeError('checkpoint/model key mismatch')
 model=model.to('cuda:0').eval()
 arr=np.asarray(Image.open(image_path).convert('RGB').resize((224,224),Image.Resampling.BICUBIC),dtype=np.float32)/255.0
 x=torch.from_numpy(arr).permute(2,0,1)
 x=(x-torch.tensor([0.485,0.456,0.406])[:,None,None])/torch.tensor([0.229,0.224,0.225])[:,None,None]
 x=x[None,None].to('cuda:0')
 action_a=torch.zeros(1,1,3,10,device='cuda:0')
 action_b=action_a.clone();action_b[:,:,:,0]=1.0
 t=time.time()
 with torch.inference_mode():
  a=model.rollout({'pixels':x[:,None].clone()},action_a,history_size=3)['predicted_emb']
  b=model.rollout({'pixels':x[:,None].clone()},action_b,history_size=3)['predicted_emb']
 result.update({'status':'generated','runtime_seconds':time.time()-t,'output_shape':list(a.shape),'all_finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'action_effect_l2':float(torch.linalg.vector_norm(a-b).item()),'output_std':float(a.std().item()),'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
 print('generated',result['output_shape'],'action_effect_l2',result['action_effect_l2'],flush=True)
except Exception as exc:
 result.update({'status':'error','error':repr(exc),'traceback':traceback.format_exc()});print('error',repr(exc),flush=True)
finally:
 (out/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
