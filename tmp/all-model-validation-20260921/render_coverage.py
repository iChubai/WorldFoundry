from pathlib import Path
import json,collections
r=Path(__file__).resolve().parent;rows=json.loads((r/'inventory.json').read_text())+json.loads((r/'additional-variants.json').read_text());obs=json.loads((r/'observations.json').read_text());by=collections.defaultdict(list)
for x in obs:by[x['model_id']].append(x)
interfaces=json.loads((r/'interfaces.json').read_text()) if (r/'interfaces.json').exists() else [];checks={x['target']:x for x in interfaces}
for x in rows:
 x['observations']=by[x['model_id']];x['verification_status']=x['observations'][-1]['status'] if x['observations'] else 'not_run'
 x['interface_checks']=[checks[t] for t in x.get('pipeline_targets',[]) if t in checks]
(r/'coverage.json').write_text(json.dumps(rows,indent=2));counts=dict(collections.Counter(x['verification_status'] for x in rows))
lines=['# 全模型验证台账','','范围合并模型目录、公共绑定、运行配置与额外变体。基础 checkpoint 能力另列于 base-model-checks.json。历史运行仅作为线索，不自动计为当前通过。','','当前状态：'+json.dumps(counts,ensure_ascii=False),'','verified_configuration 仅代表记录配置的真实推理与适用输出检查通过；verified_inference_only 仅通过推理/输出结构，未完成任务语义验证；blocked_external 为已实际核查的外部阻塞；failed_integration 为已实际复现、尚未完成修复复测的项目集成失败；failed_resource_conflict 为共享 GPU 资源冲突导致的运行失败，需换卡复测。接口检查不能替代推理。另见 runtime-variant-obligations.json 中未被目录显式列出的运行变体。','','|模型/变体|分类|当前验证|接口检查|证据|','|---|---|---|---|---|']
for x in rows:
 evidence='; '.join(o['status_evidence'] for o in x['observations']);ic=', '.join(z['status'] for z in x['interface_checks']) or 'pending'
 lines.append('|'+x['model_id']+'|'+x.get('category','uncataloged')+'|'+x['verification_status']+'|'+ic+'|'+evidence+'|')
(r/'COVERAGE.md').write_text('\n'.join(lines)+'\n');print('Current scope',len(rows),'states',counts,'interfaces checked',len(interfaces))
