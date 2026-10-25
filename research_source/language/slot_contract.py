"""Source-locked language realization for an executed SGDB explanation trace."""
import re,json,hashlib,math
ZH={'nut':'螺母','washer':'垫圈','screw':'螺丝','bolt':'螺栓','green_tape':'绿胶带','red_tape':'红胶带','black_pencil':'黑铅笔','red_pencil':'红铅笔','blue_eraser':'蓝橡皮','red_eraser':'红橡皮','yellow_cookie':'黄饼干','black_cookie':'黑饼干','square_dish':'方盘','round_dish':'圆盘','fork':'叉子','plate':'餐盘','spoon':'勺子','short_screw_length':'短螺丝长度','long_screw_length':'长螺丝长度','green_red_length_ratio':'绿/红胶带长度比','blue_red_length_ratio':'蓝/红木条长度比','pencil_length_ratio':'黑/红铅笔长度比','eraser_length_ratio':'蓝/红橡皮长度比','rope_reference_length_ratio':'绳子/参考木条长度比','utensil_order':'餐具顺序','measurement_unavailable':'测量可用性'}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def label(key,lang):
 if lang=='en':return key.replace('_',' ')
 return ZH.get(key[:-6],key[:-6])+'数量' if key.endswith('_count') else ZH.get(key,key.replace('_',' '))
def fmt(x):
 if isinstance(x,bool):return str(x).lower()
 if isinstance(x,(float,int)):return str(int(x)) if float(x).is_integer() else f'{x:.5g}'
 if isinstance(x,list):return '['+', '.join(fmt(v) for v in x)+']'
 return str(x)
def contract(report,lang='en'):
 by={r['key']:r for r in report['rules']};selected=[by[k] for k in report['language']['selected_rule_keys']]
 if report['unknown'] and not any(r['key']=='measurement_unavailable' for r in selected):selected.append(by['measurement_unavailable'])
 slots={};nodes=[];required=[]
 for r in selected:
  eid=r['evidence_id'];v=r['observed'];q=r['requirement'];op=q['operator'];unit=' px' if r['unit']=='px' else ''
  vv=(', '.join(label(s,lang) for s in v) if op=='empty' else fmt(v))+unit
  relation='compare with requirement'
  if op=='interval':
   req=f'[{fmt(q["lo"])}, {fmt(q["hi"])}]{unit}';relation='above upper bound' if v>q['hi'] else 'below lower bound' if v<q['lo'] else 'within interval'
  elif op in ['equal','at_least','every_equal']:
   req=fmt(q['value']);relation='satisfies requirement' if r['raw_residual']<=1e-8 else ('below required count' if isinstance(v,(int,float)) and v<q['value'] else 'above required count' if isinstance(v,(int,float)) and v>q['value'] else 'differs from requirement')
  elif op=='empty':req='all required operands available' if lang=='en' else '必需操作数均可用';relation='unavailable' if v else 'available'
  else:req=json.dumps(q,ensure_ascii=False);relation='satisfies requirement' if r['raw_residual']<=1e-8 else 'deviates from requirement'
  for suffix,val in [('value',vv),('requirement',req),('cite','['+eid+']')]:slots[f'<{eid}.{suffix}>']=val
  required += [f'<{eid}.value>',f'<{eid}.requirement>',f'<{eid}.cite>']
  nodes.append({'name':label(r['key'],lang),'relation':relation,'observed_slot':f'<{eid}.value>','requirement_slot':f'<{eid}.requirement>','citation_slot':f'<{eid}.cite>','operator':op})
 zero=not any(r['raw_residual']>1e-8 for r in report['rules'])
 state=('All executed predicates have zero residual.' if lang=='en' else '已执行谓词的残差均为零。') if zero else (('The selected score contributions support the recorded alarm.' if lang=='en' else '所选分数贡献足以支持已记录的报警。') if report['predicted_anomaly'] else ('The full score does not exceed the alarm threshold; observed deviations are retained.' if lang=='en' else '总分未超过报警阈值，已观测到的偏离仍予以保留。'))
 slots['<T.state>']=state;required.append('<T.state>')
 public={'decision':'anomaly' if report['predicted_anomaly'] else 'no_alarm','language':'English' if lang=='en' else '简体中文','facts':nodes,'required_slots':required,'execution_state':('all executed residuals zero' if zero else 'selected contributions sufficient for alarm' if report['predicted_anomaly'] else 'below alarm threshold; retain deviations'),'execution_state_slot':'<T.state>','scope':'Describe executed decoded facts, not an independent visual assessment or physical cause.'}
 system='''You write concise, natural inspection explanations from a verified SGDB execution trace. Return only JSON with exactly two keys: "decision" (copy the provided enum) and "explanation" (one coherent paragraph). Write in the requested language. The paragraph must include every required slot exactly once. Use the slot tokens verbatim. Their values are intentionally withheld; the program fills them after validation. Never guess or expand any slot. A citation slot already includes brackets. Combine related findings into natural sentences rather than listing field names. Integrate <T.state> as a final standalone sentence. Do not write literal digits or evidence IDs outside slots. Every value and requirement must appear only through its supplied slot. Do not add comparison numbers in words either. Preserve each supplied relation and unavailable status. You may change wording and combine related clauses, but may not add observations, causes, intent, confidence, visual verification, recommended actions or claims that all checks pass unless the provided execution state says so. Requirements are a comparison, not an observed value. A no_alarm result does not by itself mean that every rule is satisfied. No markdown or extra fields.'''
 return {'public':public,'slots':slots,'required':required,'decision':public['decision'],'language':lang,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(public,ensure_ascii=False)}],'source_trace_sha256':digest(report),'source_id':report['source_id'],'score':report['score'],'threshold':report['threshold']}
def validate(raw,c):
 errors=[]
 try:
  t=raw.strip();t=re.sub(r'^```(?:json)?\s*|\s*```$','',t);x=json.loads(t)
 except Exception:return {'valid':False,'errors':['invalid_json']}
 if set(x)!={'decision','explanation'}:errors.append('schema')
 if x.get('decision')!=c['decision']:errors.append('decision_changed')
 text=x.get('explanation','')
 if not isinstance(text,str):return {'valid':False,'errors':errors+['explanation_not_string']}
 used=re.findall(r'<[^<>]+>',text)
 for slot in c['required']:
  if used.count(slot)!=1:errors.append('slot_coverage:'+slot)
 for slot in used:
  if slot not in c['slots']:errors.append('unknown_slot:'+slot)
 outside=re.sub(r'<[^<>]+>','',text)
 if re.search(r'\d',outside):errors.append('unbound_number')
 if re.search(r'\[\s*E|\bR\d|\bO\d',outside):errors.append('unbound_citation')
 if re.search(r'制造|装配错误|人为|故意|manufactur|assembly defect|deliberately|physically verified',outside,re.I):errors.append('unsupported_cause')
 # Direction checks are local to the sentence containing each operand. Broader semantic
 # faithfulness is reviewed separately and is not equated with these mechanical checks.
 for node in c['public']['facts']:
  pos=text.find(node['observed_slot']);left=max(text.rfind('.',0,pos),text.rfind('。',0,pos),text.rfind(';',0,pos),text.rfind('；',0,pos));end_candidates=[n for n in [text.find('.',pos+len(node['observed_slot'])),text.find('。',pos),text.find(';',pos),text.find('；',pos)] if n>=0];seg=text[left+1:min(end_candidates) if end_candidates else len(text)]
  rel=node['relation']
  if rel=='above upper bound' and re.search(r'低于|below|within the (?:normal )?(?:range|interval)',seg,re.I):errors.append('direction_conflict')
  if rel=='below lower bound' and re.search(r'超过上界|above|exceeds? (?:the )?(?:upper|maximum)',seg,re.I):errors.append('direction_conflict')
  if rel=='unavailable' and not re.search(r'不可用|缺失|无法|未能|unavailable|missing|not available|cannot',text,re.I):errors.append('unavailable_omitted')
 if any(n['relation']=='unavailable' for n in c['public']['facts']) and re.search(r'all (?:checks|constraints|requirements) (?:are )?(?:satisfied|met|pass)|所有.*(?:均满足|均符合|全部通过)',outside,re.I):errors.append('unknown_treated_satisfied')
 rendered=text
 for slot,val in c['slots'].items():rendered=rendered.replace(slot,val)
 return {'valid':not errors,'errors':sorted(set(errors)),'decision':x.get('decision'),'slot_text':text,'rendered_text':rendered,'citations':[v for k,v in c['slots'].items() if k.endswith('.cite>')],'checks_scope':'JSON, decision, source-slot coverage, literal-number/citation exclusion, explicit missingness and selected lexical conflicts; not a proof of unrestricted natural-language semantics.'}
