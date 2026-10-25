"""LLM writes source-attributed claims; numeric evidence stays with the executor."""
import re,json,hashlib
from pathlib import Path
import importlib.util
p=Path(__file__).resolve().parent/'slot_contract.py';s=importlib.util.spec_from_file_location('slot_contract',p);old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
digest=old.digest

def contract(report,lang='en'):
 base=old.contract(report,lang);facts=[];evidence={}
 by={r['evidence_id']:r for r in report['rules']}
 for n in base['public']['facts']:
  eid=n['citation_slot'][1:].split('.')[0];r=by[eid]
  item={'evidence_id':eid,'name':n['name'],'relation':n['relation'],'operator':n['operator']}
  if n['operator']=='empty':item['missing_operands']=[old.label(k,lang) for k in r['observed']]
  facts.append(item)
  evidence[eid]={'name':n['name'],'observed':r['observed'],'requirement':r['requirement'],'observed_display':base['slots'][n['observed_slot']],'requirement_display':base['slots'][n['requirement_slot']],'source_paths':r['source_paths'],'object_indices':r['object_indices'],'residual':r['raw_residual'],'normalized_score':r['normalized_score']}
 public={'decision':base['decision'],'language':'English' if lang=='en' else '简体中文','facts':facts,'scope':'These are decoded facts and executed predicates, not independently verified physical ground truth.'}
 system='''Write concise natural-language inspection findings from verified executed predicates. Return ONLY JSON with exactly "decision" (copy the supplied enum) and "claims" (a nonempty list). Each claim has exactly "evidence_ids" (list of supplied IDs) and "text" (a natural sentence). Cover every supplied ID exactly once. You may combine facts with the same relation into one fluent claim, listing all their IDs. Write in the requested language. Do not include IDs, citations, literal digits or numerical quantities in the text: the executor supplies a separate evidence table and attaches the IDs. Describe qualitative rule outcomes, not raw data fields. Preserve below/above/within/satisfied/deviating and unavailable states exactly. Missing operands are the named unavailable measurements, not empty observed values. Explain only what the decoded facts indicate. Do not add physical causes, intent, hidden states, visual verification, confidence, repair advice, score claims or a global statement that the image is truly normal. Keep the total text concise. No markdown, additional fields or commentary.'''
 return {'public':public,'evidence':evidence,'decision':base['decision'],'language':lang,'messages':[{'role':'system','content':system},{'role':'user','content':json.dumps(public,ensure_ascii=False)}],'source_trace_sha256':digest(report),'source_id':report['source_id'],'score':report['score'],'threshold':report['threshold'],'execution_state_text':base['slots']['<T.state>']}

def validate(raw,c):
 errors=[]
 try:x=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip()))
 except Exception:return {'valid':False,'errors':['invalid_json']}
 if set(x)!={'decision','claims'}:errors.append('schema')
 if x.get('decision')!=c['decision']:errors.append('decision_changed')
 claims=x.get('claims');allowed={n['evidence_id']:n for n in c['public']['facts']};seen=[];rendered=[]
 if not isinstance(claims,list) or not claims:return {'valid':False,'errors':errors+['claims_not_list']}
 for claim in claims:
  if not isinstance(claim,dict) or set(claim)!={'text','evidence_ids'}:errors.append('claim_schema');continue
  ids=claim['evidence_ids'];text=claim['text']
  if not isinstance(text,str) or not isinstance(ids,list) or not ids or any(not isinstance(i,str) for i in ids):errors.append('claim_types');continue
  seen.extend(ids)
  if any(i not in allowed for i in ids):errors.append('unknown_evidence');continue
  if re.search(r'\d|\[[^]]*\]|<[^>]*>',text):errors.append('unbound_number_or_citation')
  if re.search(r'制造|装配错误|人为|故意|manufactur|assembly defect|deliberately|physically verified|visual inspection confirms',text,re.I):errors.append('unsupported_cause')
  rels={allowed[i]['relation'] for i in ids}
  if len(rels)>1:errors.append('mixed_relation_claim')
  rel=next(iter(rels))
  if rel=='above upper bound' and re.search(r'低于|below|within',text,re.I):errors.append('direction_conflict')
  if rel=='below lower bound' and re.search(r'超过上界|above|exceeds? (?:the )?(?:upper|maximum)',text,re.I):errors.append('direction_conflict')
  if rel=='below required count' and re.search(r'高于|多于|超出规定数量|above|more than|exceeds? (?:the )?required',text,re.I):errors.append('count_direction_conflict')
  if rel=='above required count' and re.search(r'低于|少于|不足|below|fewer|less than',text,re.I):errors.append('count_direction_conflict')
  if rel=='unavailable' and not re.search(r'不可用|缺失|无法|未能|unavailable|missing|not available|cannot',text,re.I):errors.append('unavailable_omitted')
  if rel in ['above upper bound','below lower bound','above required count','below required count','differs from requirement','deviates from requirement','unavailable'] and re.search(r'all.*(?:requirements|constraints|checks).*(?:satisfied|met)|所有.*(?:均满足|均符合|全部通过)',text,re.I):errors.append('deviation_treated_satisfied')
  rendered.append(text.strip()+' '+''.join('['+i+']' for i in ids))
 if sorted(seen)!=sorted(allowed):errors.append('evidence_coverage')
 return {'valid':not errors,'errors':sorted(set(errors)),'decision':x.get('decision'),'claims':claims,'rendered_text':' '.join(rendered),'citations':sorted(seen),'checks_scope':'JSON/decision/source-ID coverage, numeric/citation exclusion, relation grouping, explicit missingness and selected lexical conflicts; unrestricted semantic accuracy requires separate review.'}
