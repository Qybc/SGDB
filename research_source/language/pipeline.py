"""Evidence-bound language realization; detector decisions and numeric records remain frozen."""
from typing import Protocol
from copy import deepcopy
from contracts import contract,validate,digest
class Generator(Protocol):
 def __call__(self,messages:list[dict])->str:...
def explain(report:dict,generator:Generator,language='en')->dict:
 c=contract(report,language);raw=generator(c['messages']);checked=validate(raw,c)
 return {'source_trace_sha256':c['source_trace_sha256'],'decision':c['decision'],'score':c['score'],'threshold':c['threshold'],'language':language,'expression':'language_model' if checked['valid'] else 'structured_trace_fallback','claims':checked.get('claims',[]) if checked['valid'] else [],'text':checked.get('rendered_text') if checked['valid'] else None,'evidence':deepcopy(c['evidence']),'execution_state':c['execution_state_text'],'raw_llm_output':raw,'validation':checked,'detector_modified':False}
class RecordedGenerator:
 def __init__(self,contracts,responses):self.responses={digest(c['messages']):responses[rid]['raw_output'] for rid,c in contracts.items() if rid in responses}
 def __call__(self,messages):return self.responses[digest(messages)]
