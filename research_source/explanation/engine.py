"""Explain a frozen SGDB execution, without labels, pixels or neural models.

Rule adapters expose operands of the existing executable predicates. They never
choose a conclusion by image/category: conclusions follow evaluated predicates,
the frozen normalization, and a score-space sufficient-rule certificate.
"""
import copy
import math
import sys
from pathlib import Path

import numpy as np

EXECUTOR = Path(__file__).resolve().parents[1]/'binding'
sys.path.insert(0, str(EXECUTOR))
from original_executor import LABELS, apply_aux, raw, residual, score, ns

EXPECTED = ns['EXPECTED']
ZH = {'nut':'螺母','washer':'垫圈','screw':'螺丝','bolt':'螺栓',
      'green_tape':'绿胶带','red_tape':'红胶带','black_pencil':'黑铅笔',
      'red_pencil':'红铅笔','blue_eraser':'蓝橡皮','red_eraser':'红橡皮',
      'yellow_cookie':'黄饼干','black_cookie':'黑饼干','square_dish':'方盘',
      'round_dish':'圆盘','fork':'叉子','plate':'餐盘','spoon':'勺子',
      'short_screw_length':'短螺丝长度','long_screw_length':'长螺丝长度',
      'green_red_length_ratio':'绿/红胶带长度比','blue_red_length_ratio':'蓝/红木条长度比',
      'pencil_length_ratio':'黑/红铅笔长度比','eraser_length_ratio':'蓝/红橡皮长度比',
      'rope_reference_length_ratio':'绳子/参考木条长度比',
      'utensil_order':'餐具顺序','measurement_unavailable':'测量可用性'}


def fmt(v):
    if isinstance(v, bool): return '是' if v else '否'
    if isinstance(v, (int, float)):
        return str(int(v)) if float(v).is_integer() else f'{v:.5g}'
    if isinstance(v, list): return '['+', '.join(fmt(x) for x in v)+']'
    return str(v)


def aggregate(z):
    return float(max(z, default=0.) + .01*sum(z))


def certificate(norm, threshold):
    """Minimum-cardinality sufficient subset under the fixed nonnegative score.

    Non-selected rule scores are set to zero only inside this diagnostic. This
    is not an intervention on the image or an assertion of physical causality.
    Descending order is optimal for every fixed subset size for max+.01*sum.
    """
    # At exactly equal contributions, cite an available factual predicate
    # before the aggregate missing-evidence flag. Cardinality is unaffected.
    ordered = sorted(norm, key=lambda k: (-norm[k], k=='measurement_unavailable', k))
    full = aggregate(list(norm.values()))
    selected = []
    if full > threshold:
        for key in ordered:
            if norm[key] <= 0: continue
            selected.append(key)
            if aggregate([norm[k] for k in selected]) > threshold: break
        assert aggregate([norm[k] for k in selected]) > threshold
        shorter = aggregate([norm[k] for k in ordered[:len(selected)-1]])
        assert shorter <= threshold
        return {'kind':'sufficient_alarm_subset','rules':selected,
                'subset_score':aggregate([norm[k] for k in selected]),
                'best_one_fewer_score':shorter,'minimum_cardinality':len(selected),
                'full_score':full,'threshold':threshold,
                'diagnostic_space':'frozen normalized rule scores; omitted scores set to zero'}
    return {'kind':'full_non_alarm_trace','rules':ordered,
            'full_score':full,'threshold':threshold,
            'minimum_cardinality':None,'diagnostic_space':'all executed rules retained'}


class TraceBuilder:
    def __init__(self, category, task, record, config, support_events):
        self.cat, self.task, self.r, self.cfg = category, task, record, config
        self.f = apply_aux(copy.deepcopy(record['facts']), config.get('aux'))
        self.labels = LABELS[category]
        self.events = {e['object_index']:e for e in support_events}
        self.by = {label:[i for i,o in enumerate(record['objects']) if o['class']==c]
                   for c,label in enumerate(self.labels)}
        self.expected = dict(zip(self.labels, EXPECTED.get(category, [])))
        self.expected.update({'breakfast_box':{'tangerine':2,'peach':1},
                              'juice_bottle':{'bottle':1,'juice':1,'image_label':1,'text_label':1},
                              'pushpins':{'pushpin':15,'cell':15},
                              'screw_bag':{'nut':2,'washer':2,'screw':2},
                              'splicing_connectors':{'terminal':2,'cable':1}}.get(category, {}))
        self.base, self.mm, _ = raw(category, copy.deepcopy(self.f))
        self.rr, self.unknown = residual(category, record['facts'], config)

    def ids(self, labels):
        return sorted(i for label in labels for i in self.by.get(label, []))

    def operand(self, key):
        f, cat = self.f, self.cat
        spec = {'key':key,'kind':None,'observed':None,'requirement':None,
                'source_paths':[],'object_indices':[],'unit':None}
        def set_(kind, observed, requirement, paths, labels=(), indices=None, unit=None):
            spec.update(kind=kind,observed=observed,requirement=requirement,
                        source_paths=paths,object_indices=self.ids(labels) if indices is None else indices,
                        unit=unit)
            return spec
        if key == 'measurement_unavailable':
            return set_('availability', self.unknown, {'operator':'empty'},
                        ['executor.unknown'], indices=[i for i,o in enumerate(self.r['objects']) if o['center'] is None])
        if key in self.cfg['intervals']:
            labels=[];indices=None;kind='cached_measurement';unit='cached units'
            if key in ['short_screw_length','long_screw_length']:
                ids=[i for i in self.by['screw'] if self.r['objects'][i]['axis'] is not None]
                ids.sort(key=lambda i:math.dist(*[(np.asarray(p)*self.r['native_size']).tolist() for p in self.r['objects'][i]['axis']]))
                indices=ids[:1] if key.startswith('short') else ids[-1:]
                kind='bound_geometry';unit='px'
            recipes={'green_red_length_ratio':['green_tape','red_tape'],
                     'blue_red_length_ratio':['blue_stick','red_stick'],
                     'pencil_length_ratio':['black_pencil','red_pencil'],
                     'eraser_length_ratio':['blue_eraser','red_eraser'],
                     'rope_reference_length_ratio':['rope','reference_stick']}
            if key in recipes:labels=recipes[key];kind='bound_geometry';unit='ratio'
            for label in self.labels:
                if key in [label+'_cx',label+'_cy'] and 'attributes' in f:
                    labels=[label];kind='bound_geometry';unit='normalized coordinate'
            if key == 'fruit_rightmost_center':labels=['tangerine','peach'];kind='bound_geometry';unit='normalized x'
            if key in ['nut_visible_count','banana_visible_count']:
                labels=['nut' if key.startswith('nut') else 'banana_chips'];kind='decoded_count';unit='count'
            path='facts.measurements.'+key
            if key.startswith('cable_length_'):path='facts.cable_length'
            lohi=self.cfg['intervals'][key]
            return set_(kind,self.mm.get(key),{'operator':'interval',**lohi},[path],labels,indices,unit)
        if key.endswith('_count'):
            label=key[:-6]
            if label in self.expected:
                return set_('decoded_count',f['counts'][label],{'operator':'equal','value':self.expected[label]},
                            ['facts.counts.'+label],[label],unit='count')
        if key.endswith('_presence'):
            label=key[:-9]
            return set_('decoded_count',f['counts'][label],{'operator':'at_least','value':1},
                        ['facts.counts.'+label],[label],unit='count')
        if key == 'fruit_icon_count':
            return set_('derived_count',f['fruit_icon_count'],{'operator':'equal','value':1},
                        ['facts.fruit_icon_count'],self.labels[4:],unit='count')
        if key == 'juice_icon_correspondence':
            return set_('categorical_equality',[f.get('liquid_type'),f.get('fruit_icon_type')],
                        {'operator':'pair_equal'},['facts.liquid_type','facts.fruit_icon_type','facts.liquid_colour_descriptor'],self.labels[1:])
        if key == 'one_pin_per_cell':
            return set_('occupancy',f['pins_per_detected_cell'],{'operator':'every_equal','value':1,'empty_residual':1},
                        ['facts.pins_per_detected_cell'],['pushpin','cell'])
        if key == 'pins_outside_cells':
            return set_('nonnegative_zero',f['pins_outside_cells'],{'operator':'equal','value':0},
                        ['facts.pins_outside_cells'],['pushpin','cell'])
        if key in ['color_port_count','port_correspondence','entry_facing']:
            blocks=f['blocks']
            if key == 'color_port_count':
                return set_('occupancy',[len(b['ports']) for b in blocks],
                            {'operator':'every_equal','value':{'yellow':2,'blue':3,'red':5}[f['cable_color']],
                             'empty_residual':{'yellow':2,'blue':3,'red':5}[f['cable_color']]},
                            ['facts.blocks','facts.cable_color'],['terminal','connector'])
            if key == 'port_correspondence':
                return set_('categorical_equality',[b['occupied_row'] for b in blocks],
                            {'operator':'pair_distance'},['facts.blocks'],['terminal','connector'])
            facing=[]
            for i,b in enumerate(blocks):
                if b['entry'] is not None and b['ports']:
                    delta=b['entry'][0]-np.mean([p['center'][0] for p in b['ports']])
                    facing.append(float(max(0.,-delta if i==0 else delta)/.1))
            return set_('nonnegative_zero',max(facing,default=1.),{'operator':'equal','value':0},
                        ['facts.blocks'],['terminal','connector'])
        if key.endswith('_bin'):
            label=key[:-4];coords=[o['center'] for o in f['attributes'][label+'_instances']]
            axis=1 if cat=='Blocks' else 0
            if cat in ['Tools','Blocks']:
                position=self.labels.index(label);lo,hi=position/3,(position+1)/3
            else:lo,hi=(0.,.5) if label in ['black_pencil','blue_eraser'] else (.5,float('inf'))
            return set_('region',[p[axis] for p in coords],
                        {'operator':'every_half_open','lo':lo,'hi':None if math.isinf(hi) else hi,'axis':axis},
                        ['facts.attributes.'+label+'_instances'],[label])
        if key.endswith('_left_of_pencil'):
            eraser=key[:-15];pencil={'blue_eraser':'black_pencil','red_eraser':'red_pencil'}[eraser]
            a=f['attributes'][eraser+'_instances'][0]['center'][0]
            b=f['attributes'][pencil+'_instances'][0]['center'][0]
            return set_('ordered_pair',[a,b],{'operator':'less'},
                        ['facts.attributes.'+eraser+'_instances','facts.attributes.'+pencil+'_instances'],[eraser,pencil])
        if key in f.get('relation_features',{}):
            value=f['relation_features'][key]
            return set_('derived_relation',value,{'operator':'zero_error' if key.endswith('_error') else 'nonnegative'},
                        ['facts.relation_features.'+key],self.labels)
        raise KeyError(('No trace adapter for an executed rule',cat,key))

    @staticmethod
    def reconstruct(spec):
        v=spec['observed'];q=spec['requirement'];op=q['operator']
        if op=='empty':return float(bool(v))
        if op=='interval':return float(max(0.,q['lo']-v,v-q['hi'])/q['scale'])
        if op=='equal':return float(abs(v-q['value']))
        if op=='at_least':return float(v==0)
        if op=='pair_equal':return float(v[0]!=v[1])
        if op=='pair_distance':return float(abs(v[0]-v[1]))
        if op=='every_equal':return float(max((abs(x-q['value']) for x in v),default=q['empty_residual']))
        if op=='every_half_open':return float(sum(not q['lo']<=x<(q['hi'] if q['hi'] is not None else math.inf) for x in v))
        if op=='less':return float(v[0]>=v[1])
        if op=='zero_error':return float(max(0.,v))
        if op=='nonnegative':return float(max(0.,-v))
        raise KeyError(op)

    def build(self):
        execution=score(self.cat,self.r['facts'],self.cfg,self.task)
        threshold=self.cfg['thresholds'][self.task]['threshold']
        cert=certificate(execution['normalized_residuals'],threshold)
        rules=[]
        for j,(key,r) in enumerate(execution['rule_residuals'].items()):
            spec=self.operand(key);check=self.reconstruct(spec)
            assert math.isclose(check,r,abs_tol=1e-12,rel_tol=1e-12),(self.cat,key,check,r,spec)
            z=execution['normalized_residuals'][key]
            dep=[{'object_index':i,**self.r['objects'][i],
                  'association':self.r['association'][i],
                  'support_readout':self.events.get(i)} for i in spec['object_indices']]
            rules.append({**spec,'id':f'R{j+1:02d}','evidence_id':f'E{j+1:02d}',
                          'raw_residual':r,'normalized_score':z,
                          'residual_state':'unavailable' if key=='measurement_unavailable' and self.unknown else ('deviation' if r>1e-8 else 'satisfied'),
                          'in_certificate':key in cert['rules'],
                          'objects':dep,
                          'normal_reference_count':len(self.cfg['references'][self.task].get(key,[0.])),
                          'single_rule_score':1.01*z,
                          'score_without_this_rule':aggregate([vv for kk,vv in execution['normalized_residuals'].items() if kk!=key]),
                          'executor_source':'features.raw / program.residual / program.score'})
        return {'schema':'SGDB execution-trace explanation v1','category':self.cat,'task':self.task,
                'score':execution['score'],'threshold':threshold,
                'predicted_anomaly':execution['score']>threshold,
                'evidence_status':execution['evidence_status'],'unknown':self.unknown,
                'certificate':cert,'rules':rules,'execution':execution}


def rule_sentence(r):
    key=r['key'];q=r['requirement'];v=r['observed'];op=q['operator']
    label=ZH.get(key[:-6],key[:-6])+'数量' if key.endswith('_count') else ZH.get(key,key)
    eid=r['evidence_id']
    if op=='empty':
        return (f'以下证据不可用：'+ '、'.join(ZH.get(x,x) for x in v) if v else '未记录必需测量缺失')+f' [{eid}]'
    if op=='interval':
        direction='低于下界' if v<q['lo'] else ('超过上界' if v>q['hi'] else '位于区间内')
        unit=' px' if r['unit']=='px' else ''
        return f'{label}为 {fmt(v)}{unit}，参照区间 [{fmt(q["lo"])}, {fmt(q["hi"])}]，{direction} [{eid}]'
    if op in ['equal','at_least']:
        verb='要求等于' if op=='equal' else '要求至少'
        return f'{label}观测为 {fmt(v)}，{verb} {fmt(q["value"])} [{eid}]'
    if op=='every_half_open':
        return f'{label}的坐标为 {fmt(v)}，要求落入 [{fmt(q["lo"])}, {fmt(q["hi"]) if q["hi"] is not None else "+inf"}) [{eid}]'
    if op=='nonnegative':return f'{label}观测为 {fmt(v)}，要求不小于 0 [{eid}]'
    if op=='zero_error':return f'{label}误差为 {fmt(v)}，要求等于 0 [{eid}]'
    if op=='less':return f'{label}的两个横坐标为 {fmt(v)}，要求前者小于后者 [{eid}]'
    if op=='every_equal':return f'{label}的逐组观测为 {fmt(v)}，每组要求为 {fmt(q["value"])} [{eid}]'
    return f'{label}的两项观测为 {fmt(v)}，要求'+('一致' if op=='pair_equal' else '索引相同')+f' [{eid}]'


def render(report):
    alarm=report['predicted_anomaly'];rules=report['rules'];cert=report['certificate']
    intro=f'判定为'+('异常' if alarm else '正常')+f'：分数 {report["score"]:.6f} '+('>' if alarm else '<=')+f' 阈值 {report["threshold"]:.6f}。'
    positive=[r for r in rules if r['normalized_score']>0]
    if alarm:
        selected=[next(r for r in rules if r['key']==k) for k in cert['rules']]
        clauses=[rule_sentence(r) for r in selected]
        clauses.append(f'保留上述 {len(selected)} 条规则的分数贡献，聚合分数为 {cert["subset_score"]:.6f}，仍超过报警阈值。')
    elif positive:
        selected=sorted(positive,key=lambda r:(-r['normalized_score'],r['key']))[:2]
        clauses=[rule_sentence(r) for r in selected]
        clauses.append('这些规则存在分数贡献，但全部已执行规则的聚合分数未超过阈值。')
    else:
        deviations=[r for r in rules if r['raw_residual']>1e-8]
        # A zero-score normal report still cites concrete executed predicates.
        # Count predicates are discovered by type, never by category/case name.
        selected=deviations[:2] if deviations else [r for r in rules if r['kind']=='decoded_count' and r['requirement']['operator']=='equal']
        clauses=[rule_sentence(r) for r in selected]
        clauses.append('已执行规则的标准化分数均为 0。'+('原始偏离未超出冻结正常参照。' if deviations else '本次执行未产生规则偏离。'))
    if report['unknown'] and not any(r['key']=='measurement_unavailable' for r in selected):
        unavailable=next(r for r in rules if r['key']=='measurement_unavailable')
        clauses.append(rule_sentence(unavailable))
    other=[r for r in rules if r['raw_residual']>1e-8 and r not in selected]
    if other:clauses.append('完整轨迹另保留 '+str(len(other))+' 条原始偏离规则，见明细。')
    cited={r['evidence_id'] for r in selected}
    if report['unknown']:cited.add(next(r['evidence_id'] for r in rules if r['key']=='measurement_unavailable'))
    return {'decision_text':intro,'explanation_text':'；'.join(clauses),
            'cited_evidence_ids':sorted(cited),'all_rule_ids':[r['id'] for r in rules],
            'selected_rule_keys':[r['key'] for r in selected],
            'generation':'generic predicate verbalization from verified trace; no image-specific conclusions; no LLM'}


def explain(category,task,record,config,support_events=()):
    # This API consumes factual fields only. Dataset truth and violation labels
    # are never passed to the rule selection or language composition functions.
    assert set(record)<= {'objects','facts','association','native_size'}
    report=TraceBuilder(category,task,record,config,support_events).build()
    report['language']=render(report)
    assert not any(x in sys.modules for x in ['torch','tensorflow','transformers'])
    return report
