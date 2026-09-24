"""Metadata-only partition authority; protected payloads remain unopened."""
import numpy as np

def declared_partition(row):
    values={str(row[k]).lower() for k in ('split','proposed_split','authoring_partition') if row.get(k) is not None}
    if len(values)>1:raise ValueError('Conflicting authoring partitions')
    value=next(iter(values),None)
    if value not in (None,'train','dev','test','locked_test','blind'):raise ValueError('Unknown authoring partition')
    return value

def guard_rows(rows,plan=None):
    for row in rows:
        partition=declared_partition(row)
        if row.get('data_role')!='Development' or partition in ('test','locked_test','blind'):
            raise ValueError('Protected/non-Development payload access prohibited')
        if plan is not None:
            item=plan['layouts'].get(row['layout_id'])
            if not item or item.get('partition') not in ('train','dev'):raise ValueError('Layout absent or protected in frozen plan')
            if item.get('physical_site_id')!=row['physical_site_id']:raise ValueError('Physical site differs from plan')
            if partition is not None and partition!=item['partition']:raise ValueError('Plan changes authored partition')

def planned_split(rows,plan):
    if plan.get('schema')!='cnh-development-merge-partitions-v1':raise ValueError('Partition schema required')
    guard_rows(rows,plan);sites={};layouts={};shared=set(plan.get('shared_development_sites',[]))
    authored_sites={}
    for item in plan['layouts'].values():
        if item.get('partition') not in ('train','dev','test') or not item.get('physical_site_id'):
            raise ValueError('Every authored layout needs a site and train/dev/test partition')
        authored_sites.setdefault(item['physical_site_id'],set()).add(item['partition'])
    if any(len(parts)>1 and (site not in shared or 'test' in parts) for site,parts in authored_sites.items()):
        raise ValueError('Authored site crosses partitions, including unopened test')
    for row in rows:
        item=plan['layouts'][row['layout_id']];site=row['physical_site_id'];part=item['partition']
        sites.setdefault(site,set()).add(part);layouts.setdefault(row['layout_id'],set()).add((site,part))
        if site in shared and (site!='Street200V7-single-street-block' or row['environment_category']=='alley'):
            raise ValueError('Shared-site exception is only for original Street block')
    if any(len(v)!=1 for v in layouts.values()):raise ValueError('Layout crosses site or partition')
    if any(len(parts)>1 and site not in shared for site,parts in sites.items()):raise ValueError('Physical site crosses train/dev')
    train=np.array([plan['layouts'][r['layout_id']]['partition']=='train' for r in rows])
    if not train.any() or train.all():raise ValueError('Need both authored train and dev rows')
    return train,~train,dict(train_layouts=sorted({r['layout_id'] for r,t in zip(rows,train) if t}),
        debug_validation_layouts=sorted({r['layout_id'] for r,t in zip(rows,train) if not t}),
        scope='DEVELOPMENT_AUTHORED_PARTITIONS_WITH_DISCLOSED_LEGACY_SHARED_SITE',physical_sites=sorted(sites),
        shared_development_sites=sorted(shared),protected_test_access='NOT_OPENED_OR_EVALUATED',
        rule='Frozen authored train/dev plan; protected test excluded before payload reads')
