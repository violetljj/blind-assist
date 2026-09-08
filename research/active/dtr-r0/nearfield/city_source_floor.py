"""Source-only ground reconnaissance; hits are candidates, not walkability labels."""
import math


def grid_points(config):
    bounds=config['bounds_xy_m'];step=float(config['step_m'])
    if len(bounds)!=4 or not all(math.isfinite(v) for v in bounds) or not math.isfinite(step) or step<=0:
        raise ValueError('Finite bounds and positive spacing required')
    x0,y0,x1,y1=bounds
    if x1<=x0 or y1<=y0:raise ValueError('Ordered bounds required')
    nx,ny=math.floor((x1-x0)/step)+1,math.floor((y1-y0)/step)+1
    if nx*ny>10000:raise ValueError('Source floor grid exceeds 10000 rays')
    return [(x0+i*step,y0+j*step) for j in range(ny) for i in range(nx)]


def probe(u,world,config):
    top,bottom=float(config.get('top_z_m',3.)),float(config.get('bottom_z_m',-3.))
    if not all(math.isfinite(v) for v in (top,bottom)) or top<=bottom:raise ValueError('Ordered vertical range required')
    rows=[]
    for x,y in grid_points(config):
        hit=u.SystemLibrary.line_trace_single(world,u.Vector(x*100,y*100,top*100),
            u.Vector(x*100,y*100,bottom*100),u.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],u.DrawDebugTrace.NONE)
        row=dict(x=x,y=y,hit=bool(hit))
        if hit:
            values=hit.to_tuple();point=values[5];component=values[10]
            row.update(z=point.z/100,component_path=component.get_path_name() if component else None,
                       instance_index=values[13])
        rows.append(row)
    return dict(schema='city-source-floor-grid-v1',config=config,rows=rows,
        authority='Native complex Visibility-channel vertical first hit; not dense clearance, semantic sidewalk, or walkability proof')
