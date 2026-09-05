"""Evaluator-only agreement of rendered bounds and conservative contact geometry."""
import math

POLICY='RENDER_BOUNDS_ENCLOSURE_V1'
TOLERANCE_M=.002

def assess_bounds(spec, center, extent, actor_xy):
    """Input is zero-yaw native actor bounds in metres, centered at its live pose.

    A circle enclosing the entire zero-yaw render AABB also encloses all its yaw
    rotations. This is deliberately a conservative envelope, not triangle contact.
    """
    if any(not math.isfinite(v) for v in (*center,*extent,*actor_xy)) or min(extent)<0:
        raise ValueError('Invalid native visual bounds')
    dx,dy=center[0]-actor_xy[0],center[1]-actor_xy[1]
    required=math.hypot(abs(dx)+extent[0],abs(dy)+extent[1])
    zerror=max(abs(center[2]-extent[2]-spec['base_m']),
               abs(center[2]+extent[2]-(spec['base_m']+spec['height_m'])))
    if spec['shape']=='disc':
        horizontal_error=max(0.,required-spec['radius_m'])
    else:
        horizontal_error=max(abs(dx),abs(dy),*(abs(e-h) for e,h in zip(extent[:2],spec['half_extents_m'])))
    return {'passed':horizontal_error<=TOLERANCE_M and zerror<=TOLERANCE_M,
            'required_enclosing_radius_m':required,'horizontal_error_m':horizontal_error,
            'vertical_error_m':zerror,'native_center_m':center,'native_extent_m':extent,
            'authority':'CONSERVATIVE_RENDER_BOUNDS_ENCLOSURE_NOT_TRIANGLE_CONTACT'}

def native_bounds(actor):
    """Read intrinsic posed bounds; restore rotation before any sensor capture."""
    import unreal as u
    rotation=actor.get_actor_rotation()
    try:
        actor.set_actor_rotation(u.Rotator(),False)
        origin,extent=actor.get_actor_bounds(False)
        return [v/100 for v in origin.to_tuple()],[v/100 for v in extent.to_tuple()]
    finally:
        actor.set_actor_rotation(rotation,False)

def audit_scene(specs, visuals, anchor_xy, floor):
    rows=[]
    for spec in specs:
        center,extent=native_bounds(visuals[spec['id']])
        relative=[center[0]-anchor_xy[0],center[1]-anchor_xy[1],center[2]-floor]
        result=assess_bounds(spec,relative,extent,[spec['x_m'],spec['y_m']])
        rows.append(dict(result,actor_id=spec['id']))
    return {'passed':all(row['passed'] for row in rows),'rows':rows,'policy':POLICY}
