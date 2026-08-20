"""Compact safe policy for search, pursuit, reacquisition, and verified completion."""


def update_task_belief(previous_progress, observation):
    """Retain the greatest observed target nearness."""
    if observation.relative_nearness is not None and observation.relative_nearness > previous_progress:
        return observation.relative_nearness
    return previous_progress


def propose_actions(task_family, observation, task_belief):
    """Search directionally, align with clearance, then advance or finish."""
    if not observation.target_visible:
        if observation.target_bearing is not None and observation.target_bearing < 0:
            return ("SCAN_LEFT",)
        if observation.target_bearing is None and observation.left_free and not observation.right_free:
            return ("SCAN_LEFT",)
        return ("SCAN_RIGHT",)
    if observation.target_bearing is not None and observation.target_bearing < -5:
        if observation.left_free:
            return ("ALIGN_LEFT",)
        return ("SCAN_LEFT",)
    if observation.target_bearing is not None and observation.target_bearing > 5:
        if observation.right_free:
            return ("ALIGN_RIGHT",)
        return ("SCAN_RIGHT",)
    if task_family == "FIND_ALIGN_INTERACT" and observation.interaction_ready:
        return ("INTERACT",)
    if task_family != "FIND_ALIGN_INTERACT" and task_belief >= 0.85:
        return ("COMPLETE",)
    if observation.forward_free:
        return ("FORWARD",)
    return ("STOP",)


def select_action(proposals, observation, completion_claim):
    """Enforce the safety and completion-evidence gate on the first proposal."""
    if not proposals:
        return "STOP"
    if proposals[0] == "FORWARD" and not observation.forward_free:
        return "STOP"
    if proposals[0] == "ALIGN_LEFT" and not observation.left_free:
        return "STOP"
    if proposals[0] == "ALIGN_RIGHT" and not observation.right_free:
        return "STOP"
    if proposals[0] == "INTERACT" and not observation.interaction_ready:
        return "STOP"
    if proposals[0] == "COMPLETE" and not completion_claim:
        return "STOP"
    return proposals[0]


def detect_progress(previous_observation, observation):
    """Detect target acquisition or a strict increase in nearness."""
    if previous_observation is None:
        return False
    if observation.target_visible and not previous_observation.target_visible:
        return True
    if observation.relative_nearness is None:
        return False
    if previous_observation.relative_nearness is None:
        return True
    return observation.relative_nearness > previous_observation.relative_nearness


def recover_target(last_seen_bearing):
    """Scan toward the last known target side, defaulting right."""
    if last_seen_bearing is not None and last_seen_bearing < 0:
        return "SCAN_LEFT"
    return "SCAN_RIGHT"


def decide_completion(task_family, observation, task_belief):
    """Require visible centered targets and family-specific completion evidence."""
    if not observation.target_visible or observation.target_bearing is None:
        return False
    if not -5 <= observation.target_bearing <= 5:
        return False
    if task_family == "FIND_ALIGN_INTERACT":
        return observation.interaction_ready
    return task_belief >= 0.85
