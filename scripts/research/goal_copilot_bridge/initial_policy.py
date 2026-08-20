"""Deterministic GOAL-COPILOT-1 baseline and exported candidate surface."""


def update_task_belief(previous_progress, observation):
    if observation.relative_nearness is None:
        return previous_progress
    if observation.relative_nearness > previous_progress:
        return observation.relative_nearness
    return previous_progress


def propose_actions(task_family, observation, task_belief):
    if not observation.target_visible:
        if observation.left_free and not observation.right_free:
            return ("SCAN_LEFT", "STOP")
        if observation.right_free and not observation.left_free:
            return ("SCAN_RIGHT", "STOP")
        return ("SCAN_LEFT", "SCAN_RIGHT", "STOP")
    if observation.target_bearing is not None and observation.target_bearing < -8:
        return ("ALIGN_LEFT", "STOP")
    if observation.target_bearing is not None and observation.target_bearing > 8:
        return ("ALIGN_RIGHT", "STOP")
    if task_family == "FIND_ALIGN_INTERACT" and observation.interaction_ready:
        return ("INTERACT", "STOP")
    if task_belief >= 0.85:
        return ("COMPLETE", "STOP")
    return ("FORWARD", "STOP")


def select_action(proposals, observation, completion_claim):
    if not proposals:
        return "STOP"
    if proposals[0] == "FORWARD" and not observation.forward_free:
        return "STOP"
    if proposals[0] == "ALIGN_LEFT" and not observation.left_free:
        return "STOP"
    if proposals[0] == "ALIGN_RIGHT" and not observation.right_free:
        return "STOP"
    if proposals[0] == "COMPLETE" and not completion_claim:
        return "STOP"
    return proposals[0]


def detect_progress(previous_observation, observation):
    if previous_observation is None:
        return False
    if previous_observation.relative_nearness is None:
        return observation.relative_nearness is not None
    if observation.relative_nearness is None:
        return False
    return observation.relative_nearness > previous_observation.relative_nearness


def recover_target(last_seen_bearing):
    if last_seen_bearing is not None and last_seen_bearing > 0:
        return "SCAN_RIGHT"
    return "SCAN_LEFT"


def decide_completion(task_family, observation, task_belief):
    if not observation.target_visible:
        return False
    if observation.target_bearing is None:
        return False
    if observation.target_bearing < -5 or observation.target_bearing > 5:
        return False
    if task_family == "FIND_ALIGN_INTERACT":
        return observation.interaction_ready
    return task_belief >= 0.85
