import math


# compute vehicle speed based on tracked motion
def compute_speed(
    track_state,              # dictionary storing per-track historical data
    tid,                      # unique track ID for the vehicle
    foot_y,                   # y-coordinate of the bottom of the bounding box (vehicle "foot")
    box_h,                    # height of the bounding box (used for depth correction)
    frame_idx,                # current frame index
    fps,                      # frames per second of the video
    *,                        # force following arguments to be passed as keyword-only
    REF_BOX_HEIGHT,           # reference bounding-box height for depth scaling
    EMA_ALPHA,                # Exponential Moving Average smoothing factor
    METERS_PER_PIXEL,         # conversion ratio from pixels to meters
    MAX_REALISTIC_SPEED       # maximum plausible vehicle speed (km/h)
):
    # check this track ID has not seen before or not
    if tid not in track_state:
        # initialize tracking state for this vehicle
        track_state[tid] = {
            "y": foot_y,              # store initial vertical position
            "frame": frame_idx,       # store initial frame index
            "speed": 0.0,             # initialize speed to zero
            "last_seen": frame_idx,   # record last frame where vehicle was seen
        }

        # return zero speed for the first observation
        return 0.0

    # retrieve existing tracking state for this vehicle
    st = track_state[tid]

    # update last-seen frame index
    st["last_seen"] = frame_idx

    # calculate number of frames since last speed update
    df = frame_idx - st["frame"]

    # if frame difference is invalid or zero, return last known speed
    if df <= 0:
        return st["speed"]

    # convert frame difference into elapsed time (seconds)
    dt = df / fps

    # compute vertical movement in pixels
    motion = abs(foot_y - st["y"])

    # apply depth correction based on bounding box height
    # smaller box → vehicle farther away → larger correction factor
    depth = math.sqrt(REF_BOX_HEIGHT / max(1, box_h))

    # compute instantaneous speed:
    # pixels → meters → meters/second → km/h
    inst = (motion * depth * METERS_PER_PIXEL / dt) * 3.6

    # discard unrealistic or noisy speed values
    if inst < 1 or inst > MAX_REALISTIC_SPEED:
        inst = 0.0

    # apply Exponential Moving Average (EMA) to smooth speed values
    st["speed"] = EMA_ALPHA * inst + (1 - EMA_ALPHA) * st["speed"]

    # update stored vertical position for next calculation
    st["y"] = foot_y

    # update stored frame index for next calculation
    st["frame"] = frame_idx

    # return smoothed speed estimate
    return st["speed"]


# function to remove expired track states
def cleanup_tracks(track_state, frame_idx, ttl):
    # return list of track IDs that have not been seen within the TTL window
    return [
        tid for tid, st in track_state.items()  

        # check track is expired or not
        if frame_idx - st["last_seen"] > ttl
    ]