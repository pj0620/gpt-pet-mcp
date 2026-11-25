# gpt-pet-mcp

The MCP server that drives the **real GPTPet robot** with exactly the tool contract of
[ai2thor-mcp](../ai2thor-mcp), the simulator server: same tool names, input schemas,
descriptions and result shapes. The brain ([gpt-pet-v2](../gpt-pet-v2)) swaps between the two
with an ini profile (`sim` → ai2thor-mcp, `real` → this server at `http://pj-ubuntu.local:8000/mcp`).

The robot is a four-wheeled mecanum base with a Kinect a foot above the floor, an RPLIDAR and a
Jetson Nano; SLAM (slam_toolbox) and Nav2 run on the server **pj-ubuntu** from
`gpt-pet-v2/ros2_ws` (the ROS 2 workspaces live there, not here). This server runs on pj-ubuntu
next to that launch and speaks ROS 2 (rclpy) on one side and MCP (FastMCP, streamable HTTP) on
the other.

## The contract

| Tool | On the robot |
|---|---|
| `get_current_view()` | The Kinect frame (scaled to 500×375), annotated with numbered markers, plus the JSON payload: `agent {x, z, yaw}` from TF `map→base_link`, `camera {height_m, tilt_deg}`, `attention`, `destinations` — objects from an open-vocabulary detector on the GPU (YOLO-World, labels mapped to AI2-THOR object types, distance and map position from the depth) and passages from the depth image (the simulator's detector, unchanged), waypoints snapped to free cells of Nav2's global costmap. |
| `set_nav_goal(destination_id, speed_mps=0.35)` | `compute_path_to_pose` for the path (a destination with no path fails at once), then `navigate_to_pose`; returns immediately. Objects are approached to 0.5 m, people to 1.3 m, passages exactly. `speed_mps` is clamped to the contract's range and capped by Nav2's `vx_max` (0.1 m/s); the effective speed is returned. |
| `get_nav_status()` / `cancel_nav()` | The same states and fields as the simulator; `distance_traveled_m` integrated from TF, `distance_remaining_m` and `collisions` (recoveries) from Nav2's feedback. Preempted by a new goal, by `do_move`/`do_rotate`/`do_look`, by attention. |
| `list_destinations()` | Everything seen so far; object ids are stable tracks (`obj:sofa#1`), passages `psg:n`. |
| `get_map()` | slam_toolbox's `/map` in the ROS palette (grey unknown, white free, black occupied; 5 px per 0.05 m cell), cropped to the known area, with the simulator's overlays (robot triangle, remaining path, destination dots, goal star). |
| `do_move(action, moveMagnitude)` | Ahead/back through Nav2's `drive_on_heading`/`backup` behaviours (odometry-closed, collision-checked); left/right by streaming `/cmd_vel` (the mecanum base strafes) with a lidar guard; the simulator's texts, including partial progress. |
| `do_rotate(action, degrees)` | Nav2's `spin` behaviour. |
| `do_look(direction)` | The Kinect's tilt motor through the driver node's `tilt_deg` parameter: 27° up or level. |
| `set_attention` / `cancel_attention` / `get_attention_status` | Phase 4 (person detector + follower). Not served until then (`contract.ATTENTION_ENABLED`). |

Resource `view://rgb`: the last RGB frame as PNG.

Coordinates: the contract speaks AI2-THOR's frame (floor positions `x`/`z`, yaw in degrees
clockwise from +z). The mapping from ROS is `sim.x = ros.x`, `sim.z = ros.y`,
`sim.yaw = 90 − degrees(theta)`: no mirror, so the modules copied from the simulator
(`describe`, `annotate`, `perception`, the map overlays) work untouched (`frames.py`).

## Running

On pj-ubuntu (ROS 2 Humble, system Python 3.10 carries rclpy):

```bash
git clone https://github.com/pj0620/gpt-pet-mcp ~/gpt-pet-mcp && cd ~/gpt-pet-mcp
uv venv --python /usr/bin/python3.10 && uv sync --extra dev            # + --extra detector for objects
source env/pj-ubuntu.env                                              # ROS 2 + the launch's DDS settings
uv run python scripts/check_ros.py                                    # what the server sees on the graph
uv run gpt-pet-mcp --http                                             # streamable HTTP on 0.0.0.0:8000/mcp
```

Run it in the tmux session that holds the server launch (`tmux new-window -t server -n mcp`), or as a
user systemd unit later. Health check (what gpt-pet-v2's `simlaunch.mcp_alive` sends):

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://pj-ubuntu.local:8000/mcp \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"health","version":"0"}}}'
```

The server starts at once and degrades until the graph is up: without `/map` the map is a grey
placeholder, without TF the view says the pose is unknown, without the Kinect it says which topics
are silent, without Nav2 drives say so. The log prints the health every 10 s until everything is
green, then on change.

### Switches (environment)

| variable | default | meaning |
|---|---|---|
| `FASTMCP_PORT` | 8000 | port (also `--port`) |
| `GPTPET_MCP_CAMERA_HEIGHT` | 0.3 | Kinect height above the floor, m |
| `GPTPET_MCP_CAMERA_FORWARD` | 0.08 | Kinect ahead of base_link, m |
| `GPTPET_MCP_FRAME_WIDTH` | 500 | width of the frames the tools return |
| `GPTPET_MCP_CAMERA_TRANSPORT` | compressed | `compressed` (`/compressed` JPEG + `/compressedDepth` PNG) or `raw` |
| `GPTPET_MCP_RGB_TOPIC`, `GPTPET_MCP_DEPTH_TOPIC`, `GPTPET_MCP_CAMERA_INFO_TOPIC` | `/kinect/image_raw`, `/kinect/depth/image_raw`, `/kinect/camera_info` | Kinect topics |
| `GPTPET_MCP_KINECT_NODE` | `/kinect/kinect_ros2` | the node whose `tilt_deg` parameter is the tilt motor |
| `GPTPET_MCP_DETECTOR` | `yolov8s-worldv2.pt` | YOLO-World weights; `none` = passages only |
| `GPTPET_MCP_POSE_MODEL` | `yolov8n-pose.pt` | person detector (Phase 4) |
| `GPTPET_MCP_VOCABULARY` | packaged `vocabulary.json` | detector prompt → AI2-THOR label |
| `GPTPET_MCP_DETECTOR_CONF`, `GPTPET_MCP_DETECTOR_DEVICE` | 0.3, `cuda:0` | |
| `GPTPET_MCP_MAX_SPEED_MPS` | 0.1 | what the robot drives at (Nav2 MPPI `vx_max`) |
| `GPTPET_MCP_FLOOR_TOL` | 0.06 | passage detector: depth points this close to the floor are floor |
| `GPTPET_MCP_ROS_WAIT_S` | 0 | block startup until map, pose and Nav2 are seen (0: start at once) |

## Layout

```
python/gpt_pet_mcp/
  main.py        the FastMCP app: the tools, verbatim signatures and docstrings from ai2thor-mcp
  robot.py       the tool bodies: camera -> detector -> registry -> annotation, nav, moves, look, map
  ros_bridge.py  the rclpy node (executor thread, TF, subscriptions, publishers, action clients)
  camera.py      Kinect frames (decode, scale) and the tilt motor
  nav.py         RobotNavManager: Nav2 goals behind the simulator's NavManager API
  moves.py       do_move / do_rotate through Nav2's behaviours and /cmd_vel strafes
  detector.py    YOLO-World object detection; vocabulary.json maps prompts to AI2-THOR labels
  objects.py     tracks: stable ids and map positions for detected things
  frames.py      ROS <-> contract coordinate conversions
  maprender.py   the simulator's map palette and overlays on slam_toolbox's grid
  describe.py, annotate.py, perception.py, model.py   copied from ai2thor-mcp (import path only)
  attention.py   the simulator's attention helpers; the behaviour itself is Phase 4
  contract.py    which tools are served; config.py constants and switches
regression/      tests (see below); data/sim_tools.json is the simulator's tool list
scripts/         check_ros.py, snapshot_sim_tools.py, diff_contract.py, capture_map.py, nav_probe.py
env/pj-ubuntu.env
```

## Testing

No mocks: the live tiers talk to the real simulator, the real server and the real robot.

```bash
uv run --extra dev pytest regression -m "not robot and not server and not parity"   # offline: geometry, copied modules, contract vs snapshot
python scripts/diff_contract.py                                                     # signatures + docstrings vs ai2thor-mcp's main.py
GPTPET_MCP_ROBOT_URL=http://pj-ubuntu.local:8000/mcp uv run --extra dev pytest regression -m server   # the server on pj-ubuntu, robot may be off
GPTPET_MCP_SIM_URL=http://127.0.0.1:8765/mcp GPTPET_MCP_ROBOT_URL=http://pj-ubuntu.local:8000/mcp \
    uv run --extra dev pytest regression -m parity                                  # both servers live: same tools, schemas, shapes
GPTPET_MCP_ROBOT_OK=1 uv run --extra dev pytest regression -m robot                 # drives the robot: clear floor
```

Refresh the simulator snapshot after a contract change there:
`cd ../ai2thor-mcp && uv run python ../gpt-pet-mcp/scripts/snapshot_sim_tools.py --import --out ../gpt-pet-mcp/regression/data/sim_tools.json`.

The brain's own contract tests run against this server with
`cd ../gpt-pet-v2/python && GPTPET_TEST_PROFILE=real uv run pytest tests/live/test_mcp_live.py -m live`.

## Robot-side prerequisites (gpt-pet-v2/ros2_ws)

- The Kinect driver fork (`github.com/pj0620/kinect_ros2`) needs the `tilt_deg` parameter (the
  motor), `publish_rate_hz` (5 Hz over WiFi), `depth_registered` and a device-wait loop; the Jetson
  needs `ros-humble-image-transport-plugins` for the compressed topics.
- `server.launch.py`'s `cmd_vel_shaper` must relay unchanged (`r1_no_reverse:=false`, the default
  once the R1 reverse repair is confirmed): with the rewrite on, Nav2's `backup` becomes a stop and
  clockwise spins become arcs.
