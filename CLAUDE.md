# gpt-pet-mcp: working notes

This repo is the MCP server for the **real** GPTPet robot. It reproduces ai2thor-mcp's robot
contract (12 tools + `view://rgb`) over ROS 2; the ROS 2 workspaces themselves live in
`gpt-pet-v2/ros2_ws` and run on pj-ubuntu (server) and the Jetson (robot). See README.md.

## Rules

- **Never change a contract tool's signature or docstring in one repo only.** `main.py` copies them
  verbatim from `ai2thor-mcp/python/ai2thor_mcp/main.py`; `scripts/diff_contract.py` and
  `regression/test_contract_offline.py` fail otherwise. A contract change starts in ai2thor-mcp, then
  the snapshot (`scripts/snapshot_sim_tools.py`) and this server follow.
- `fastmcp==2.14.2` and `pydantic==2.12.5` stay pinned to the simulator's versions: they generate
  the tool schemas both servers must publish identically.
- No mocks or fakes in tests. Offline tests cover geometry and the copied modules; everything else
  runs against the real simulator (`parity`), the real server (`server`) and the real robot (`robot`).
- Tool bodies live in `robot.py`; `main.py` stays thin and importable without rclpy (the robot is
  created lazily), so schemas can be listed anywhere.
- Lock rule (the simulator's): a manager's lock guards its own fields and is never held across a
  ROS call. Subscription callbacks only store messages; decoding happens in tool threads.

## Where things run

- pj-ubuntu (`ssh pj-ubuntu.local`): ROS 2 Humble, system Python 3.10 with rclpy, `uv` in
  `~/.local/bin`, the clone at `~/gpt-pet-mcp`; `source /opt/ros/humble/setup.bash && source
  env/pj-ubuntu.env` before `uv run`. The server launch runs in tmux session `server`; this server
  belongs in a window `mcp` of the same session.
- The Mac: offline tests and the parity test (simulator on 127.0.0.1:8765, see gpt-pet-v2's notes).
- The robot (gptpetclient.local) must be on for `robot` tests; the Kinect node and its tilt
  parameter come from the kinect_ros2 fork (gpt-pet-v2 submodule).

## Commands

```bash
uv run --extra dev pytest regression -m "not robot and not server and not parity"
python scripts/diff_contract.py
GPTPET_MCP_ROBOT_URL=http://pj-ubuntu.local:8000/mcp uv run --extra dev pytest regression -m server
GPTPET_MCP_SIM_URL=http://127.0.0.1:8765/mcp GPTPET_MCP_ROBOT_URL=http://pj-ubuntu.local:8000/mcp uv run --extra dev pytest regression -m parity
GPTPET_MCP_ROBOT_OK=1 uv run --extra dev pytest regression -m robot
```
