---
name: restart-gpt-pet-mcp
description: Restart the robot's MCP server (gpt-pet-mcp) on pj-ubuntu — stop the process on port 8000 in the tmux `server` session's `mcp` window, relaunch it from ~/gpt-pet-mcp with the ROS 2 environment, and health-check the endpoint. Use when the server is stale, hung, or running old code after a deploy.
---

Restart gpt-pet-mcp on pj-ubuntu (reached with `ssh pj-ubuntu.local`, key auth).

1. **Deploy the code first** when the working tree changed: `git push` here, then on pj-ubuntu
   `cd ~/gpt-pet-mcp && git pull && uv sync --extra dev` (`--extra detector` when the detector is
   wanted). Without a push, `rsync -a --exclude .venv --exclude .git ./ pj-ubuntu.local:gpt-pet-mcp/`.

2. **Find the running server:** `ssh pj-ubuntu.local 'ss -ltnp | grep ":8000 "'` gives the PID.
   With `pgrep -f`, build the pattern at runtime (`pat="g""pt-pet-mcp --http"; pgrep -f "$pat"`):
   a literal `gpt-pet-mcp --http` anywhere in the same remote command line (the relaunch string
   below, for one) makes `pgrep -f` match the remote shell itself, and `kill` then ends the ssh
   session with exit 255 and no output.

3. **Stop it** in a remote command of its own (never in the one that relaunches): `kill <PID>`,
   wait 2 s; `kill -9` only if it still listens. Its shutdown sends zero twists and cancels any
   Nav2 goal. pj-ubuntu's sshd drops a second connection opened within seconds of another, so
   reuse one multiplexed connection (`ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%r@%h-%p
   -o ControlPersist=1800 ...`) for the whole sequence, and start background work there with
   `setsid nohup ... < /dev/null & disown`.

4. **Relaunch** in the tmux session that holds the ROS 2 server launch, window `mcp`
   (`tmux new-window -t server -n mcp` if it does not exist):

   ```bash
   tmux send-keys -t server:mcp 'cd ~/gpt-pet-mcp && source /opt/ros/humble/setup.bash && source env/pj-ubuntu.env && mkdir -p log && uv run gpt-pet-mcp --http 2>&1 | tee -a log/gpt-pet-mcp.log' Enter
   ```

5. **Health-check** (up within a few seconds; the ROS graph fills in behind it):

   ```bash
   curl -s -o /dev/null -w "%{http_code}" -X POST http://pj-ubuntu.local:8000/mcp \
     -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" \
     -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"health","version":"0"}}}'
   ```

   Expect `200`. Then `tail -20 ~/gpt-pet-mcp/log/gpt-pet-mcp.log` for the health line (map, pose,
   nav2, camera, tilt node, detector).

6. **Report**: what was stopped, the health-check result, and the health line. The brain
   (`gpt-pet serve --profile real`) reconnects on its next call; its tool list is cached for
   `tool_list_cache_ttl_s`, so a changed tool list needs the brain restarted too.
