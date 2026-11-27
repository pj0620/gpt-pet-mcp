# Robot tool test campaign, 2026-11-27 (in progress)

Every gpt-pet-mcp tool exercised on the real GPTPet with the user as operator: objects and the
person placed by tape measure, moves measured marker → bumper, the iPad on a tripod recording
(clips cut by wall clock), ROS pose from `scripts/pose_truth.py`, lidar from `scripts/clearance.py`.
Driver: `scripts/campaign.py`; logs and images under `~/robot-rig/campaign/2026-11-27/<station>/`.
Guide the user followed: `gptpet-plans/Robot Tool Test Setup 11_27_2026.html`.

## Pre-flight

| check | result |
|---|---|
| P1 iPad stream | OctoStream at rtsp://192.168.1.121:554/stream → mediamtx; recording since 11:43 the day before; frames and clips cut fine |
| P2 clocks | Mac vs pj-ubuntu offset −0.01 s (was +0.56 s on 11-26) |
| P3 server/ROS | robot rebooted at 10:04 (no tmux): client launch started by ssh; "Kinect streaming" + "mecanum_drive_controller is active" at 10:05; server health ok (camera 0.4 s, pose, scan, tilt node, detector, people); `pytest -m server` 7 passed |
| P4 clearance + HOME | lidar at HOME: ahead 1.19 m (toward the hallway), left 0.74, behind 0.93, right 0.93 (the user and props nearby at the time); HOME stored x −0.36, z −1.19, yaw 136.9 |
| P5 camera geometry | floor-plane fit: height 0.287 m (config 0.30), pitch −1.8° (level ±2°): pass. Forward-offset estimate not usable in this scene (camera centre ray hit something at 3.7 m, lidar 6.5 m down the hall); redo against a wall at S3 |
| P6 baseline | first view: Shoe 1.1 m, Person 1.3 m (the user placing the box), Shoe 1.6 m, Mirror 1.6 m; `get_map` image ok; status idle |
| P7 stop switch | `bin/stop_robot.py` and `bin/nav/nav_goal.py` present on pj-ubuntu |
| P8 doorway | the user's tape numbers not yet taken; map ray-cast along the heading: first occupied cell at 2.95 m, free width ≈ 1.5 m at 2–2.5 m (the hallway), no constriction found |

Scene: the robot stands on the hard floor at the edge of a rug, nose toward a hallway (a doorway
with a mirror at its end), a white sofa to its right.

## S1 detector recall and range

Protocol change at 10:30: the floor ticks were dropped; the user chooses the objects (guide page
0d) and tapes them from the bumper to the nearest face; the readings are the truth.

### Round 1, first attempt (10:11, user line "box 30 cm, backpack 60 cm, chair 90 cm")

Those were not bumper-to-nearest-face readings (the clip shows the box ≈ 0.75 m out, the chair in
the hallway). Views: the flat box labelled **Bench 0.9 m**, no Backpack, **Chair 2.3 m** (3/3),
HousePlant 3.7 m (a real plant down the hall). Kept as `s1_r1_attempt1/`; not scored.

### Round 1 (10:49, tape: Box 76.2 cm, Backpack 152.4 cm, Chair 228.6 cm)

| object | tape | seen | `distance_m` | error | bbox bearing | note |
|---|---|---|---|---|---|---|
| Box (flat cardboard lid, ~12 cm tall) | 0.762 | 0/3 as Box; 3/3 as **Bench** | 0.8 | +0.04 | — | label miss: the detector's "box" prompt does not match a flat lid; "bench" wins |
| Backpack (upright) | 1.524 | 3/3 | 1.7 | +0.18 (12 %) | −7.5° | just outside max(0.15 m, 10 %) |
| Chair (dining chair, in the hallway) | 2.286 | 3/3 | 2.3 | +0.01 | −5.3° | pass |

Other labels per view: Rug, Sofa, Pillow (all real), HousePlant 3.7 m (real), so the "false
labels" count of 4.3/view is mostly true extras; the only mislabel is Bench. ids not stable
across the three views (extras come and go). `list_destinations` 14 known.

A **false passage**: `psg:1 "doorway ahead, ~1.1 m wide, opens 1.7 m deep"` at 0.5 m, drawn in the
gap between the box and the sofa (bbox x 308–336, i.e. a 28 px strip). To look at in S3 with the
saved frame `regression/data/robot_frames/s1_r1.npz`.

Both detected objects sit 5–7° left of the camera's centre line; the clip shows them close to the
nose line, the backpack slightly left. Either a small camera yaw or placement; S2 (a person on the
nose line) and S10 settle it.

Camera geometry during the round: floor height 0.283 m, pitch −1.8°.

Clip: `s1_r1.mp4` (10:47:53 + 222 s), sheet `s1_r1_watch/sheet.jpg`.

### Round 2 (11:10, the user's objects: Box 76 cm, WaterBottle 152 cm, Chair 250 cm, SprayBottle 152 cm ahead / 41 cm left)

| object | tape | seen as | `distance_m` | error | bbox bearing | expected bearing | note |
|---|---|---|---|---|---|---|---|
| Box (the same flat lid) | 0.76 | **Bench** 3/3 | 0.8 | +0.04 | −8° | 0° | same label miss as round 1 |
| WaterBottle (brown bottle, on the line) | 1.52 | **GarbageCan** 3/3 | 1.5 | −0.02 | −5.2° | 0° | distance right, label wrong; a "water bottle" prompt is missing (only "bottle") |
| Chair | 2.50 | Chair 3/3 | 2.5 | 0.00 | −5.3° | 0° | pass; its lower half also labelled Stool/SideTable at 1.6 m (duplicate boxes) |
| SprayBottle (left of the line) | hypot(1.52, 0.41) = 1.57 | **Bottle** 3/3 | 1.6 | +0.03 | −20.7° | −15.1° | distance right; "Bottle" is the right family |

Every object on the nose line reads 5–8° **left** of the image centre, and the spray bottle's
bearing is 5.6° further left than its tape position: a consistent **camera yaw of about −5.5°**
relative to the base heading (the Kinect sits turned a little to the right, or the base's "nose"
differs from the lidar heading). Worth a `CAMERA_YAW_DEG` offset in the contract's bearing →
target conversion if S2/S10 confirm it with a person on the line.

Same false passage (`psg:1` at 0.5 m, 1.0–1.1 m wide) as in round 1: the gap between the box and
the sofa/rug. Extras per view: Rug, Sofa, Pillow, HousePlant (real), Stool and SideTable (duplicates
of the chair). Frames saved: `regression/data/robot_frames/s1_r2.npz`. Clip `s1_r2.mp4` (11:10:33
+ 19 s).

## S2 do_look (camera tilt) with a person on the nose line

### Spot 1 (11:15, the user at 2 m by tape; their line listed "100 cm" and "200 cm" for the two spots, the camera says 2.0 m)

- `do_look` texts verbatim ("Camera level (height 0.3 m)", "Camera tilted up 27 degrees (height
  0.3 m)"); `camera.tilt_deg` 0 → 27 → 0 twice: the contract side passes.
- Person at **2.0 m** (`distance_m`) against the tape's 2.0 m: pass. Level, the Person bbox
  touches the top of the frame (bbox[1] = 0), as expected at 2 m.
- **The Kinect did not physically tilt.** Evidence: the "up" frame is identical in framing to the
  level frame (same bbox [161, 0, 306, 246]; the tape on the floor at the same pixel rows); the
  floor-plane pitch read −1.9° while commanded up (it would read ≈ −29°); `/kinect/tilt_state`
  publishes a constant 0.0. The driver did receive the commands (`kinect_ros2` log: "tilt -> 27
  deg", "tilt -> 0 deg"; `tilt_deg` parameter applied). The unit is a **Kinect for Windows**
  (USB 045e:02c2 "Kinect for Windows NUI Motor", not the Xbox 360's 02b0): libfreenect's motor
  control for that device id is the suspect. Yesterday's "tilt up 27 / framed" results were the
  commanded value, not a measured one.
- Consequence for the contract: `camera.tilt_deg` reports the commanded angle, so a brain that
  looks up sees nothing new. To do on the robot side (kinect_ros2 fork): read the accelerometer for
  the real tilt, report it in `tilt_state`, and fix or document K4W motor control; until then
  `do_look` is a no-op on this unit and the attention manager's "tilt up when close" is cosmetic.
### Spot 2 (11:19, the user at 1 m by tape)

- Person `distance_m` **1.0–1.1 m** against 1.0 m: pass. Bbox touches the top (bbox[1] = 0) and
  sits in the right half of the frame because the robot had turned 17° left after spot 1.
- Tilt: same result, no physical motion (floor pitch −1.2° while "up"; identical framing).
- Second `RotateLeft 10` → the robot ended ≈ 27–35° left of HOME; returned with `campaign.py home`
  before S3.

- `do_rotate RotateLeft 10` after spot 1: SLAM yaw 138.6 → 121.2 = **17.4°** for 10° (the spin
  overshoot is worst for small angles: 10 → 17, 30 → 36, 90 → 99.5). The person's bbox moved
  right accordingly; the tilt (0) was kept.

## S3 passages at the hallway, and a measured 1 m roll (11:21–11:25)

- **Far views (from HOME, the hallway end ≈ 3 m away): no passage in 3/3.** The S1 false
  passage (box/sofa gap) disappeared with the box.
- **`MoveAhead 1.0`**: tape marker → bumper **100 cm**; wheel odometry 1.025 m forward, 0.0 m
  sideways, yaw −1.0°; the tool said "Successfully moved 1.00 m ahead". Pass (within 10 % + 5 cm).
  `pose_truth` had no SLAM pose this time (TF not up within its wait; the server's own pose was
  fine): make `pose_truth.py` wait for `map→base_link` longer.
- **Near views (1 m closer)**: view 1 `psg:3 "hallway ahead, ~1.6 m wide, opens 3.0 m deep"` at
  1.6 m (the hallway: the map says ≈ 1.5 m wide: plausible); views 2–3 `psg:4 "doorway ahead, ~0.9
  m wide, opens 0.9 m deep"` at 3.2 m (the door at the hallway's end) with a **degenerate bbox
  [260, 172, 260, 176]** (zero width): a rendering/bbox bug to fix in `objects.py`/passages. So
  the doorway shows up only from ≤ 2 m, and the kind flips between hallway and doorway from one
  view to the next (ids not stable).
- **Wall view (30° left, a wall ≈ 0.6 m away): no passage.** Pass. The floor-plane fit broke there
  (height 0.495, pitch −77°: RANSAC locked onto the wall), but the ahead depth vs lidar gave the
  **camera forward offset 0.084 m** against the configured 0.08: confirmed.
- `list_destinations` after S3 knows four passages (psg:1 opening ahead-right 1.5 m, psg:2 doorway
  0.8 m, psg:3 hallway 1.6 m, psg:4 doorway 0.9 m): the registry keeps stale/duplicate passages.
- The user's tape for the doorway (A1) is still open; the map gave the hallway ≈ 1.5 m and the
  far wall at 2.95 m from HOME.

Verdict: passages work at ≤ 2 m with the right order of magnitude (0.8–0.9 m doorway, 1.6 m
hallway) but not from 3 m; boundary candidate `GPTPET_MCP_FLOOR_TOL=0.10` and the offline sweep on
`s3_far.npz`/`s3_near.npz`.

## S4 map baseline (11:26)

`get_map` renders: unknown grey, free white, walls black, the robot's red triangle at HOME, green
object dots along the hallway, blue passage dots. Pass. `list_destinations` payload printed as `{}`
by the driver's short form (the full list is in the jsonl).

## S5 rotation (11:29–11:31, marker at the bumper)

| command | SLAM Δyaw | wheel odometry Δ | EKF Δ |
|---|---|---|---|
| RotateLeft 90 | 100.1 | 105.9 | 98.5 |
| RotateRight 90 | 100.7 | 106.6 | 101.6 |
| RotateLeft 30 | 36.9 | 39.9 | 33.8 |
| RotateRight 30 | 38.3 | 40.0 | 38.1 |
| RotateLeft 90 ×4 | 100.6, 99.5, 99.7, 98.4 (sum 398.2) | 105.8, 105.8, 106.6, 109.4 | 99.7, 98.3, 99.6, 97.0 |

Actual/commanded by SLAM: mean **1.145, CV 5.6 %**; the overshoot is closer to a constant
**+7 to +10°** per spin than to a scale (10 → 17, 30 → 37.5, 90 → 100): the Nav2 Spin behaviour's
deceleration/stop criterion, not odometry. Wheel odometry reads 6 % more than SLAM (an odometry
scale error on top), EKF sits between. Texts verbatim ("Successfully rotated"), tilt kept, no
translation (SLAM position unchanged within 2 cm). Decision table → "mean > 5° and CV < 15 %":
subtract the measured constant in `Moves.rotate` (env knob, default 7°) and open a gpt-pet-v2 task
to lower `max_rotational_vel`/`rotational_acc_lim` in the behavior server, plus the wheel
odometry scale. Closure after the four lefts: 38° over by SLAM → the bumper centre should sit
≈ 13 cm from the marker and the robot should point ≈ 38° left of the hallway. **The user's tape:
0 cm, points left.** The direction agrees with SLAM; the 0 cm does not fit a 38° error about the
wheelbase centre unless the spin centre sits at the bumper or the reading was coarse (the clip
shows clean spins on the spot). After the station the server's SLAM yaw read 12° left of HOME
(not 38°): SLAM re-localised during the spins, so the per-spin deltas (read within 1 s of each
spin) are the trustworthy figure and the "38° sum" is not. `home` reported itself already at the
goal (0.19 m, 12° off, inside Nav2's tolerance) and did not move; a `RotateRight 6` brought the
yaw from 124.7 to 137.0 (HOME 136.9), i.e. +6° overshoot again.

## S6 manual moves (11:38–, marker before each, tape after)

| move | tape marker → bumper | wheel odometry | EKF | tool text | note |
|---|---|---|---|---|---|
| MoveAhead 0.5 | **58.4 cm** | +0.527 fwd, −0.014 left, −1.2° | +0.526, −0.023, −1.3° | "Successfully moved 0.50 m ahead" | long by 8 cm (16 %); inside 10 % + 5 cm |
| MoveBack 0.5 | **55.4 cm** | −0.529 fwd, +0.001, 0.0° | −0.528, +0.003, −1.7° | "Successfully moved 0.50 m back" | long by 5 cm |
| MoveRight 0.3 | **20.3 cm**, body turned | 0.0 fwd, **−0.409** (right), −0.1° | +0.232 fwd, −0.305 right, **+72.1°** | "Successfully moved 0.31 m to the right" | wheels slipped (41 cm believed, 20 real); a **large uncommanded left turn**: EKF +72°, SLAM +55° (yaw 137 → 81.5), the iPad clip shows the body swing ≈ 45° between t = 15 and 19 s; the user's top-down photo shows the marker at the front corner. The tool noticed nothing |

| MoveLeft 0.3 | not measured | +0.007 fwd, **+0.981 left**, −0.7° | +0.718 fwd, +0.046, **−151.2°** | "Moved 0.05 of 0.30 m to the left, then blocked by an obstacle" (12.6 s) | the robot **spun ≈ 170°** (SLAM yaw 137 → 308) while the wheels kept strafing, ended 0.65 m from HOME at the rug edge by the chair, stopped by the lidar blocker check. The clip: the body starts turning within 2 s of the command and keeps turning for 12 s |

### Fix and re-test (12:00–12:09, campaign paused at the user's request "keep trying the left move until it works")

`Moves._strafe` now holds the heading: `wz = -STRAFE_YAW_KP (2.0) × (odom yaw − start yaw)`,
capped at 0.6 rad/s, lateral speed 0.07 m/s, abort as "blocked by heading drift (+N degrees)" past
35°, and a square-up turn on the spot to within 3° afterwards (commit on feature/mcp-server).
Server restarted 12:02. Trials with the robot turned 90° right so its left faced the open hallway:

| trial | tool text | SLAM | EKF | wheel odometry | server log |
|---|---|---|---|---|---|
| MoveLeft 0.3 (12:07:49) | "Successfully moved 0.31 m to the left" | left +0.315, fwd +0.024, Δyaw −0.7° | left +0.318, Δyaw −1.2° | left +0.279, fwd −0.122, **Δyaw +55°** (the correction the hold had to apply) | heading error peak 9.5°, final −0.8° |
| MoveRight 0.3 (12:08:36) | "Successfully moved 0.32 m to the right" | left −0.310, fwd +0.027, Δyaw +1.6° | left −0.322, Δyaw +2.7° | left −0.278, fwd −0.125, Δyaw −56° | peak 11.0°, final 0.8° |

Trial 3 (12:38:22, the user's marker down): tool 0.31 m, SLAM left +0.315, EKF +0.317, yaw −2.4°.
**The user: "it didn't go left"**; the iPad clip shows a small displacement, hard to judge at that
range. Trial 4 (12:45:11, `scripts/strafe_test.py` outside the server, speed 0.12 m/s): EKF left
+0.331 m in 3.8 s, yaw −0.7°; no clip: the iPad stream stopped at 12:42:17 (RTSP port refused).
Open question: the robot's sensors agree with each other but the EKF is wheel odometry + IMU and
SLAM is weakly constrained along the corridor axis the robot slides along, so neither is
independent of wheel slip. The user's tape decides.

Independent check (12:49): a right strafe bracketed by two camera views; the plant at 3.4 m shifted
14 px (21 px after removing the 1° yaw change) → the camera really moved **≈ 0.12–0.17 m** for an
odometry claim of 0.34 m. So the mecanum rollers slip about **2:1 sideways** on this floor, and
neither odometry nor SLAM (weak along the corridor axis) could see it. A `STRAFE_SLIP_RATIO` knob
(2.2) was added and tried; the lidar-sector check around a trial was too coarse to confirm it.

**The user's decision (12:52): stop strafing; treat the mecanum wheels like normal wheels.**
`MoveLeft`/`MoveRight` now turn 90° to the side, drive the distance with `DriveOnHeading`, turn
back and square up (`GPTPET_MCP_SIDESTEP_MODE=turn`, the default; `strafe` keeps the held strafe).
`do_rotate` now asks for the angle minus the measured constant overshoot
(`GPTPET_MCP_ROTATE_OVERSHOOT_DEG` 7). First trial (12:53:47, through the server):
"Successfully moved 0.30 m to the left"; **SLAM left +0.311 m, fwd +0.003, yaw −2.2°; wheels
+0.308; EKF +0.309**: for the first time all three agree, because a forward drive does not slip.
About 15 s per side-step instead of 4, and it needs the footprint clear to turn.

By its own sensors the held strafe moved the commanded distance within 2 cm and ended within 2° of the start
heading, but the camera parallax showed the body travelling about half of it. The wheels having to "turn" 55° per 0.3 m of side-step to keep the body straight is a
systematic coupling, not random slip: the base's mecanum kinematics (a wheel's lateral sign or a
wheel not pulling its share sideways) deserves a look in gpt-pet-v2's controller/hardware.

### S6 resumed with side-steps by turning (12:56–13:06, robot reset to HOME by the user, marker down)

| move | tape | wheel odometry | EKF | tool | time |
|---|---|---|---|---|---|
| MoveRight 0.3 (turn-drive-turn) | **38.1 cm** | −0.321 left, −0.028 fwd, +5.8° | −0.323, −0.002, +1.7° | "Successfully moved 0.30 m to the right" | 7.5 s |
| MoveLeft 0.3 (turn-drive-turn) | **33.1 cm** | +0.311, −0.030, −4.4° | +0.311, +0.004, −5.6° | "Successfully moved 0.30 m to the left" | 8.4 s |
| MoveAhead 1.0 | **100 cm** | +1.027, −0.025, −1.6° | +1.027, −0.015, −1.5° | "Successfully moved 1.00 m ahead" | 10.4 s |
| MoveBack 1.0 | **100 cm** | −1.024 fwd, +0.001 left, −0.2° | −1.023, +0.025, −1.7° | "Successfully moved 1.00 m back" | 10.9 s |

Side-steps by turning land within 3–8 cm of the request by tape (the drive's own long-running
bias at short distances) with the heading back within 2–6°; the square-up tolerance (3°) can be
tightened. The 1 m drive is exact by tape (100 cm) while the odometry says 103: the short moves run
long, the long one does not, in both S6 runs.

The forward moves run 5–8 cm long by tape while wheel odometry says 2.7 cm long: the wheel
odometry under-reads by ≈ 5 % on top (same direction as the spin finding). The strafe is the real
problem: on this floor a 0.3 m side-step yaws the robot tens of degrees and the IMU/EKF yaw races
(+72° in 4.6 s). Decision table → "> 5°: yaw hold in the strafe loop" (close the loop on the EKF
yaw with `wz`), and shorten strafes; until then `MoveLeft/MoveRight` should be treated as unreliable
by the brain. SLAM pose again missing from `pose_truth.py` (fix the TF wait).

## S7 blocked move (13:13, the user's blocker: a water bottle, nearest face 50 cm)

**Failed, and informatively.** The detector never labelled the bottle at 0.5 m (60 s of views:
too close for the depth, and "Bottle" was hit-or-miss at 1.5 m in S1). `MoveAhead 2.0` returned
**"Successfully moved 2.00 m ahead"** and, per the user, the robot pushed the bottle along in front
of it the whole way (no clip: the iPad had stopped streaming at 12:42). Nav2's DriveOnHeading only
stops for what the local costmap holds, and the costmap only holds what the lidar plane cuts; a
bottle is thin and low, so it was never there. The blocked-move text was never going to appear.

Consequences: (1) the blocked-move test needs a blocker the lidar sees (≥ 30 cm tall and wide, e.g.
the box on its edge or the chair); (2) the robot drives through anything below or thinner than the
lidar plane: a near-field check on the Kinect depth in the drive path (`do_move`, and the attention
approach) is the fix to build, so "blocked by Bottle" becomes possible.

### Fix and re-test (13:20–13:38, the user put the bottle back at 50 cm)

`Robot.obstacle_ahead_m()`: the latest depth frame, 2×2 subsampled, samples within 1.5 m kept
when they lie in a 0.5 m wide lane and 6–60 cm above the floor (camera 0.3 m up, level); the
60th-nearest sample, minus the camera-to-bumper offset (0.12 m). `Moves.move` refuses a
MoveAhead when that is under `LANE_STOP_M` (0.45 m) and cancels one in flight. Offline on a frame of
the scene: 2222 lane samples, bottle at 0.40 m from the bumper, heights 0.08–0.21 m. Server
restarted 13:26.

| run | result | robot's own numbers |
|---|---|---|
| MoveAhead 2.0 with the bottle at ≈ 0.40 m | **"Move blocked by an obstacle: did not move"** (refused in 0.1 s) | lane check 0.40 m |
| MoveBack 0.25, then MoveAhead 2.0 from ≈ 0.7 m | **"Moved 0.23 of 2.00 m ahead, then blocked by an obstacle"** | SLAM +0.271 m, wheels +0.267, EKF +0.267, yaw −1.1°; lane check 0.44 m when it stopped |

Both contract texts for a blocked move now occur for a thin, low obstacle. The blocker is unnamed
("an obstacle") because the detector never labelled the bottle this close: the "named blocker"
half stays partial until the detector sees it (a taller blocker, or a near-range re-detect at the
stop).

**The user's tape of the final gap: 0 cm**, and the robot's own picture shows the bottle against
it: the stop came late. Why, from the frames and the URDF:
- the Kinect returns **~0.52 m for anything closer than its minimum range** (two frames bottomed
  out at exactly 0.519/0.521 m with the bottle touching), so the check read "0.40 m" whatever the
  true distance under half a metre;
- the camera sits level 0.3 m up with a 43° vertical field: it **cannot see the floor closer than
  0.76 m and loses a 20 cm bottle inside ~0.25 m** (blind zone under the lens), so a low object
  disappears from the check as it gets close;
- the camera-to-bumper offset was wrong: the body is 0.21 m square (URDF), so its front face is
  0.105 m ahead of base_link and the Kinect's lens (0.084 m, measured at S3) is only ~2 cm behind it,
  not 12; the bottle after the "reset" was also much closer than 50 cm (its picture fills the frame).

Fix (13:43, deployed): `BUMPER_FORWARD_M` 0.105; samples at ≤ 0.53 m count as contact range (the
check returns 0); `LANE_STOP_M` 0.6 m so the stop comes while a low object is still in view. To be
re-measured with the bottle at a taped 60 cm.

## S8 navigation, first attempt (13:48): planning failed, and why

The user's placements: chair 200 cm, box 150 cm ahead / 60 cm right (taped before an unplanned
0.97 m advance of the robot during a self-test of the lane check; Nav2 brought it back to within
0.43 m). `set_nav_goal("obj:nope")` → **"Unknown destination id 'obj:nope'. Call get_current_view
or list_destinations first."** (pass). `do_look up` text pass. `set_nav_goal(chair)` →
**"Could not plan a path to 'obj:chair#1': no path to (1.80, −1.48)"**, twice; the box was never
detected (at 1.07 m ahead / 0.6 m right it sat at a 29° bearing, just outside the 57° view).

Diagnosis (raw `/global_costmap/costmap`, `/map`, `/scan`): the global costmap was 99–100 (lethal
/ inscribed) over almost the whole known area while the SLAM map looked clean at a glance; clearing
the costmap and even disabling its obstacle layer changed nothing, so the paint came from the
**static layer**: the SLAM map held hundreds of occupied specks and a false wall segment across
the hallway entrance (184 occupied cells within 1 m of the robot, one lethal cell 25 cm ahead),
each inflated by the inscribed radius. slam_toolbox's `occupancy_threshold` was the default **0.1**:
a person or object standing in the lidar for a few scans becomes a permanent wall. `robot_radius`
0.14 (the body's real half-diagonal) helped a little (1056 → 790 lethal cells within 1 m) but the
planner still failed beyond 0.8 m. The threshold is read at startup only.

Fix (13:57): `occupancy_threshold: 0.65` in `slam_toolbox_parameters.yaml` (gpt-pet-v2
feature/real-profile-tests, and patched in place on pj-ubuntu), server launch restarted (fresh
map), MCP server restarted (fresh registry and HOME). After: 0 occupied map cells within 1 m, the
planner reaches 0.6, 1.2, 1.8 and 2.1 m down the hallway. Lessons for the stack: the server launch
had been up 2.5 days accumulating specks; the costmap's `robot_radius` 0.20 is generous for a
0.21 m body.

## S8 navigation, second attempt on the fresh map (14:00)

- `set_nav_goal("obj:nope")` → the contract's unknown-id error (pass).
- Goal A = chair: planned at once (path 2.34 m, one corner), `state: following` (pass).
- Goal B (the plant down the hall, the box being outside the camera's view) was not in the view
  taken for it, so **preemption is still untested**.
- `cancel_nav` after 5 s → `{"canceled": true, "state": "canceled"}`, 0.48 m travelled (pass).
- Full drive to the chair: `succeeded` after 84 s for a 1.99 m path; `distance_traveled_m` climbed to
  **10.99 m** with no SLAM jumps (140 samples, none > 0.3 m): the controller dithered its way there
  (MPPI at 0.1 m/s on a fresh map), which the progress accumulator faithfully summed. The ETA logic
  and the brain's `max_wait_s` need to be read against that.
- **Arrival: the robot's camera frame is a blurry close-up of the chair's seat fabric**; the last
  view had no detections. Nav2 called it a success. The 0.5 m standoff is measured from the
  object's detected centre (the seat/back), the lidar only sees the legs, and the seat is above its
  plane, so the nose went under the seat's front edge. Backed off 0.3 m by hand (via `do_move`);
  the lidar then read 0.56 m ahead and 0.34 m on the right.

Fix (14:08, deployed): `OBJECT_STANDOFF_M` 0.5 → **0.7**, and the Kinect lane check now runs
inside Nav2 drives too (`RobotNavManager.set_lane_source`, polled every 0.3 s): a lane stop with
≤ 1.0 m of path left on a drive to an object or a person ends as `succeeded` (the camera kept the
standoff); earlier it ends `failed: blocked by X near (x, z) (seen by the camera N m ahead)`.

### S8 runs 3 and 4 (14:07, 14:10): the camera check tuned in the drive

- Run 3: unknown-id error pass; goal A planned; `cancel_nav` pass; the full drive ended
  **`failed: blocked near (0.88, −0.21) (seen by the camera 0.00 m ahead)`** after 0.52 m, with
  nothing within 1 m (lidar 1.09 m ahead afterwards; a fresh depth frame had 0 saturated lane
  samples): the robot was turning past the sofa and one frame showed the minimum-range value.
  Fix: saturation has to persist two checks (0.6 s) and 40 samples (`LANE_SATURATED_FRAMES`,
  `LANE_SATURATED_POINTS`).
- Run 4: the full drive ended **`failed: blocked by Sofa near (1.40, 0.06) (seen by the camera 0.60 m
  ahead)`** after 0.77 m: a real reading (lidar 0.71 m ahead at the stop), the blocker named from
  the registry (pass for the naming), but the stop itself is wrong: the MPPI controller curves
  past the sofa closer than 0.6 m on purpose, the sofa is in the costmap. Fix: a separate in-drive
  threshold `NAV_LANE_STOP_M` 0.45 (at the Kinect's saturation floor: in practice "something inside
  0.5 m of the lens for two checks"), blind `do_move` drives keep 0.6.
- Preemption (goal B) still untested: the plant was not in the one view the station took for it.
  The station now falls back to any other visible object for goal B.

### S8 run 5 (14:13): preemption passes; the camera stops the drive for a can in the hallway

- **Preemption passes**: goal B (the rug, the only other object in view; 0.21 m path) returned
  `preempted_goal_id` = goal A's id, `get_nav_status` showed goal B; it finished in 2 s so the
  `cancel_nav` 3 s later found nothing to cancel (`{"canceled": false, "state": "succeeded"}`),
  which is the right answer for an idle robot (the cancel of a live drive passed in runs 2–4).
- Full drive: stopped after 1.06 m by the camera check (saturation, two checks) with a **black
  cylindrical can dead ahead at ≈ 0.5 m** (lidar 0.58 m) and the chair still 2.0 m away on the
  left; the station's arrival rule (≤ 1 m of path left) called it `succeeded`. The stop was right,
  the label wrong: the arrival now also requires the destination within 1.5 m
  (`LANE_ARRIVAL_TARGET_M`), otherwise "blocked by X"; and a Nav2 drive watches a 0.3 m wide lane
  (`NAV_LANE_HALF_WIDTH_M` 0.15) so furniture passed close on one side does not stop it.

### S8 run 6 (14:16)

Preemption passes again (goal B = the sofa). The full drive to the chair: 24 s, 2.58 m of
progress summed for 0.75 m of net advance (the controller dithering around the can in the
hallway), the robot ended turned ≈ 90° toward the sofa and the camera stopped it (saturation,
lidar 0.56 m ahead): `failed: blocked near (0.93, 0.11) (seen by the camera 0.00 m ahead)`. With
the can standing in the 1.5 m hallway between HOME and the chair, MPPI does not find its way past
at 0.1 m/s; the chair drive succeeded once (run 2) before the can was there. S8 verdict so far:
unknown id, immediate view, preemption, cancel: pass; full drive: pass only without the can;
standoff gap: not measured (the one arrival was into the seat).

### S8: the root cause of the hallway drives, and the drive that worked (14:20–14:37)

Tracing the pose during the chair drives showed the same shape every time: within 5 s of
starting, the SLAM yaw went from ≈ 92 (facing the hallway) to ≈ 5 while x kept growing, i.e. the
robot **turned ~90° and crabbed sideways down the hallway**. Nav2's MPPI controller had
`motion_model: "Omni"` (a trial since 11-24) and was commanding `vy`: the mecanum strafe that
S6 measured as slipping 2:1 and yawing, now under the controller, with the camera pointed at the
sofa instead of the direction of travel, which is what the lane check kept stopping. A live
`ros2 param set` to DiffDrive was accepted but left the optimizer resetting ("Failed to make
progress" ×26, "Optimizer reset" ×14, 14 recoveries, a `home` drive that timed out). A clean
restart of the server stack with `motion_model: "DiffDrive"`, `vy_max 0`, `vy_std 0` (gpt-pet-v2
feature branch, patched on pj-ubuntu) fixed it:

**14:37 drive to the chair: `succeeded` in 16 s, 1.54 m, heading within ±20° of the direction of
travel the whole way, no recoveries, no lane stop; lidar 0.88 m ahead at the end.** The user's
tape of the gap: pending.

So S8 passes: unknown id, immediate view, preemption (runs 5–6), cancel (runs 2–4), the full drive
(this run). Not done: the speed check (`s8 --step speed`) and the named-blocker half. The two
controller findings go to gpt-pet-v2: no Omni on this base, and the MPPI dithering at 0.1 m/s.

Detector verdict after S1: distances within 4 cm for every object seen (except the backpack's
+18 cm in round 1); labels miss for a flat box ("Bench"), a water bottle ("GarbageCan"); object
families right for the chair and the spray bottle. Actions for the boundary restart: add prompts
"cardboard box", "flat box" → Box, "water bottle" → Bottle; consider "spray bottle" → Bottle.
