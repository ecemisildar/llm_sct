# Delivery audit: run_20260912_235529_023152857

ROS timestamps confirm the chronological order below. Robot CSV elapsed times have different origins. Zone a is red; zone b is blue.

| Drop | Robot | Zone | Red | Blue | Red − blue |
|---:|---|---|---:|---:|---:|
| 1 | robot_2 | red | 1 | 0 | 1 |
| 2 | robot_1 | blue | 1 | 1 | 0 |
| 3 | robot_2 | blue | 1 | 2 | -1 |
| 4 | robot_1 | red | 2 | 2 | 0 |
| 5 | robot_0 | blue | 2 | 3 | -1 |
| 6 | robot_0 | red | 3 | 3 | 0 |

The recorded callbacks pass the balance requirement and report six deliveries, ending at 3 red and 3 blue.

Physical six-box completion is not verified: the saved world numbers boxes 1–6, but robots request boxes 0–5. Box 0 is absent and box 6 is never assigned. The numbering has been corrected for future runs; this run must be repeated to validate all six distinct boxes.

The saved task summary reports 18/18 because it repeats the team count of six for each of three robots; it is not a count of distinct boxes.

## Nearly simultaneous removals

ROS logs confirm removal of robot_1 carried cargo at Unix time 1789250262.176273018 and robot_0 carried cargo at 1789250262.186744414, about 10.5 ms apart. These correspond to the red and blue deliveries at drops 4 and 5. The balance moved from red 1 / blue 2 to 2 / 2 and then 2 / 3, so it stayed within ±1. Robot_0 selected another pickup about 0.294 s after its delivery. Ground-box DeleteEntity responses are not logged or checked, so the logs do not establish exact removal times or success for the original ground boxes.
