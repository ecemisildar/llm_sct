# Balanced delivery automata

These models describe a color-independent delivery loop for three robots,
three generic boxes, and two delivery zones. Every state is marked; runtime
task completion is reported after the team confirms all three deliveries.

The shared balance abstraction has three states: `equal`,
`zone_a_has_extra`, and `zone_b_has_extra`. From `equal`, either zone is
allowed. After a confirmed delivery to one zone, only a drop toward the other
zone is enabled until its delivery is confirmed.

`delivered_zone_a` and `delivered_zone_b` are local, uncontrollable completion
events. Their remote equivalents are `received_delivered_zone_a` and
`received_delivered_zone_b`. The runtime must publish a delivery notification
after a successful local drop and convert notifications from other robots into
the corresponding `received_*` event.

The models assume the environment/runtime serializes successful drops and does
not authorize another drop until every robot has processed the preceding
delivery notification. `serialized_drop_plant.xml` prevents a single robot
from beginning another local drop before its current drop is confirmed; it is
not by itself a distributed mutex.

The existing `motion_plant.xml`, `obstacle_sensor.xml`, and
`collision_avoidance.xml` are reusable without color-specific changes.
`delivery_cycle_specification.xml` synchronizes their `search_object`,
`approach_object`, `search_zone`, and `approach_zone` actions with generic
object and Zone A/B observations. If a robot reaches a currently disallowed
zone, `search_zone` returns it to zone search instead of leaving it blocked.
