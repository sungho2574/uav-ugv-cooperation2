// GCS 3D dashboard front end. Polls the Flask REST API (backed by ROS2 topics
// on the gcs_node side) and renders everything with Three.js. World frame is
// (x, y) on the ground plane + z = altitude; mapped into Three.js's right-handed
// Y-up space as (-x, z_world, y_world) so "z up" in Three.js == altitude.
//
// Why the world x is NEGATED here: the operator watches the arena from the
// +y (far) side (see the camera setup in setMap), and from that viewpoint the
// world +x direction runs to their LEFT. Mapping world x straight to Three.js
// +x put it on screen-right instead, so on real hardware a drone flying +x
// appeared to move the wrong way across the map (x mirrored; y, the depth
// axis, still looked right). Negating x reflects the whole scene about the
// x-axis so screen left/right matches the physical room. This is purely a
// rendering convention -- /states, control_node commands and the drone's own
// estimate are all already world-consistent (the drone flies correctly); only
// the on-screen picture was mirrored. Every world->Three.js path must apply
// the same negation: w2t() below, and polygonShape() (ground + zone meshes).

const FALLBACK_COLORS = { cf1: '#ff5555', cf2: '#55aaff', cf3: '#55dd77' };

function w2t(x, y, z) {
  return new THREE.Vector3(-x, z || 0, y);
}

function makeTextSprite(text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 128; canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = 'bold 36px sans-serif';
  ctx.fillStyle = color || '#ffffff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.6, 0.3, 1);
  return sprite;
}

function polygonShape(points) {
  // Shapes are built in the XY plane then rotateX(PI/2)'d onto the ground, so a
  // shape point (sx, sy) lands at Three.js (sx, 0, sy). To match w2t()'s
  // world-x negation (see the file header), the shape's x must be -world_x.
  const shape = new THREE.Shape();
  points.forEach(([x, y], i) => {
    if (i === 0) shape.moveTo(-x, y); else shape.lineTo(-x, y);
  });
  return shape;
}

function lineLoopFromPoints(points, color, yLift) {
  const pts = points.concat([points[0]]).map(([x, y]) => w2t(x, y, yLift || 0.01));
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.Line(geom, new THREE.LineBasicMaterial({ color }));
}

// World-frame axis gizmo, drawn at the world origin (0,0,0) -- which is also
// the map boundary's first corner (see mission_map.yaml), so it lands right
// on a visible corner of the coverage area instead of floating in empty
// space. X=red, Y=green, Z=blue (altitude, "up" in Three.js after w2t()).
const WORLD_AXES = [
  { dir: [1, 0, 0], color: '#ff4444', label: 'X' },
  { dir: [0, 1, 0], color: '#44ff66', label: 'Y' },
  { dir: [0, 0, 1], color: '#4499ff', label: 'Z' },
];

// Populates `group` directly (rather than returning a nested group) so
// clearGroup() -- which only disposes an object's *own* geometry/material,
// not recursively -- can dispose these children correctly on rescale.
function populateWorldAxes(group, length) {
  const origin = w2t(0, 0, 0);
  group.add(new THREE.Mesh(
    new THREE.SphereGeometry(length * 0.03, 10, 10),
    new THREE.MeshBasicMaterial({ color: 0xffffff })));
  WORLD_AXES.forEach(({ dir, color, label }) => {
    const end = w2t(dir[0] * length, dir[1] * length, dir[2] * length);
    const geom = new THREE.BufferGeometry().setFromPoints([origin, end]);
    group.add(new THREE.Line(geom, new THREE.LineBasicMaterial({ color })));
    const sprite = makeTextSprite(label, color);
    sprite.position.copy(end);
    group.add(sprite);
  });
  const originLabel = makeTextSprite('(0,0)', '#ffffff');
  originLabel.position.copy(origin).add(new THREE.Vector3(0, length * 0.15, 0));
  group.add(originLabel);
}

function circlePoints(cx, cy, radius, segments) {
  const pts = [];
  const n = segments || 20;
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2;
    pts.push([cx + radius * Math.cos(a), cy + radius * Math.sin(a)]);
  }
  return pts;
}

// setZones/setPaths used to rebuild hundreds of THREE.js meshes from scratch
// on *every* poll (every 300ms) without ever disposing the old geometries/
// materials -- an unbounded WebGL GPU-memory leak that eventually crashes the
// browser tab. disposeObject3D()/clearGroup() must be used everywhere a
// group's children get thrown away and rebuilt.
function disposeObject3D(obj) {
  const materials = Array.isArray(obj.material) ? obj.material : (obj.material ? [obj.material] : []);
  materials.forEach((m) => {
    if (m.map) m.map.dispose();
    m.dispose();
  });
  if (obj.geometry) obj.geometry.dispose();
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children[0];
    disposeObject3D(child);
    group.remove(child);
  }
}

class GcsScene {
  constructor(container) {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x14161a);

    this.camera = new THREE.PerspectiveCamera(
      55, container.clientWidth / container.clientHeight, 0.05, 500);
    this.camera.position.set(5, 12, 14);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(5, 0, 3);
    this.controls.update();

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 0.6);
    sun.position.set(10, 20, 5);
    this.scene.add(sun);

    this.worldAxesGroup = new THREE.Group(); this.scene.add(this.worldAxesGroup);
    populateWorldAxes(this.worldAxesGroup, 1.5); // rescaled once the map loads, see setMap()
    this.mapGroup = new THREE.Group(); this.scene.add(this.mapGroup);
    this.zoneGroup = new THREE.Group(); this.scene.add(this.zoneGroup);
    // Planned (gray dashed) path barely ever changes after the mission is
    // planned, unlike the visited (white) tube which grows every leg -- kept
    // as separate groups so the static one can be cache-skipped instead of
    // rebuilding it from scratch on every single poll for no reason.
    this.plannedPathGroup = new THREE.Group(); this.scene.add(this.plannedPathGroup);
    this.visitedPathGroup = new THREE.Group(); this.scene.add(this.visitedPathGroup);
    this.markerPlaceholderGroup = new THREE.Group(); this.scene.add(this.markerPlaceholderGroup);
    this.markerGroup = new THREE.Group(); this.scene.add(this.markerGroup);
    this.droneGroup = new THREE.Group(); this.scene.add(this.droneGroup);
    this.tetherGroup = new THREE.Group(); this.scene.add(this.tetherGroup);

    this.droneMeshes = {};
    this.tethers = {}; // dashed line from each drone's ground point up to its flying position
    this.markerMeshes = {};
    this.placeholderMeshes = {}; // sim-only "not found yet" ground-truth markers
    this._lastZonesCacheKey = null;
    this._lastPlannedPathsKey = null;

    window.addEventListener('resize', () => this.onResize(container));
  }

  onResize(container) {
    this.camera.aspect = container.clientWidth / container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  }

  setMap(mapInfo) {
    clearGroup(this.mapGroup);
    const boundary = mapInfo.boundary || [];
    if (boundary.length >= 3) {
      this.mapGroup.add(lineLoopFromPoints(boundary, 0x44cc44, 0.006));
      const shape = polygonShape(boundary);
      const geom = new THREE.ShapeGeometry(shape);
      geom.rotateX(Math.PI / 2);
      const ground = new THREE.Mesh(
        geom, new THREE.MeshBasicMaterial({ color: 0x222831, side: THREE.DoubleSide }));
      this.mapGroup.add(ground);

      let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
      boundary.forEach(([x, y]) => {
        minx = Math.min(minx, x); maxx = Math.max(maxx, x);
        miny = Math.min(miny, y); maxy = Math.max(maxy, y);
      });

      // Grid lines at each coverage cell boundary, so it's visible on the
      // ground which cell is which (matches the drones' actual coverage unit).
      const cell = mapInfo.coverage_line_spacing || 0.5;
      const gridMat = new THREE.LineBasicMaterial({ color: 0x384049 });
      const gridPts = [];
      for (let x = minx; x <= maxx + 1e-6; x += cell) {
        gridPts.push(w2t(x, miny, 0.002), w2t(x, maxy, 0.002));
      }
      for (let y = miny; y <= maxy + 1e-6; y += cell) {
        gridPts.push(w2t(minx, y, 0.002), w2t(maxx, y, 0.002));
      }
      const gridGeom = new THREE.BufferGeometry().setFromPoints(gridPts);
      this.mapGroup.add(new THREE.LineSegments(gridGeom, gridMat));

      const cx = (minx + maxx) / 2, cy = (miny + maxy) / 2;
      // Scene x is negated (see file header), so the arena center sits at
      // Three.js x = -cx; aim the camera there to keep it framed.
      this.controls.target.set(-cx, 0, cy);
      this.camera.position.set(-cx, Math.max(maxx - minx, maxy - miny) * 1.3 + 3, maxy + 6);
      this.controls.update();

      // Size the axis gizmo relative to the map instead of a fixed guess, so
      // it's readable whether the map is a 2m test rig or a 50m field.
      const axisLen = Math.max(0.5, Math.min(3, Math.max(maxx - minx, maxy - miny) * 0.2));
      clearGroup(this.worldAxesGroup);
      populateWorldAxes(this.worldAxesGroup, axisLen);
    }
    (mapInfo.dead_zones || []).forEach((pts) => {
      if (pts.length < 3) return;
      this.mapGroup.add(lineLoopFromPoints(pts, 0xcc4444, 0.009));
      const shape = polygonShape(pts);
      const geom = new THREE.ShapeGeometry(shape);
      geom.rotateX(Math.PI / 2);
      geom.translate(0, 0.008, 0);
      const mesh = new THREE.Mesh(
        geom, new THREE.MeshBasicMaterial({
          color: 0xcc4444, transparent: true, opacity: 0.35, side: THREE.DoubleSide }));
      this.mapGroup.add(mesh);
    });
  }

  // Ground-layer stacking order (low to high y-lift, i.e. bottom to top from
  // the top-down camera's point of view): ground(0) < grid(0.002) < zone
  // fill(0.004) < boundary outline(0.006) < dead-zone(0.008-0.009) < marker
  // placeholder(0.012) < planned/visited path(0.02, see setPaths). Path used
  // to sit *below* the zone fill, which visually washed it out -- it needs to
  // be the topmost flat layer since it's the thing you actually want to read.
  // `visitedByDrone`: droneId -> Set of "x.xxx,y.xxx" cell-center keys that
  // have actually been visited (see main()'s pollState, which derives this
  // from `paths` + the authoritative `progress.waypoint_index` -- NOT a
  // client-side nearest-point guess). Zone data essentially never changes
  // after the mission is planned, but the visited set changes every leg, so
  // the cache key covers both -- this used to rebuild every one of ~230
  // little cell squares on *every single poll* (300ms) forever, which is
  // what was slowly leaking WebGL memory until the tab crashed.
  setZones(zones, visitedByDrone) {
    visitedByDrone = visitedByDrone || {};
    const cacheKey = JSON.stringify(zones) + '|' + JSON.stringify(
      Object.fromEntries(Object.entries(visitedByDrone).map(([k, v]) => [k, Array.from(v).sort()])));
    if (cacheKey === this._lastZonesCacheKey) return;
    this._lastZonesCacheKey = cacheKey;

    clearGroup(this.zoneGroup);
    zones.forEach((zone) => {
      const color = new THREE.Color(zone.color || '#cccccc');
      const visited = visitedByDrone[zone.drone_id] || new Set();
      zone.polygons.forEach((pts) => {
        if (pts.length < 3) return;
        // Square corners are [min,min],[max,min],[max,max],[min,max] (see
        // control_node's _publish_plan) -- center is the midpoint of the
        // two opposite corners.
        const cx = (pts[0][0] + pts[2][0]) / 2;
        const cy = (pts[0][1] + pts[2][1]) / 2;
        const isVisited = visited.has(`${cx.toFixed(3)},${cy.toFixed(3)}`);
        const shape = polygonShape(pts);
        const geom = new THREE.ShapeGeometry(shape);
        geom.rotateX(Math.PI / 2);
        geom.translate(0, 0.004, 0);
        const mesh = new THREE.Mesh(
          geom, new THREE.MeshBasicMaterial({
            color, transparent: true, opacity: isVisited ? 0.55 : 0.14, side: THREE.DoubleSide }));
        this.zoneGroup.add(mesh);
      });
    });
  }

  // Planned path barely ever changes after the mission is planned (published
  // once, before flight), so it's cached and skipped when unchanged -- unlike
  // the visited tube (see setVisitedPaths), rebuilding this every poll for no
  // reason was a big chunk of the WebGL memory leak (see setZones).
  // Drawn gray/dashed and projected onto the *ground* (z ~ 0) even though the
  // drone actually flies it at cruise altitude, so the floor reads as a
  // top-down coverage map; see setDrones for the altitude tether line.
  setPlannedPaths(paths) {
    const cacheKey = JSON.stringify(paths);
    if (cacheKey === this._lastPlannedPathsKey) return;
    this._lastPlannedPathsKey = cacheKey;

    clearGroup(this.plannedPathGroup);
    Object.values(paths).forEach((wps) => {
      if (!wps || wps.length < 2) return;
      const full = wps.map(([x, y]) => w2t(x, y, 0.02));
      const geom = new THREE.BufferGeometry().setFromPoints(full);
      const line = new THREE.Line(geom, new THREE.LineDashedMaterial({
        color: 0x9aa0a6, dashSize: 0.15, gapSize: 0.1, opacity: 0.85, transparent: true }));
      line.computeLineDistances();
      this.plannedPathGroup.add(line);
    });
  }

  // `progress` is the authoritative per-drone {waypoint_index, total_waypoints}
  // published by control_node itself (the thing actually driving the FSM), not
  // a client-side guess. An earlier version guessed "how far along" a drone
  // was by finding whichever waypoint was spatially nearest to its current
  // position -- unreliable on a zig-zag lawnmower path, where many waypoints
  // on *different* rows can sit close to the current position without being
  // anywhere near "next", which is why cf2/cf3 showed as almost fully
  // "visited" right from the start of the mission.
  // This one genuinely needs a rebuild every poll (the tube grows as
  // waypoint_index advances), but it's only ~3 small tube meshes -- cheap,
  // and now properly disposed via clearGroup instead of leaking.
  setVisitedPaths(paths, progress) {
    clearGroup(this.visitedPathGroup);
    Object.entries(paths).forEach(([droneId, wps]) => {
      if (!wps || wps.length < 2) return;
      const visitedUpTo = (progress[droneId] && progress[droneId].waypoint_index) || 0;
      const visitedPts = wps.slice(0, Math.min(wps.length, visitedUpTo + 1))
        .map(([x, y]) => w2t(x, y, 0.02));
      // Dedup consecutive coincident points -- CatmullRomCurve3 chokes on a
      // zero-length segment (NaN tangents) which would make the tube vanish.
      const dedup = [];
      visitedPts.forEach((p) => {
        if (!dedup.length || p.distanceTo(dedup[dedup.length - 1]) > 1e-6) dedup.push(p);
      });
      if (dedup.length >= 2) {
        // LineBasicMaterial's `linewidth` is ignored on most GPUs/browsers (a
        // long-standing WebGL/ANGLE limitation) so a "thicker visited path"
        // has to actually be geometry, not a fatter line -- a thin white tube
        // mesh along the traveled points.
        const curve = new THREE.CatmullRomCurve3(dedup, false, 'catmullrom', 0);
        const tubeGeom = new THREE.TubeGeometry(
          curve, Math.max(1, dedup.length * 4), 0.035, 6, false);
        this.visitedPathGroup.add(
          new THREE.Mesh(tubeGeom, new THREE.MeshBasicMaterial({ color: 0xffffff })));
      }
    });
  }

  // Sim-only debug overlay: all ground-truth marker positions, dim/outlined
  // since they haven't actually been "found" by a drone's camera yet. Empty
  // on real hardware (gcs_node never gets true_markers_path there), so this
  // is a no-op and nothing pre-known ever shows up -- exactly as it should be.
  setAllMarkers(allMarkers) {
    clearGroup(this.markerPlaceholderGroup);
    this.placeholderMeshes = {};
    allMarkers.forEach((m) => {
      const circle = lineLoopFromPoints(circlePoints(m.x, m.y, 0.14), 0x888888, 0.012);
      const label = makeTextSprite('#' + m.id, '#888888');
      label.position.copy(w2t(m.x, m.y, (m.z || 0) + 0.25));
      this.markerPlaceholderGroup.add(circle);
      this.markerPlaceholderGroup.add(label);
      this.placeholderMeshes[m.id] = [circle, label];
    });
  }

  setMarkers(markers) {
    const seen = new Set();
    markers.forEach((m) => {
      seen.add(m.id);
      if (!this.markerMeshes[m.id]) {
        const geom = new THREE.SphereGeometry(0.08, 12, 12);
        const mat = new THREE.MeshBasicMaterial({ color: 0xf5c542 });
        const mesh = new THREE.Mesh(geom, mat);
        const label = makeTextSprite('#' + m.id, '#f5c542');
        label.position.set(0, 0.3, 0);
        mesh.add(label);
        this.markerGroup.add(mesh);
        this.markerMeshes[m.id] = mesh;
        // "before -> after found": once real detection arrives, the dim
        // ground-truth placeholder (if any) is replaced by the bright marker.
        const placeholder = this.placeholderMeshes[m.id];
        if (placeholder) placeholder.forEach((obj) => { obj.visible = false; });
      }
      this.markerMeshes[m.id].position.copy(w2t(m.x, m.y, m.z));
    });
    Object.keys(this.markerMeshes).forEach((id) => {
      if (!seen.has(Number(id))) {
        disposeObject3D(this.markerMeshes[id]);
        this.markerGroup.remove(this.markerMeshes[id]);
        delete this.markerMeshes[id];
      }
    });
  }

  setDrones(drones, zoneColors) {
    const seen = new Set();
    drones.forEach((d) => {
      seen.add(d.id);
      if (!this.droneMeshes[d.id]) {
        const color = new THREE.Color(zoneColors[d.id] || FALLBACK_COLORS[d.id] || '#ffffff');
        const geom = new THREE.SphereGeometry(0.12, 16, 16);
        // MeshBasicMaterial (unlit) so the drone stays clearly visible regardless
        // of scene lighting -- matches the marker spheres, which use the same
        // approach and were visible even when the (lit) drone sphere wasn't.
        const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ color }));
        const axes = new THREE.AxesHelper(0.3);
        axes.material.depthTest = false;
        // AxesHelper draws its arrows in plain Three.js local space (+x,+y,+z),
        // but w2t() negates world x when mapping into Three.js (see file
        // header) -- so at yaw=0 the unflipped red (+x) arrow would point
        // toward Three +x, i.e. world -x, backwards. Green (Y, altitude) and
        // blue (Z -> world y) need no flip since neither of those axes is
        // negated by w2t(). Scaling just the local x by -1 mirrors the red
        // arrow consistently at every yaw (scale is applied before the
        // mesh's own yaw rotation, so the flip survives rotation correctly).
        axes.scale.x = -1;
        mesh.add(axes);
        const label = makeTextSprite(d.id, '#' + color.getHexString());
        label.position.set(0, 0.3, 0);
        mesh.add(label);
        this.droneGroup.add(mesh);
        this.droneMeshes[d.id] = mesh;
      }
      const mesh = this.droneMeshes[d.id];
      mesh.position.copy(w2t(d.x, d.y, d.z));
      // yaw is rotation about the world's vertical (Z) axis, which maps to
      // Three.js's Y axis. w2t() maps world y->Three.js z (one handedness flip,
      // giving -yaw) and additionally negates world x (a second flip), so the
      // two cancel and the heading is +yaw here -- keeping the drone's heading
      // arrow pointing the correct physical way in the reflected scene.
      mesh.rotation.set(0, d.yaw, 0);

      // Dashed tether down to the ground point directly below the drone --
      // paths/visited cells are drawn on the ground (see setPaths), so this
      // keeps the actually-flying drone visually anchored to its cell.
      // Rebuilt every poll (position changes every time), so it must be
      // disposed before replacing -- this and setZones/setPaths rebuilding
      // without disposal is what was leaking WebGL memory until the tab
      // crashed after a while.
      if (this.tethers[d.id]) {
        disposeObject3D(this.tethers[d.id]);
        this.tetherGroup.remove(this.tethers[d.id]);
      }
      const color = new THREE.Color(zoneColors[d.id] || FALLBACK_COLORS[d.id] || '#ffffff');
      const tetherGeom = new THREE.BufferGeometry().setFromPoints(
        [w2t(d.x, d.y, 0.02), w2t(d.x, d.y, d.z)]);
      const tether = new THREE.Line(
        tetherGeom, new THREE.LineDashedMaterial({ color, dashSize: 0.08, gapSize: 0.06 }));
      tether.computeLineDistances();
      this.tetherGroup.add(tether);
      this.tethers[d.id] = tether;
    });
    Object.keys(this.droneMeshes).forEach((id) => {
      if (!seen.has(id)) {
        disposeObject3D(this.droneMeshes[id]);
        this.droneGroup.remove(this.droneMeshes[id]);
        delete this.droneMeshes[id];
      }
      if (!seen.has(id) && this.tethers[id]) {
        disposeObject3D(this.tethers[id]);
        this.tetherGroup.remove(this.tethers[id]);
        delete this.tethers[id];
      }
    });
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

// Rough 1S LiPo voltage->percent mapping (linear between typical full/empty
// resting voltages) -- good enough for an at-a-glance indicator, not a
// precise fuel gauge. See docs/RUNNING.md / crazyflies.yaml's
// voltage_warning/voltage_critical for the thresholds crazyswarm2 itself uses.
function voltageToPercent(voltage) {
  const FULL_V = 4.2, EMPTY_V = 3.3;
  const pct = ((voltage - EMPTY_V) / (FULL_V - EMPTY_V)) * 100;
  return Math.max(0, Math.min(100, Math.round(pct)));
}

function buildLegend(zones, linkStatus) {
  const el = document.getElementById('legend');
  linkStatus = linkStatus || {};
  el.innerHTML = zones.map((z) => {
    const status = linkStatus[z.drone_id];
    // No entry at all -> sim (sim_perception_node never publishes
    // /mission/link_status, see SharedState.link_status) -- skip the
    // battery UI entirely rather than show a permanently-unknown "--".
    // status present but battery_voltage still 0.0 -> real hardware that
    // just hasn't gotten its first /status message yet -- show "--" since
    // a reading should arrive soon.
    if (!status) {
      return `<div class="row">
        <span class="swatch" style="background:${z.color}"></span>
        <span class="drone-id">${z.drone_id}</span>
      </div>`;
    }
    const voltage = status.battery_voltage;
    const known = voltage > 0;
    const pct = known ? voltageToPercent(voltage) : 0;
    const level = pct >= 50 ? 'ok' : pct >= 20 ? 'low' : 'critical';
    return `<div class="row">
      <span class="swatch" style="background:${z.color}"></span>
      <span class="drone-id">${z.drone_id}</span>
      <div class="battery-track"><div class="battery-fill${known ? ' ' + level : ''}" style="width:${pct}%"></div></div>
      <span class="battery-label">${known ? `${pct}% (${voltage.toFixed(2)}V)` : '--'}</span>
    </div>`;
  }).join('') || '<div>zones not planned yet</div>';
}

function pollVideo(droneId) {
  const img = document.getElementById('frame-' + droneId);
  const noSignal = document.getElementById('no-signal-' + droneId);
  const tick = () => {
    fetch(`/api/frame/${droneId}?t=${Date.now()}`)
      .then((res) => {
        if (res.status === 204) {
          img.style.display = 'none'; noSignal.style.display = 'block';
          return null;
        }
        return res.blob();
      })
      .then((blob) => {
        if (!blob) return;
        img.style.display = 'block'; noSignal.style.display = 'none';
        const url = URL.createObjectURL(blob);
        const old = img.src;
        img.src = url;
        if (old && old.startsWith('blob:')) URL.revokeObjectURL(old);
      })
      .catch(() => {})
      .finally(() => setTimeout(tick, 200));
  };
  tick();
}

function main() {
  const scenePane = document.getElementById('scene-pane');
  const gcsScene = new GcsScene(scenePane);

  fetch('/api/map').then((r) => r.json()).then((map) => gcsScene.setMap(map));

  let totalMarkerCount = null; // null = unknown (real hardware / no ground truth)
  fetch('/api/all_markers').then((r) => r.json()).then((allMarkers) => {
    totalMarkerCount = allMarkers.length || null;
    gcsScene.setAllMarkers(allMarkers);
  });

  const startBtn = document.getElementById('start-btn');
  const killBtn = document.getElementById('kill-btn');
  const markerStatusEl = document.getElementById('marker-status');

  const videoPane = document.getElementById('video-pane');
  const videoToggle = document.getElementById('video-toggle');
  const resizeScene = () => gcsScene.onResize(scenePane);
  videoPane.addEventListener('transitionend', resizeScene);
  videoToggle.addEventListener('click', () => {
    const collapsed = videoPane.classList.toggle('collapsed');
    videoToggle.textContent = collapsed ? '◀' : '▶';
    resizeScene(); // immediate resize too, in case transitions are disabled
  });

  // Run each render step independently: one step throwing (bad/unexpected data
  // shape, etc.) must not silently prevent the *other* steps from running --
  // that's what was hiding the drone icons earlier (an earlier step's
  // exception meant setDrones() below it in the chain never ran, and the
  // blanket .catch(() => {}) swallowed the error with no console trace).
  function safeCall(label, fn) {
    try {
      fn();
    } catch (err) {
      console.error(`gcs render step "${label}" failed:`, err);
    }
  }

  // Covering elapsed-time timer is purely client-side: the browser just
  // remembers wall-clock time when it first sees mission_state flip to
  // COVERING (and when it flips away again), no backend change needed --
  // /mission/progress + /mission/state already carry everything else.
  let coveringStartMs = null;
  let coveringEndMs = null;
  const formatElapsed = (ms) => {
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  };

  let zoneColors = {};
  const pollState = () => {
    fetch('/api/state').then((r) => r.json()).then((state) => {
      safeCall('phase-text', () => {
        document.getElementById('phase-text').textContent = state.mission_state;
      });
      safeCall('coverage-status', () => {
        if (state.mission_state === 'COVERING') {
          if (coveringStartMs === null) { coveringStartMs = Date.now(); coveringEndMs = null; }
        } else if (coveringStartMs !== null && coveringEndMs === null) {
          coveringEndMs = Date.now();
        }
        if (state.mission_state === 'AWAITING_START') {
          coveringStartMs = null;
          coveringEndMs = null;
        }
        const elapsedEl = document.getElementById('covering-elapsed');
        elapsedEl.textContent = coveringStartMs === null
          ? '--:--' : formatElapsed((coveringEndMs || Date.now()) - coveringStartMs);

        const progressByDrone = state.progress || {};
        let visited = 0;
        let total = 0;
        Object.values(progressByDrone).forEach((p) => {
          visited += (p.visited_indices || []).length;
          total += p.total_waypoints || 0;
        });
        const pct = total > 0 ? Math.round((visited / total) * 100) : 0;
        document.getElementById('coverage-progress-fill').style.width = pct + '%';
        document.getElementById('coverage-progress-label').textContent = pct + '%';
      });
      safeCall('start-btn', () => {
        const canStart = state.mission_state === 'AWAITING_START';
        startBtn.disabled = !canStart;
        startBtn.textContent = canStart ? 'Start Mission' : `Mission ${state.mission_state}`;
      });
      safeCall('kill-btn', () => {
        const killed = state.mission_state === 'KILLED';
        killBtn.classList.toggle('killed', killed);
        killBtn.textContent = killed ? '■ KILLED' : '■ KILL';
      });
      zoneColors = {};
      (state.zones || []).forEach((z) => { zoneColors[z.drone_id] = z.color; });

      // Which cell-center points has each drone actually reached, per the
      // authoritative /mission/progress visited_indices -- feeds setZones'
      // "darken the cells actually visited" and is derived fresh every poll
      // (cheap: just string keys, no THREE.js objects involved).
      // Uses the exact visited-index set (not a waypoint_index prefix): the
      // backend checks every remaining cell independently each tick, so a
      // corner-cut miss on one cell no longer blocks every later cell from
      // being marked visited -- a prefix slice here would silently undo
      // that fix by re-imposing the "all-or-nothing up to N" assumption.
      const visitedByDrone = {};
      Object.entries(state.paths || {}).forEach(([droneId, wps]) => {
        const visitedIndices = (state.progress && state.progress[droneId] &&
          state.progress[droneId].visited_indices) || [];
        const set = new Set();
        visitedIndices.forEach((idx) => {
          if (idx >= 0 && idx < wps.length) {
            const [x, y] = wps[idx];
            set.add(`${x.toFixed(3)},${y.toFixed(3)}`);
          }
        });
        visitedByDrone[droneId] = set;
      });

      safeCall('link-badges', () => {
        const linkStatus = state.link_status || {};
        (window.DRONE_IDS || []).forEach((droneId) => {
          const status = linkStatus[droneId];
          const radioBadge = document.getElementById('radio-badge-' + droneId);
          const wifiBadge = document.getElementById('wifi-badge-' + droneId);
          if (!radioBadge || !wifiBadge) return;
          // No entry at all (sim, or real_perception_node hasn't published
          // yet) -> neutral "unknown" styling, not a false "disconnected".
          radioBadge.className = 'link-badge' + (status ? (status.radio_connected ? ' up' : ' down') : '');
          wifiBadge.className = 'link-badge' + (status ? (status.wifi_connected ? ' up' : ' down') : '');
        });
      });

      safeCall('setZones', () => gcsScene.setZones(state.zones || [], visitedByDrone));
      safeCall('buildLegend', () => buildLegend(state.zones || [], state.link_status || {}));
      safeCall('setPlannedPaths', () => gcsScene.setPlannedPaths(state.paths || {}));
      safeCall('setVisitedPaths', () => gcsScene.setVisitedPaths(state.paths || {}, state.progress || {}));
      safeCall('setMarkers', () => gcsScene.setMarkers(state.markers || []));
      safeCall('setDrones', () => gcsScene.setDrones(state.drones || [], zoneColors));
      safeCall('marker-status', () => {
        const found = (state.markers || []).slice().sort((a, b) => a.id - b.id);
        const countText = totalMarkerCount != null
          ? `${found.length}/${totalMarkerCount}` : `${found.length}`;
        const rows = found.length
          ? found.map((m) => `<div>#${m.id} (${m.x.toFixed(2)}, ${m.y.toFixed(2)})</div>`).join('')
          : '<div>(없음)</div>';
        markerStatusEl.innerHTML =
          `<div class="marker-count">발견한 조난자: ${countText}</div>` +
          `<div class="marker-ids">${rows}</div>`;
      });
    }).catch((err) => {
      console.error('gcs /api/state poll failed:', err);
    }).finally(() => setTimeout(pollState, 300));
  };
  pollState();

  (window.DRONE_IDS || []).forEach(pollVideo);

  startBtn.addEventListener('click', () => {
    startBtn.disabled = true;
    fetch('/api/mission/start', { method: 'POST' }).then((r) => r.json()).then((res) => {
      if (!res.success) {
        alert('mission start failed: ' + res.message);
        startBtn.disabled = false;
      }
      // On success the next /api/state poll will flip mission_state away from
      // AWAITING_START and keep the button disabled from there on.
    }).catch(() => { startBtn.disabled = false; });
  });

  // Emergency kill switch: cut all motors + halt the mission FSM. No confirm
  // dialog on purpose -- when a drone is about to hit a wall, the whole point
  // is an instant reaction. Bound to both the button and the 'k' key.
  function killMission() {
    fetch('/api/mission/kill', { method: 'POST' }).then((r) => r.json()).then((res) => {
      if (!res.success) alert('kill failed: ' + res.message);
    }).catch(() => alert('kill request failed'));
  }
  killBtn.addEventListener('click', killMission);
  window.addEventListener('keydown', (e) => {
    if (e.key !== 'k' && e.key !== 'K') return;
    const tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;  // don't fire while typing
    killMission();
  });

  function animate() {
    requestAnimationFrame(animate);
    gcsScene.render();
  }
  animate();
}

main();
