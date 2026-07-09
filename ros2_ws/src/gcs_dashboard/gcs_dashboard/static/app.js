// GCS 3D dashboard front end. Polls the Flask REST API (backed by ROS2 topics
// on the gcs_node side) and renders everything with Three.js. World frame is
// (x, y) on the ground plane + z = altitude; mapped into Three.js's right-handed
// Y-up space as (x, z_world, y_world) so "z up" in Three.js == altitude.

const FALLBACK_COLORS = { cf1: '#ff5555', cf2: '#55aaff', cf3: '#55dd77' };

function w2t(x, y, z) {
  return new THREE.Vector3(x, z || 0, y);
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
  const shape = new THREE.Shape();
  points.forEach(([x, y], i) => {
    if (i === 0) shape.moveTo(x, y); else shape.lineTo(x, y);
  });
  return shape;
}

function lineLoopFromPoints(points, color, yLift) {
  const pts = points.concat([points[0]]).map(([x, y]) => w2t(x, y, yLift || 0.01));
  const geom = new THREE.BufferGeometry().setFromPoints(pts);
  return new THREE.Line(geom, new THREE.LineBasicMaterial({ color }));
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

    this.mapGroup = new THREE.Group(); this.scene.add(this.mapGroup);
    this.zoneGroup = new THREE.Group(); this.scene.add(this.zoneGroup);
    this.pathGroup = new THREE.Group(); this.scene.add(this.pathGroup);
    this.markerPlaceholderGroup = new THREE.Group(); this.scene.add(this.markerPlaceholderGroup);
    this.markerGroup = new THREE.Group(); this.scene.add(this.markerGroup);
    this.droneGroup = new THREE.Group(); this.scene.add(this.droneGroup);

    this.droneMeshes = {};
    this.markerMeshes = {};
    this.placeholderMeshes = {}; // sim-only "not found yet" ground-truth markers

    window.addEventListener('resize', () => this.onResize(container));
  }

  onResize(container) {
    this.camera.aspect = container.clientWidth / container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  }

  setMap(mapInfo) {
    while (this.mapGroup.children.length) this.mapGroup.remove(this.mapGroup.children[0]);
    const boundary = mapInfo.boundary || [];
    if (boundary.length >= 3) {
      this.mapGroup.add(lineLoopFromPoints(boundary, 0x44cc44, 0.01));
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
      const cx = (minx + maxx) / 2, cy = (miny + maxy) / 2;
      this.controls.target.set(cx, 0, cy);
      this.camera.position.set(cx, Math.max(maxx - minx, maxy - miny) * 1.3 + 3, maxy + 6);
      this.controls.update();
    }
    (mapInfo.dead_zones || []).forEach((pts) => {
      if (pts.length < 3) return;
      this.mapGroup.add(lineLoopFromPoints(pts, 0xcc4444, 0.02));
      const shape = polygonShape(pts);
      const geom = new THREE.ShapeGeometry(shape);
      geom.rotateX(Math.PI / 2);
      geom.translate(0, 0.015, 0);
      const mesh = new THREE.Mesh(
        geom, new THREE.MeshBasicMaterial({
          color: 0xcc4444, transparent: true, opacity: 0.35, side: THREE.DoubleSide }));
      this.mapGroup.add(mesh);
    });
  }

  setZones(zones) {
    while (this.zoneGroup.children.length) this.zoneGroup.remove(this.zoneGroup.children[0]);
    zones.forEach((zone) => {
      const color = new THREE.Color(zone.color || '#cccccc');
      zone.polygons.forEach((pts) => {
        if (pts.length < 3) return;
        const shape = polygonShape(pts);
        const geom = new THREE.ShapeGeometry(shape);
        geom.rotateX(Math.PI / 2);
        geom.translate(0, 0.008, 0);
        const mesh = new THREE.Mesh(
          geom, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.16, side: THREE.DoubleSide }));
        this.zoneGroup.add(mesh);
      });
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
  setPaths(paths, progress) {
    while (this.pathGroup.children.length) this.pathGroup.remove(this.pathGroup.children[0]);
    Object.entries(paths).forEach(([droneId, wps]) => {
      if (!wps || wps.length < 2) return;
      const color = new THREE.Color(FALLBACK_COLORS[droneId] || '#cccccc');
      const visitedUpTo = (progress[droneId] && progress[droneId].waypoint_index) || 0;

      const full = wps.map(([x, y, z]) => w2t(x, y, z));
      const plannedGeom = new THREE.BufferGeometry().setFromPoints(full);
      this.pathGroup.add(new THREE.Line(
        plannedGeom, new THREE.LineDashedMaterial({ color, dashSize: 0.15, gapSize: 0.1, opacity: 0.5, transparent: true })).computeLineDistances());

      const visitedPts = wps.slice(0, Math.min(wps.length, visitedUpTo + 1))
        .map(([x, y, z]) => w2t(x, y, z));
      // Dedup consecutive coincident points -- CatmullRomCurve3 chokes on a
      // zero-length segment (NaN tangents) which would make the tube vanish.
      const dedup = [];
      visitedPts.forEach((p) => {
        if (!dedup.length || p.distanceTo(dedup[dedup.length - 1]) > 1e-6) dedup.push(p);
      });
      if (dedup.length >= 2) {
        // LineBasicMaterial's `linewidth` is ignored on most GPUs/browsers (a
        // long-standing WebGL/ANGLE limitation) so a "thicker visited path"
        // has to actually be geometry, not a fatter line -- a thin tube mesh
        // along the traveled points.
        const curve = new THREE.CatmullRomCurve3(dedup, false, 'catmullrom', 0);
        const tubeGeom = new THREE.TubeGeometry(
          curve, Math.max(1, dedup.length * 4), 0.035, 6, false);
        this.pathGroup.add(new THREE.Mesh(tubeGeom, new THREE.MeshBasicMaterial({ color })));
      }
    });
  }

  // Sim-only debug overlay: all ground-truth marker positions, dim/outlined
  // since they haven't actually been "found" by a drone's camera yet. Empty
  // on real hardware (gcs_node never gets true_markers_path there), so this
  // is a no-op and nothing pre-known ever shows up -- exactly as it should be.
  setAllMarkers(allMarkers) {
    while (this.markerPlaceholderGroup.children.length) {
      this.markerPlaceholderGroup.remove(this.markerPlaceholderGroup.children[0]);
    }
    this.placeholderMeshes = {};
    allMarkers.forEach((m) => {
      const circle = lineLoopFromPoints(circlePoints(m.x, m.y, 0.14), 0x888888, 0.006);
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
      // Three.js's Y axis under w2t() -- rotating the whole group about local
      // Y turns the axes helper's local X (red) into the drone's heading.
      mesh.rotation.set(0, -d.yaw, 0);
    });
    Object.keys(this.droneMeshes).forEach((id) => {
      if (!seen.has(id)) {
        this.droneGroup.remove(this.droneMeshes[id]);
        delete this.droneMeshes[id];
      }
    });
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function buildLegend(zones) {
  const el = document.getElementById('legend');
  el.innerHTML = zones.map((z) =>
    `<div><span class="swatch" style="background:${z.color}"></span>${z.drone_id}</div>`).join('')
    || '<div>zones not planned yet</div>';
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
  const markerStatusEl = document.getElementById('marker-status');

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

  let zoneColors = {};
  const pollState = () => {
    fetch('/api/state').then((r) => r.json()).then((state) => {
      safeCall('phase-text', () => {
        document.getElementById('phase-text').textContent = state.mission_state;
      });
      safeCall('start-btn', () => {
        const canStart = state.mission_state === 'AWAITING_START';
        startBtn.disabled = !canStart;
        startBtn.textContent = canStart ? 'Start Mission' : `Mission ${state.mission_state}`;
      });
      zoneColors = {};
      (state.zones || []).forEach((z) => { zoneColors[z.drone_id] = z.color; });
      safeCall('setZones', () => gcsScene.setZones(state.zones || []));
      safeCall('buildLegend', () => buildLegend(state.zones || []));
      safeCall('setPaths', () => gcsScene.setPaths(state.paths || {}, state.progress || {}));
      safeCall('setMarkers', () => gcsScene.setMarkers(state.markers || []));
      safeCall('setDrones', () => gcsScene.setDrones(state.drones || [], zoneColors));
      safeCall('marker-status', () => {
        const found = (state.markers || []).slice().sort((a, b) => a.id - b.id);
        const countText = totalMarkerCount != null
          ? `${found.length}/${totalMarkerCount}` : `${found.length}`;
        const idList = found.length
          ? found.map((m) => '#' + m.id).join(', ') : '(없음)';
        markerStatusEl.innerHTML =
          `<div class="marker-count">발견한 마커: ${countText}</div>` +
          `<div class="marker-ids">${idList}</div>`;
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

  function animate() {
    requestAnimationFrame(animate);
    gcsScene.render();
  }
  animate();
}

main();
