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
    this.markerGroup = new THREE.Group(); this.scene.add(this.markerGroup);
    this.droneGroup = new THREE.Group(); this.scene.add(this.droneGroup);

    this.droneMeshes = {};
    this.markerMeshes = {};

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
      geom.rotateX(-Math.PI / 2);
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
      geom.rotateX(-Math.PI / 2);
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
        geom.rotateX(-Math.PI / 2);
        geom.translate(0, 0.008, 0);
        const mesh = new THREE.Mesh(
          geom, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.16, side: THREE.DoubleSide }));
        this.zoneGroup.add(mesh);
      });
    });
  }

  setPaths(paths, drones, zoneColors) {
    while (this.pathGroup.children.length) this.pathGroup.remove(this.pathGroup.children[0]);
    Object.entries(paths).forEach(([droneId, wps]) => {
      if (!wps || wps.length < 2) return;
      const color = new THREE.Color(zoneColors[droneId] || FALLBACK_COLORS[droneId] || '#cccccc');
      const drone = drones.find((d) => d.id === droneId);
      let visitedUpTo = 0;
      if (drone) {
        let bestDist = Infinity;
        wps.forEach((wp, idx) => {
          const d = Math.hypot(wp[0] - drone.x, wp[1] - drone.y);
          if (d < bestDist) { bestDist = d; visitedUpTo = idx; }
        });
      }
      const full = wps.map(([x, y, z]) => w2t(x, y, z));
      const plannedGeom = new THREE.BufferGeometry().setFromPoints(full);
      this.pathGroup.add(new THREE.Line(
        plannedGeom, new THREE.LineDashedMaterial({ color, dashSize: 0.15, gapSize: 0.1, opacity: 0.5, transparent: true })).computeLineDistances());

      const visited = wps.slice(0, visitedUpTo + 1).map(([x, y, z]) => w2t(x, y, z));
      if (visited.length >= 2) {
        const visitedGeom = new THREE.BufferGeometry().setFromPoints(visited);
        const visitedLine = new THREE.Line(visitedGeom, new THREE.LineBasicMaterial({ color, linewidth: 3 }));
        this.pathGroup.add(visitedLine);
      }
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
        const geom = new THREE.ConeGeometry(0.12, 0.28, 8);
        const mesh = new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ color }));
        const label = makeTextSprite(d.id, '#' + color.getHexString());
        label.position.set(0, 0.35, 0);
        mesh.add(label);
        this.droneGroup.add(mesh);
        this.droneMeshes[d.id] = mesh;
      }
      const mesh = this.droneMeshes[d.id];
      mesh.position.copy(w2t(d.x, d.y, d.z));
      mesh.rotation.set(Math.PI / 2, 0, -d.yaw);
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

  let zoneColors = {};
  const pollState = () => {
    fetch('/api/state').then((r) => r.json()).then((state) => {
      document.getElementById('phase-text').textContent = state.mission_state;
      zoneColors = {};
      state.zones.forEach((z) => { zoneColors[z.drone_id] = z.color; });
      gcsScene.setZones(state.zones);
      buildLegend(state.zones);
      gcsScene.setPaths(state.paths, state.drones, zoneColors);
      gcsScene.setMarkers(state.markers);
      gcsScene.setDrones(state.drones, zoneColors);
    }).catch(() => {}).finally(() => setTimeout(pollState, 300));
  };
  pollState();

  (window.DRONE_IDS || []).forEach(pollVideo);

  document.getElementById('start-btn').addEventListener('click', () => {
    fetch('/api/mission/start', { method: 'POST' }).then((r) => r.json()).then((res) => {
      if (!res.success) alert('mission start failed: ' + res.message);
    });
  });

  function animate() {
    requestAnimationFrame(animate);
    gcsScene.render();
  }
  animate();
}

main();
