# -*- coding: utf-8 -*-
"""Check what Roboflow workspace and datasets are available with this API key."""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from roboflow import Roboflow
rf = Roboflow(api_key="RU6y0QNjPWtW7fgAcMCs")

# Check workspace
try:
    ws = rf.workspace()
    print(f"Workspace: {ws.name} (id: {ws.id})")
    print(f"Projects: {[p.name for p in ws.projects()]}")
except Exception as e:
    print(f"Workspace error: {e}")

# Try universe search for parking datasets
print("\nSearching Roboflow Universe for parking datasets...")
try:
    # Try public datasets that might work
    datasets = [
        ("keremberke", "aerial-parking-lot-detection", 1),
        ("new-workspace-yqfqv", "parking-lot-lbxre", 1),
        ("roboflow-universe-projects", "parking-lot-object-detection", 1),
        ("object-detection-un0ux", "parking-lot-glhak", 2),
    ]
    for ws_id, proj_id, ver in datasets:
        try:
            project = rf.workspace(ws_id).project(proj_id)
            version = project.version(ver)
            print(f"  [OK] Found: {ws_id}/{proj_id} v{ver}")
            print(f"       Classes: {version}")
            break
        except Exception as ex:
            print(f"  [SKIP] {ws_id}/{proj_id}: {str(ex)[:80]}")
except Exception as e:
    print(f"Search error: {e}")
