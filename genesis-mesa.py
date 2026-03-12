"""
Genesis Human Agent Training Script
=====================================
Genesis のビジュアライザーで学習の様子を表示しながら学習し、
ONNX形式でポリシーをエクスポートします。

インストール:
  pip install genesis-world torch onnx onnxruntime numpy

実行:
  python genesis_train.py

出力:
  policy.onnx        ← Three.js (city_sim.html) で読み込む
  policy_meta.json   ← 状態空間のメタ情報
"""

import numpy as np
import torch
import torch.nn as nn
import json

# デバイス選択:
# GenesisはMPS(Apple GPU)非対応のためCPUで動作するが、
# PyTorchの学習部分はMPSで高速化できる
def _select_device():
    if torch.cuda.is_available():
        print("[Device] CUDA GPU detected → using cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[Device] Apple MPS detected → using mps for training")
        return torch.device("mps")
    else:
        print("[Device] Using CPU")
        return torch.device("cpu")

TRAIN_DEVICE = _select_device()  # 学習用 (MPS/CUDA/CPU)
GENESIS_DEVICE = torch.device("cpu")  # Genesis連携は常にCPU

# ============================================================
# Genesis インポート
# ============================================================
try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    print("[WARNING] genesis not found. Running in mock mode.")

# ============================================================
# 環境設定
# ============================================================
GRID_SIZE  = 10    # 10x10 グリッド都市
N_AGENTS   = 6     # エージェント数
MAX_STEPS  = 200   # 1エピソードの最大ステップ
N_EPISODES = 50000   # 学習エピソード数
CELL       = 1.0   # 1セル = 1.0 meter

BUILDINGS = [
    (1,1,1.2),(1,3,0.9),(3,1,1.5),(3,7,1.0),
    (5,3,1.8),(5,7,0.7),(7,1,1.3),(7,5,1.6),
    (8,8,0.8),(2,6,1.1),(6,2,2.0),(4,5,1.4),
]

AGENT_COLORS = [
    (1.0,0.3,0.3,1.0),
    (0.3,0.8,1.0,1.0),
    (0.3,1.0,0.3,1.0),
    (1.0,1.0,0.3,1.0),
    (1.0,0.5,0.0,1.0),
    (0.8,0.3,1.0,1.0),
]

BUILDING_COLORS = [
    (0.6,0.5,0.4,1.0),(0.4,0.5,0.6,1.0),(0.5,0.6,0.4,1.0),
    (0.7,0.4,0.3,1.0),(0.3,0.4,0.7,1.0),(0.5,0.4,0.7,1.0),
    (0.6,0.6,0.3,1.0),(0.3,0.6,0.6,1.0),(0.7,0.3,0.5,1.0),
    (0.4,0.7,0.4,1.0),(0.5,0.5,0.5,1.0),(0.6,0.4,0.5,1.0),
]

# ============================================================
# Genesis シーン構築
# ============================================================
def build_genesis_scene():
    gs.init(precision="32", logging_level="warning", backend=gs.cpu)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.0, -3.0, 16.0),
            camera_lookat=(5.0, 5.0, 0.0),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=True,
        show_FPS=True,
    )

    # 地面
    scene.add_entity(gs.morphs.Plane())

    # 建物
    for i, (bx, by, bh) in enumerate(BUILDINGS):
        color = BUILDING_COLORS[i % len(BUILDING_COLORS)]
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(bx * CELL + 0.4, by * CELL + 0.4, bh / 2),
                size=(0.8, 0.8, bh),
            ),
            surface=gs.surfaces.Default(color=color),
        )

    # 道路グリッドライン
    for i in range(GRID_SIZE + 1):
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(GRID_SIZE * CELL / 2, i * CELL, 0.002),
                size=(GRID_SIZE * CELL, 0.04, 0.004),
            ),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7, 1.0)),
        )
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(i * CELL, GRID_SIZE * CELL / 2, 0.002),
                size=(0.04, GRID_SIZE * CELL, 0.004),
            ),
            surface=gs.surfaces.Default(color=(0.7, 0.7, 0.7, 1.0)),
        )

    # エージェント
    agents = []
    for i in range(N_AGENTS):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        a = scene.add_entity(
            material=gs.materials.Rigid(rho=500.0, friction=0.5),
            morph=gs.morphs.Box(
                pos=(float(i) * CELL + 0.5, 0.5, 0.25),
                size=(0.3, 0.3, 0.5),
            ),
            surface=gs.surfaces.Default(color=color),
        )
        agents.append(a)

    scene.build()
    return scene, agents


# ============================================================
# ポリシーネットワーク
# ============================================================
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 都市シミュレーター
# ============================================================
MOVES = [(0,1),(0,-1),(-1,0),(1,0),(0,0)]

class CitySimulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.positions  = np.random.randint(0, GRID_SIZE, size=(N_AGENTS, 2)).astype(float)
        self.visited    = [set() for _ in range(N_AGENTS)]
        self.meetings   = np.zeros(N_AGENTS)
        self.step_count = 0
        for i in range(N_AGENTS):
            self.visited[i].add(tuple(self.positions[i].astype(int)))
        return self._obs(0)

    def _obs(self, aid):
        pos     = self.positions[aid]
        others  = np.delete(self.positions, aid, axis=0)
        diffs   = others - pos
        nearest = diffs[np.argmin(np.linalg.norm(diffs, axis=1))]
        obs = np.array([
            pos[0] / GRID_SIZE,
            pos[1] / GRID_SIZE,
            len(self.visited[aid]) / (GRID_SIZE * GRID_SIZE),
            min(self.meetings[aid] / 20.0, 1.0),
            nearest[0] / GRID_SIZE,
            nearest[1] / GRID_SIZE,
        ], dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def step(self, aid, action):
        dx, dy  = MOVES[action]
        new_pos = np.clip(self.positions[aid] + [dx, dy], 0, GRID_SIZE - 1)
        self.positions[aid] = new_pos

        for i in range(N_AGENTS):
            if i != aid:
                rdx, rdy = MOVES[np.random.randint(5)]
                self.positions[i] = np.clip(self.positions[i] + [rdx, rdy], 0, GRID_SIZE - 1)

        self.step_count += 1

        reward = 0.0
        cell = tuple(new_pos.astype(int))
        if cell not in self.visited[aid]:
            reward += 2.0
            self.visited[aid].add(cell)

        others = np.delete(self.positions, aid, axis=0)
        dists  = np.linalg.norm(others - new_pos, axis=1)
        n_meet = int(np.sum(dists < 1.5))
        reward += float(n_meet)
        self.meetings[aid] += n_meet

        if action == 4:
            reward -= 0.2

        done = self.step_count >= MAX_STEPS
        return self._obs(aid), reward, done


# ============================================================
# Genesis エージェント位置更新
# ============================================================
def update_genesis_agents(agents, sim):
    for i, agent in enumerate(agents):
        px = float(sim.positions[i][0]) * CELL + 0.15
        py = float(sim.positions[i][1]) * CELL + 0.15
        agent.set_pos(torch.tensor([[px, py, 0.25]], device='cpu'))


# ============================================================
# 学習ループ (REINFORCE)
# ============================================================
def train():
    print("=" * 60)
    print("Genesis Human Agent Training  —  REINFORCE")
    print("=" * 60)

    policy    = PolicyNet().to(TRAIN_DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    sim       = CitySimulator()

    scene, genesis_agents = None, None
    if GENESIS_AVAILABLE:
        print("[Genesis] Building scene...")
        scene, genesis_agents = build_genesis_scene()
        print("[Genesis] Scene ready. Training starts...\n")
    else:
        print("[Mock] Training without visualizer.\n")

    episode_rewards = []

    for ep in range(N_EPISODES):
        obs       = sim.reset()
        log_probs = []
        rewards   = []

        for _ in range(MAX_STEPS):
            logits = policy(torch.FloatTensor(obs).unsqueeze(0).to(TRAIN_DEVICE))
            probs  = torch.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            obs, reward, done = sim.step(0, action.item())
            rewards.append(reward)

            if scene is not None:
                try:
                    update_genesis_agents(genesis_agents, sim)
                    scene.step()
                except Exception:
                    pass

            if done:
                break

        # REINFORCE 更新
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        R = torch.tensor(returns, dtype=torch.float32, device=TRAIN_DEVICE)
        R = (R - R.mean()) / (R.std() + 1e-8)

        loss = -(torch.stack(log_probs) * R).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

        if ep % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"Ep {ep:4d} | AvgReward {avg:6.1f} | "
                  f"Visited {len(sim.visited[0]):2d}/{GRID_SIZE**2} | "
                  f"Meetings {sim.meetings[0]:.0f}")

    print("\n[Done] Training complete.")
    return policy


# ============================================================
# ONNX エクスポート
# ============================================================
def export_onnx(policy, path="policy.onnx"):
    policy.eval()
    policy = policy.to('cpu')  # export前にCPUへ
    dummy = torch.zeros(1, 6, device='cpu')
    torch.onnx.export(
        policy, dummy, path,
        input_names=["state"],
        output_names=["logits"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=11,
    )
    print(f"[ONNX] Saved: {path}")

    meta = {
        "input_size": 6,
        "output_size": 5,
        "actions": ["up", "down", "left", "right", "stay"],
        "obs_keys": [
            "pos_x/GRID", "pos_y/GRID", "visited_ratio",
            "meeting_rate", "nearest_dx/GRID", "nearest_dy/GRID",
        ],
        "grid_size": GRID_SIZE,
        "n_agents": N_AGENTS,
        "cell_size": CELL,
    }
    meta_path = path.replace(".onnx", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[JSON] Saved: {meta_path}")


# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    policy = train()
    export_onnx(policy, "policy.onnx")
    print("\n→ Open city_sim.html in your browser and load policy.onnx !")
