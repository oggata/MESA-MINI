# MESA — MultiEntity Simulation Architecture

**ペルソナを定義するだけで、AIエージェントの行動特性が変わる都市シミュレーション**

LLMが「20歳の探索好き」「効率重視のビジネスマン」などのペルソナ説明から報酬関数を自動生成し、**一人称視点のカメラ画像をCNNで処理するPPO強化学習**でエージェントごとの行動ポリシーを学習させます。学習済みモデルをブラウザで動かし、ペルソナの違いが行動の違いとして現れることをリアルタイムで観察できます。

---

## コンセプト

```
「人間の行動は、その人のペルソナ（性格・価値観）によって形成される」

ペルソナ A: 探索者   → マップを広く歩き回る
ペルソナ B: 慎重派   → 最短経路を繰り返す
ペルソナ C: 社交家   → 他エージェントに近づく
ペルソナ D: 効率主義 → 直進してゴールへ
ペルソナ E: 観光客   → 建物周辺をゆっくり巡る
```

同じ都市、同じルール、**違うペルソナ → 違う行動**

---

## アーキテクチャ全体図

```
┌─────────────────────────────────────────────────────────────┐
│                    MESA FPV Pipeline                        │
│                                                             │
│  ① ペルソナ定義                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  "20歳・探索好き・新しい場所に積極的に訪れる"          │   │
│  └────────────────────┬────────────────────────────────┘   │
│                       │ Claude API                          │
│                       ▼                                     │
│  ② 報酬関数の自動生成                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  explore_bonus:  4.5  ← 未訪問セルへの強いインセン  │   │
│  │  revisit_penalty: 1.8 ← 同じ場所に戻るペナルティ    │   │
│  │  goal_reward:   10.0  ← 目的地到達報酬              │   │
│  └────────────────────┬────────────────────────────────┘   │
│                       │                                     │
│                       ▼                                     │
│  ③ FPV画像生成 + CNN強化学習 (PPO)                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  PersonaVecEnv (N_ENVS=4096, GPU並列)               │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ... × 4096             │   │
│  │  │Env 0 │ │Env 1 │ │Env 2 │                        │   │
│  │  └──┬───┘ └──┬───┘ └──┬───┘                        │   │
│  │     └────────┴────────┘                             │   │
│  │              ▼  render_fp_batch() GPU一括レンダリング│   │
│  │     一人称視点画像 (64×64×3ch RGB)                   │   │
│  │              ▼                                      │   │
│  │     CNN PolicyNet (PPO Actor-Critic)                │   │
│  │     Conv×3 → FC256 → 行動: 前進/左回転/右回転        │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │ ONNX export (フラット形式 12288次元)     │
│                   ▼                                         │
│  ④ ペルソナ別ONNXポリシー                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  persona_A.onnx  persona_B.onnx  persona_C.onnx    │   │
│  │  persona_D.onnx  persona_E.onnx                    │   │
│  └────────────────┬────────────────────────────────────┘   │
│                   │ ブラウザで読み込み                       │
│                   ▼                                         │
│  ⑤ Three.js ビジュアライザー                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  30×30 有機的都市マップで5ペルソナが同時自律行動      │   │
│  │  学習と同じFPV画像をHTMLで生成 → ONNX推論            │   │
│  │  俯瞰 ↔ 一人称視点 リアルタイム切り替え              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## MESA とは

**MultiEntity Simulation Architecture** — 複数のエージェントが同一環境で自律行動するシミュレーション基盤。

| 概念 | 説明 |
|------|------|
| **Multi-Entity** | 複数のエージェントが同時に独立した行動ポリシーを持つ |
| **Persona-Driven** | エージェントの行動特性をペルソナ（人物像）で定義 |
| **FPV-CNN** | 一人称視点カメラ画像をCNNで処理して行動を決定 |
| **RL-Based** | 報酬関数によって行動を強化学習で獲得、ルールベースではない |
| **No BFS** | 経路探索アルゴリズムを使わず、画像から直接行動を学習 |

---

## セルタイプと行動ルール

```
マップ構成 (30×30 有機的グリッド)

  ██ ROAD     (白/グレー) — 通行可
  ██ BUILDING (赤)        — 目的地（到達で報酬）
  ██ TREE     (緑)        — 立入禁止
  ██ OTHER    (青)        — 立入禁止

マップ生成:
  格子道路を敷いた後、道路の一部をランダム削除
  BFSで全建物の接続性を保証 → 行き止まり・T字路が自然発生

行動ルール:
  ✓ 建物 → 道路 → 建物 の経路で移動
  ✓ 移動中は道路セルのみ通過
  ✗ 木・OTHERへの侵入はペナルティ
```

---

## エージェントの観測（FPV カメラ画像）

```
一人称視点 64×64×3ch RGB画像:

  エージェントの目線から前方を見た景色
    視野角: 60度
    レイ本数: 64本 (画像幅と同じ)
    最大検知距離: 8セル

  ┌────────────────────────────┐
  │  (空)  ██   (空)  ██  (空)│  ← 上半分: 空
  │        ██建物     ██木     │  ← 壁の柱 (距離に反比例した高さ)
  │  (地)  ██   (地)  ██  (地)│  ← 下半分: 地面
  └────────────────────────────┘

チャンネル別の色:
  ROAD     → RGB(176, 180, 172)  グレー
  BUILDING → RGB(196,  32,  32)  赤
  TREE     → RGB( 35, 104,  40)  緑
  OTHER    → RGB( 12,  30,  74)  濃い青

Python学習とHTML推論で完全に同じ画像生成ロジックを使用
→ 学習済みONNXをそのままブラウザで推論可能
```

---

## ファイル構成

```
mesa/
│
├── README.md
│
├── 📓 Step1: ペルソナ → 報酬パラメータ生成
│   └── step1_persona_reward_gen.ipynb   Claude APIでペルソナから報酬関数を自動生成
│
├── 📓 Step2: FPV-CNN 強化学習 (Colab GPU)
│   └── step2_persona_train.ipynb        FPV画像+PPOで5ペルソナを順次学習 → ONNX生成
│
├── 🌐 Step3: ビジュアライザー
│   └── step3_persona_city_sim.html      5ペルソナ同時動作 / 俯瞰・一人称切り替え
│
├── 📄 サンプルデータ
│   └── persona_rewards.json             API keyなしで試せるサンプル報酬パラメータ
│
├── 📁 data/                             (学習後に配置)
│   ├── persona_A.onnx
│   ├── persona_A_meta.json
│   └── ...
│
└── 🔧 修正パッチ (トラブル時)
    ├── patch_fp_cnn.ipynb               FPV-CNN版セル3〜9パッチ
    ├── patch_cell4_organic_map.ipynb    有機的マップ生成パッチ
    ├── fix_onnx_singlefile.ipynb        ONNX外部データ問題の修正
    └── gpu_diagnosis.ipynb              GPU使用率診断
```

---

## クイックスタート

### ペルソナをすぐ試す（ONNXなし）

```
step3_persona_city_sim.html をブラウザで開く
  ↓
Skip (Random Mode) を選択
  ↓
5ペルソナがランダムポリシーで動作開始
上部のペルソナカードをクリックしてFPビューを確認
```

### Default ONNXモード（学習済みモデルを使う）

```
step3_persona_city_sim.html と同じ階層に data/ フォルダを作成
  ↓
data/ に persona_A.onnx, persona_A_meta.json ... を配置
  ↓
python3 -m http.server 8000
  ↓
http://localhost:8000/step3_persona_city_sim.html を開く
  ↓
▶ Start Default ONNX を選択
```

### フルパイプラインを実行する

#### Step 1: ペルソナ定義 → 報酬パラメータ生成

```python
# step1_persona_reward_gen.ipynb のセル2にAPIキーを設定
ANTHROPIC_API_KEY = 'sk-ant-...'
```

ペルソナ定義（セル3）を自由に編集できます：

```python
PERSONAS = [
    {
        "id": "A",
        "name": "探索者タロウ",
        "description": "20歳。新しい場所に積極的に訪れる探索好き。"
                        "マップの隅々まで歩き回り、同じ道は通りたがらない。"
    },
    # ... B, C, D, E を定義
]
```

→ `persona_rewards.json` が生成されます

#### Step 2: FPV-CNN 強化学習（Colab T4 GPU）

```
Google Colab で step2_persona_train.ipynb を開く
  ↓
ランタイム → T4 GPU に変更
  ↓
セル1〜12 を順番に実行
  ↓
ペルソナA〜Eを順次学習（各約60M steps）
  ↓
マイドライブ / mesa_persona_onnx / に保存
  persona_A.onnx + persona_A_meta.json
  persona_B.onnx + persona_B_meta.json  ...
```

#### Step 3: ブラウザで動かす

```
step3_persona_city_sim.html をブラウザで開く
  ↓
各ペルソナの .onnx と _meta.json を同時に選択してロード
  ↓
Start Simulation
  ↓
🎲 New Map で有機的マップをランダム生成しても
  同じONNXで動作 (マップ汎化済みポリシー)
```

---

## ペルソナ設計の仕組み

LLMが以下のパラメータを自動設計します：

| パラメータ | 説明 | 探索者A | 慎重派B | 社交家C | 効率D | 観光客E |
|-----------|------|--------|--------|--------|-------|--------|
| `explore_bonus` | 未訪問セル到達ボーナス | **4.5** | 0.1 | 0.5 | 0.0 | 2.0 |
| `building_bonus` | 建物滞在ボーナス/step | 0.1 | 0.5 | **1.2** | 0.3 | **2.0** |
| `forward_bias` | 前進行動への傾き | 0.3 | 0.6 | 0.2 | **0.9** | 0.1 |
| `revisit_penalty` | 既訪問セルへのペナルティ | **1.8** | 0.0 | 0.0 | 0.2 | 0.3 |
| `goal_reward` | 目的地到達報酬 | 10.0 | 28.0 | 15.0 | **30.0** | 20.0 |
| `step_penalty` | 毎ステップのコスト | 0.05 | **0.4** | 0.08 | **0.5** | 0.02 |
| `social_bonus` | 他エージェント近接ボーナス | 0.0 | 0.0 | **3.0** | 0.0 | 0.5 |
| `stall_penalty` | 滞留ペナルティ | 1.5 | 2.0 | 0.3 | 2.0 | **0.1** |

---

## 技術スタック

| 技術 | 用途 |
|------|------|
| **PyTorch + PPO** | 強化学習アルゴリズム |
| **TorchScript JIT** | GPU並列FPVレンダリングの高速化 |
| **CNN (NatureA3Cスタイル)** | 64×64×3ch画像から行動を推定 |
| **ONNX** | 学習済みモデルのブラウザ推論用エクスポート |
| **onnxruntime-web** | ブラウザ上でのONNX推論（WebAssemblyバックエンド） |
| **Three.js** | 3Dビジュアライザー（俯瞰・一人称） |
| **Claude API** | ペルソナ説明 → 報酬パラメータの自動生成 |

---

## 学習環境

| 環境 | 設定 | バッチサイズ | 速度目安 |
|------|------|------------|---------|
| Google Colab T4 | N_ENVS=4096, CUDA | 524,288 | ~200M steps/時 |

---

## ビジュアライザーの操作方法

| 操作 | 内容 |
|------|------|
| ペルソナカードをクリック | そのペルソナのFPV画面に切り替え |
| **▶ Start Default ONNX** | `./data/` のONNXを自動読み込み |
| **🎲 New Map** | 有機的マップをランダム生成（ONNX再読み込み不要） |
| **俯瞰 / Orbit** | カメラモード切り替え |
| **FP View** | 右下に一人称カメラ（推論に使う実際の画像）を表示 |
| **Trail** | エージェントの軌跡表示 |
| **Speed ×1/2/4** | シミュレーション速度変更 |
| **Pause / Reset** | 一時停止 / リセット |

---

## Python ↔ HTML 完全一致

学習（Python）とブラウザ推論（HTML）で同じ画像生成ロジックを使用しています：

| 要素 | Python (`render_fp_batch`) | HTML (`renderFPImage`) |
|------|--------------------------|----------------------|
| 画像サイズ | 64×64×3ch | 64×64×3ch |
| 視野角 | FOV=60° | FP_FOV=60° |
| レイ本数 | N_RAYS=64 | IMG_W=64 |
| 建物色 | RGB(196,32,32) | RGB(196,32,32) |
| 配列形式 | CHW Tensor | CHW Float32Array |
| 推論入力 | flatten→(N, 12288) | (1, 12288) |

---

## 開発の流れ（MESA進化の歴史）

```
Conway's Game of Life
  ↓ セルオートマトンから自律エージェントへ
Grid-based Multi-Agent RL (10×10)
  ↓ グリッド座標 → 連続座標へ
Continuous Coordinate RL (30×30)
  ↓ 俯瞰観測 → レイキャストセンサーへ
First-Person Raycast RL (22次元ベクトル)
  ↓ ベクトル → カメラ画像へ
MESA FPV-CNN Persona Simulation  ← 現在
  ↓ (次のステップ)
DINOv2 Visual Encoder / Sim2Real / World Model Integration
```

---

## Sim2Real ロードマップ

```
現在: マップ数値 → FPV画像生成 → CNN → 行動
  ↓ エンコーダ部分を差し替えるだけ
将来: 実カメラ映像 → DINOv2 → 同じポリシーヘッド → 行動

  self.cnn = ...  # 現在のCNN
  ↓
  self.cnn = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
  # fc の入力次元を 384 に変えるだけ
```

---

## ライセンス

MIT License

---

## 関連プロジェクト

- [NVIDIA Cosmos WorldModel](https://www.nvidia.com/en-us/research/cosmos/)
- [LeRobot (HuggingFace)](https://github.com/huggingface/lerobot)
- [DINOv2 (Meta)](https://github.com/facebookresearch/dinov2)
