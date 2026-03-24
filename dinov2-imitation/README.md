# MESA — MultiEntity Simulation Architecture

**ペルソナを定義するだけで、AIエージェントの行動特性が変わる都市シミュレーション**

LLMがペルソナ説明から報酬関数を自動生成し、FPV画像・DINOv2・PPO強化学習でエージェントごとのポリシーを学習。さらにブラウザ上でのキーボード操作デモを使ったACTスタイル模倣学習で、ペルソナの個性を精密に調整できます。

---

## コンセプト

```
「人間の行動は、その人のペルソナ（性格・価値観）によって形成される」

ペルソナ A: 探索者   → マップを広く歩き回る
ペルソナ B: 慎重派   → 最短経路を繰り返す
ペルソナ C: 社交家   → 他エージェントに近づく
ペルソナ D: 効率主義 → 直進してゴールへ
ペルソナ E: 観光客   → 建物周辺をゆっくり巡る

さらに:
  「お腹が空いた → 牛丼屋を探して入る」
  → DINOv2が建物を分類
  → セグメンテーションで前方の道路を認識
  → マップ配列に頼らず、カメラ画像だけで行動
```

---

## パイプライン全体図

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MESA Full Pipeline                            │
│                                                                     │
│  Step 1     ペルソナ定義 → Claude API → 報酬パラメータ               │
│                                                                     │
│  Step 1.5   建物分類ヘッド学習                                        │
│             店舗写真 → DINOv2 CLS → 8クラス分類器                    │
│                                                                     │
│  Step 1.6   セグメンテーションヘッド学習                              │
│             FPV画像(自動生成) → DINOv2 Patch → 5クラスセグ           │
│             「前方が道路か」を画像から判断                             │
│                                                                     │
│  Step 2     FPV + DINOv2 + PPO強化学習                               │
│             4096並列環境 × GPU一括レンダリング                        │
│             → persona_A〜E.onnx                                      │
│                                                                     │
│  Step 2.5   ACT模倣学習 Fine-tune  ← ブラウザのデモを使用            │
│             HTML でキーボード操作 → JSON収集                          │
│             → Transformer でK個の行動を一括予測                       │
│             → persona_A_act.onnx                                     │
│                                                                     │
│  Step 3     ブラウザ推論・可視化                                       │
│             Three.js + 実写テクスチャ + Transformers.js DINOv2       │
│             PPOポリシー or ACTポリシーを切り替えて比較可能            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ファイル構成

```
mesa/
│
├── README.md
│
├── 📓 メインパイプライン
│   │
│   ├── step1_persona_reward_gen.ipynb
│   │     ペルソナ説明 → Claude API → 報酬パラメータJSON
│   │     スキップ可 (persona_rewards.json のサンプルを使用)
│   │
│   ├── step1_5_building_classifier.ipynb
│   │     店舗写真 → DINOv2 CLS token → 分類ヘッド学習
│   │     → building_classifier.onnx  (8クラス)
│   │     スキップ可 (分類スコアがゼロ代替)
│   │
│   ├── step1_6_seg_head.ipynb
│   │     FPV画像自動生成 → DINOv2 Patch tokens → セグヘッド学習
│   │     → seg_head.onnx  (5クラス: sky/ground/open/building/tree)
│   │     スキップ可 (移動判定がマップ配列フォールバック)
│   │
│   ├── step2_persona_train.ipynb
│   │     FPV + DINOv2 + PPO学習 (T4 GPU必須)
│   │     → persona_A〜E.onnx
│   │
│   ├── step2_5_imitation_finetune.ipynb
│   │     ブラウザデモ JSON → ACT fine-tune
│   │     → persona_X_act.onnx
│   │     スキップ可 (PPOポリシーをそのまま使用)
│   │
│   └── step3_persona_city_sim.html
│         ブラウザビジュアライザー
│         キーボード操作デモの記録・エクスポート機能付き
│
├── 📄 サンプルデータ
│   └── persona_rewards.json
│         Step1をスキップしてすぐ試せるサンプル
│
└── 📁 data/  ← 学習後に作成してここにONNXを配置
    ├── persona_A.onnx + persona_A_meta.json       (PPOポリシー)
    ├── persona_B.onnx + persona_B_meta.json
    ├── persona_C.onnx + persona_C_meta.json
    ├── persona_D.onnx + persona_D_meta.json
    ├── persona_E.onnx + persona_E_meta.json
    ├── persona_A_act.onnx + persona_A_act_meta.json  (ACTポリシー)
    ├── building_classifier.onnx + building_classifier_meta.json
    └── seg_head.onnx + seg_head_meta.json
```

---

## 実行手順

### 事前準備

```bash
cd mesa/
python3 -m http.server 8000
# → http://localhost:8000/step3_persona_city_sim.html
```

---

### Step 1: ペルソナ定義 → 報酬パラメータ生成

**ファイル:** `step1_persona_reward_gen.ipynb`
**環境:** Colab (CPU可) / **スキップ可**

```python
ANTHROPIC_API_KEY = 'sk-ant-...'

PERSONAS = [
    { "id": "A", "name": "探索者タロウ",
      "description": "20歳。新しい場所に積極的に訪れる探索好き。" },
    # B, C, D, E...
]
```

**出力:** `persona_rewards.json`

---

### Step 1.5: 建物分類ヘッド学習

**ファイル:** `step1_5_building_classifier.ipynb`
**環境:** Colab T4推奨 / **スキップ可** / 所要時間: 約10分

```
セル3: Unsplashから各クラス5枚を自動ダウンロード
セル4: DINOv2で特徴抽出 → t-SNE可視化
セル5: 分類ヘッド学習 (200epoch・数秒)
セル6: 混同行列で精度確認
セル7: building_classifier.onnx をエクスポート
```

**精度目安:** 各5枚→70〜80% / 各20枚→85〜95%

**自分の写真を追加する場合:**
```
/content/drive/MyDrive/mesa_persona/building_images/gyudon/my_photo.jpg
→ セル4〜7を再実行するだけ
```

---

### Step 1.6: セグメンテーションヘッド学習

**ファイル:** `step1_6_seg_head.ipynb`
**環境:** Colab T4推奨 / **スキップ可** / 所要時間: 約20〜30分

このステップの意義:
```
従来: MAP[r][c] == ROAD でマップ配列を直接参照 (チート)
導入後: FPV画像 → DINOv2 Patch tokens → セグヘッド
        → 前方中央ピクセルが open(2) かどうかで移動可否を判断
        → 現実のロボットと同じ知覚プロセス
```

```
セル4: FPV画像 + セグマスクを自動生成 (アノテーション不要)
セル5: 5000枚のデータを数分で生成
セル6: DINOv2 Patch tokens + SegHead定義
セル7: 学習 (20epoch・mIoUで評価)
セル8: 予測結果を可視化 (FPV / 正解 / 予測を並べて表示)
セル9: seg_head.onnx をエクスポート
```

**セグメンテーションクラス:**

| ID | クラス | 説明 |
|----|--------|------|
| 0 | sky | 空 |
| 1 | ground | 地面 |
| **2** | **open** | **道路方向 ← 移動可否の判定基準** |
| 3 | building | 建物 |
| 4 | tree | 木 |

---

### Step 2: FPV + DINOv2 + PPO学習

**ファイル:** `step2_persona_train.ipynb`
**環境:** Colab **T4 GPU必須** / 所要時間: 1〜2時間/ペルソナ

**重要な設定 (セル3):**
```python
IMG_W   = 224          # DINOv2の入力サイズ
DINO_MODEL = 'dinov2_vits14'  # 384次元
N_ENVS  = 4096         # VRAMが足りなければ 1024 に
ROLLOUT = 128          # VRAMが足りなければ 64 に
```

**Domain Randomization (セル9):**
```python
MAP_RANDOMIZE_EVERY = 20  # 20 update ごとにマップ変更
                    = 0   # 固定マップ (動作確認用)
```

**セル7実行後に表示される状態:**
```
✓ DINOv2 loaded (全frozen)
✓ 建物分類ヘッド読み込み: 8クラス  ← Step1.5が必要
✓ seg_head読み込み: mIoU=0.XXX   ← Step1.6が必要
```

**出力:** `persona_A〜E.onnx` + `persona_X_meta.json`

---

### Step 2.5: ACT模倣学習 Fine-tune

**ファイル:** `step2_5_imitation_finetune.ipynb`
**環境:** Colab T4推奨 / **スキップ可** / 所要時間: 約30分

#### なぜACTか

```
BC (Behavior Cloning):
  観測(t) → 行動(t)  ← 1ステップ独立予測
  問題: ミスが積み重なると見たことない状態に陥る

ACT (Action Chunking with Transformers):
  観測(t) → [行動(t), ..., 行動(t+K-1)]  K=10個を一括予測
  → 滑らかで一貫性のある動き
  → LeRobotで使われているのと同じ方式
```

#### Step A: HTMLでデモを収集

```
1. python3 -m http.server 8000 でHTMLを開く
2. 記録したいペルソナのカードをクリック
3. [⏺ Record] ボタンをクリック
   → キーボード操作モードが有効になる
      W / ↑ : 前進
      A / ← : 左回転
      D / → : 右回転
4. 「このペルソナらしい動き」をキーボードで操作
5. [⏹ Stop] で1エピソード完了
6. 4〜5を 10〜30回繰り返す
7. [⬇ Export Demo] でJSONをダウンロード
8. Google Driveの mesa_persona/ に mesa_demo_latest.json として配置
```

**推奨エピソード数と品質:**

| エピソード数 | 期待精度 | 特徴 |
|------------|---------|------|
| 5〜10 | 60〜70% | スタイルの方向性だけ注入 |
| 10〜30 | 75〜85% | 個性がはっきり出る |
| 50〜 | 85%+ | 非常に精密なペルソナ表現 |

#### Step B: ColabでACT学習

```
セル3: JSON読み込み・エピソード確認
セル4: FPV画像 → DINOv2特徴抽出 → K=10のチャンクに分割
セル5: ACTPolicy (Transformer 2層) 定義
セル6: 学習 (100epoch)
セル7: persona_A_act.onnx をエクスポート
セル8: デモ vs ACT予測の行動分布を比較グラフで表示
```

**出力:** `persona_A_act.onnx` + `persona_A_act_meta.json`

---

### Step 3: ブラウザで動かす

**ファイル:** `step3_persona_city_sim.html`
**環境:** Chrome / Firefox (ローカルサーバー必須)

#### data/ フォルダの配置

```
mesa/
├── step3_persona_city_sim.html
└── data/
    ├── persona_A.onnx + persona_A_meta.json        ← PPO
    ├── persona_B〜E.onnx ...
    ├── persona_A_act.onnx + persona_A_act_meta.json ← ACT (任意)
    ├── building_classifier.onnx + _meta.json        ← Step1.5 (任意)
    └── seg_head.onnx + _meta.json                   ← Step1.6 (任意)
```

#### 起動と推論モード

```bash
python3 -m http.server 8000
# → http://localhost:8000/step3_persona_city_sim.html
```

| ボタン | 内容 |
|--------|------|
| **▶ Start Default ONNX** | `./data/` のONNXを自動読み込み (推奨) |
| **Start Simulation** | 手動でファイルを選択 |
| **Skip (Random Mode)** | ONNXなしでランダム動作 |

#### ステータス表示 (右下)

```
✓ DINOv2 Ready  |  ✓ SegHead Ready  ← 全機能有効
⚠ DINOv2 unavailable (CNN fallback)  ← フォールバック動作
```

DINOv2の初回ロードは30秒〜1分かかります (2回目以降はキャッシュ)。

---

## 操作方法

| 操作 | 内容 |
|------|------|
| ペルソナカードをクリック | そのペルソナのFPV画面に切り替え |
| **⏺ Record** | キーボード操作モード開始・デモ記録 |
| **⏹ Stop** | 記録停止・エピソード保存 |
| **⬇ Export Demo** | 記録したデモをJSONでダウンロード |
| **🎲 New Map** | 有機的マップをランダム生成 |
| **FP View** | 右下に一人称視点 (推論に使う実際の画像) を表示 |
| **俯瞰 / Orbit** | カメラモード切り替え |
| **Trail** | エージェントの軌跡表示 |
| **Speed ×1/2/4** | シミュレーション速度変更 |
| **Pause / Reset** | 一時停止 / リセット |

---

## 根幹技術の解説

### 1. DINOv2 — 2種類の出力の使い分け

```python
out = dino.forward_features(image)

# CLS token: 画像全体の要約 (384次元)
#   → 「この建物は何屋か」の分類に使用
cls = out['x_norm_clstoken']      # (N, 384)

# Patch tokens: 空間的な局所特徴 (256パッチ × 384次元)
#   → 「どのピクセルが道路か」のセグメンテーションに使用
patch = out['x_norm_patchtokens'] # (N, 256, 384)
```

DINOv2をこの方式で使う理由: ImageNetで事前学習済みのため、
少ないデータ (各クラス5枚) でファインチューンできます。

### 2. ACT — なぜチャンキングが有効か

```
通常のBC:
  観測(0) → 行動(0)
  観測(1) → 行動(1)  ← 各ステップ独立
  ...
  問題: 観測(1)が学習分布から少しずれると
        「見たことない状態」に陥りエラーが累積する

ACT (K=10):
  観測(0) → [行動(0)〜行動(9)] を一括予測して実行
  観測(10) → [行動(10)〜行動(19)] を一括予測して実行
  ...
  → チャンク内は一貫した意図に基づく動き
  → 分布シフトの頻度が 1/K に減る
```

```python
class ACTPolicy(nn.Module):
    def forward(self, obs_feat):  # (B, 392)
        # Transformer で K 個のクエリを処理
        queries = pos_embed + obs_embed    # (B, K, d_model)
        hidden  = transformer(queries)     # (B, K, d_model)
        return action_head(hidden)         # (B, K, ACT_DIM=3)
```

### 3. セグメンテーションによる移動判定

```
旧: MAP[r][c] == ROAD → マップ配列を直接参照 (シミュレーターのみ)

新: FPV画像 → DINOv2 Patch tokens
           → SegHead → (N, 5, 224, 224)
           → 前方中央ピクセル (112, 112) のクラスが open(2) か判断
           → True なら前進可能

→ 実カメラ映像に差し替えても同じポリシーで動く (Sim2Real)
```

### 4. PPO学習の工夫

```python
# GPU並列: 4096環境を同時実行
N_ENVS = 4096  # バッチサイズ = 4096 × 128 = 524,288

# JITコンパイル: レンダリングをGPU一括処理
@torch.jit.script
def render_fp_batch(xs, ys, ths, ...) -> torch.Tensor:
    # N=4096エージェント分のFPV画像を一括生成
    # returns: (N, 3, 224, 224)

# Domain Randomization: 20 update ごとにマップを変更
# → 特定マップへの過学習を防ぎ汎化性能を向上
```

### 5. Python ↔ HTML 完全一致

学習とブラウザ推論で同じ画像・同じ処理を使います。

| 要素 | Python | HTML |
|------|--------|------|
| 画像サイズ | 224×224×3ch | 224×224×3ch |
| 視野角 | FOV=60° | FP_FOV=60° |
| 建物色 | CELL_RGB | FP_CELL_RGB |
| 配列形式 | CHW Tensor | CHW Float32Array |
| DINOv2 | dinov2_vits14 | Xenova/dinov2-small |
| 移動判定 | seg_head.onnx | seg_head.onnx |

---

## 技術スタック

| 技術 | 用途 |
|------|------|
| **PyTorch + PPO** | 強化学習アルゴリズム |
| **DINOv2 ViT-S/14** | 視覚特徴抽出 (CLS + Patch tokens) |
| **ACT (Transformer)** | 模倣学習・アクションチャンキング |
| **TorchScript JIT** | GPU並列FPVレンダリングの高速化 |
| **ONNX** | 全モデルのブラウザ推論用エクスポート |
| **onnxruntime-web** | ブラウザ上のONNX推論 (WebAssembly) |
| **Transformers.js** | ブラウザ上のDINOv2推論 (HuggingFace) |
| **Three.js** | 3Dビジュアライザー |
| **TextureLoader** | 実写テクスチャの建物 (8種類) |
| **Claude API** | ペルソナ説明 → 報酬パラメータの自動生成 |

---

## 学習環境

| ステップ | 環境 | 所要時間 |
|---------|------|---------|
| Step 1 | Colab CPU可 | 約5分 |
| Step 1.5 | Colab T4推奨 | 約10分 |
| Step 1.6 | Colab T4推奨 | 約20〜30分 |
| Step 2 | Colab **T4必須** | 1〜2時間/ペルソナ |
| Step 2.5 | Colab T4推奨 | 約30分 |

**VRAMが足りない場合 (Step 2):**
```python
N_ENVS  = 1024  # 4096 → 1024
ROLLOUT = 64    # 128  → 64
```

---

## 開発ロードマップ

```
Conway's Game of Life
  ↓ セルオートマトン → 自律エージェント
Grid RL → Continuous RL → Raycast RL (22次元)
  ↓ ベクトル観測 → カメラ画像
FPV-CNN → FPV + DINOv2
  ↓ CNN → 事前学習済み視覚モデル
FPV + DINOv2 + 建物分類 + セグメンテーション
  ↓ PPOだけ → 模倣学習で個性を精密化
PPO + ACT模倣学習 Fine-tune  ← 現在
  ↓
Hierarchical Agent:
  LLM (計画) × DINOv2 (認識) × PPO+ACT (行動)
  ↓
Sim2Real: 実カメラ映像で同じポリシーが動く
```

---

## Sim2Realへの道

```python
# 現在: シミュレーション内のFPV画像
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

# 将来: 実カメラ映像でも同じモデルが動く
# → カメラをリサイズして渡すだけ
# seg_head / building_classifier / ACTPolicy はそのまま流用可能
```

---

## ライセンス

MIT License

---

## 関連プロジェクト

- [DINOv2 (Meta AI)](https://github.com/facebookresearch/dinov2)
- [ACT: Action Chunking with Transformers](https://github.com/tonyzhaozh/act)
- [LeRobot (HuggingFace)](https://github.com/huggingface/lerobot)
- [Transformers.js](https://huggingface.co/docs/transformers.js)
- [NVIDIA Cosmos WorldModel](https://www.nvidia.com/en-us/research/cosmos/)
