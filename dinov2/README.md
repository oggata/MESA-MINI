# MESA — MultiEntity Simulation Architecture

**ペルソナを定義するだけで、AIエージェントの行動特性が変わる都市シミュレーション**

LLMがペルソナ説明から報酬関数を自動生成し、**一人称視点カメラ画像 + DINOv2 + 建物分類**を組み合わせたPPO強化学習でエージェントごとの行動ポリシーを学習させます。学習済みモデルをブラウザで動かし、ペルソナの違いが行動の違いとして現れることをリアルタイムで観察できます。

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
  「お腹が空いた → 牛丼屋を探す」
  → DINOv2が視野内の建物を認識
  → 牛丼屋らしい建物に向かう
```

---

## アーキテクチャ全体図

```
┌──────────────────────────────────────────────────────────────────┐
│                      MESA FPV Pipeline                           │
│                                                                  │
│  ① ペルソナ定義 + 報酬設計                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  "20歳・探索好き"  →  Claude API  →  報酬パラメータJSON  │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│  ② 建物分類ヘッド学習 (Step 1.5)                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  店舗写真 (各クラス5〜50枚)                               │   │
│  │    ↓ DINOv2 (frozen) で特徴抽出                          │   │
│  │  384次元ベクトル → 分類ヘッド学習 → building_classifier.onnx │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│  ③ FPV-CNN + DINOv2 強化学習 (Step 2)                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PersonaVecEnv (N_ENVS=4096, GPU並列)                    │   │
│  │    ↓ render_fp_batch()  → FPV画像 224×224×3ch            │   │
│  │    ↓ DINOv2 (frozen)    → 384次元特徴                    │   │
│  │    ↓ 建物分類ヘッド      → 8クラス確率                    │   │
│  │    ↓ concat (392次元)   → FC → Actor/Critic              │   │
│  │  PPO学習  →  persona_X.onnx                              │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                        │
│  ④ ブラウザ推論 (Step 3)                                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Three.js + 実写テクスチャ建物 + 有機的マップ              │   │
│  │  Transformers.js DINOv2 → building_classifier.onnx       │   │
│  │  → persona_X.onnx → 5ペルソナ同時自律行動                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 観測パイプライン詳細

```
FPV画像 (224×224×3ch  RGB)
  視野角 60度 / レイ224本 / 最大検知距離 8セル

    ┌──────────────────────────────────┐
    │  空  ██建物██  空  ██木██  空   │ ← 実写テクスチャ付き
    │      ██    ██       ██           │   (牛丼屋/カフェ/etc.)
    │  地  ██    ██  地  ██  ██  地  │
    └──────────────────────────────────┘
         ↓
    DINOv2 ViT-S/14 (frozen, 384次元)
         ↓                    ↓
    視覚特徴           建物分類ヘッド
    (384次元)          (8クラス確率)
         └──────┬──────┘
               concat (392次元)
                 ↓
            FC PolicyNet
                 ↓
           前進 / 左回転 / 右回転
```

---

## 建物クラス (8種類)

| ID | クラス | 説明 | 3D表示 |
|----|--------|------|--------|
| 0 | gyudon | 牛丼屋・定食屋 | 実写テクスチャ 🥩 |
| 1 | ramen | ラーメン屋 | 実写テクスチャ 🍜 |
| 2 | bento | 弁当屋 | 実写テクスチャ 🍱 |
| 3 | cafe | カフェ | 実写テクスチャ ☕ |
| 4 | office | オフィスビル | 実写テクスチャ 🏢 |
| 5 | house | 住宅 | 実写テクスチャ 🏠 |
| 6 | conbini | コンビニ | 実写テクスチャ 🏪 |
| 7 | hospital | 病院 | 実写テクスチャ 🏥 |

---

## ファイル構成

```
mesa/
│
├── README.md
│
├── 📓 メインパイプライン
│   ├── step1_persona_reward_gen.ipynb      Step1: ペルソナ → 報酬パラメータ生成
│   ├── step1_5_building_classifier.ipynb  Step1.5: DINOv2 建物分類ヘッド学習
│   ├── step2_persona_train.ipynb           Step2: FPV+DINOv2 PPO学習 → ONNX生成
│   └── step3_persona_city_sim.html         Step3: ブラウザビジュアライザー
│
├── 📄 サンプルデータ
│   └── persona_rewards.json               API keyなしで試せるサンプル
│
├── 📁 data/                               (学習後に配置)
│   ├── persona_A.onnx
│   ├── persona_A_meta.json
│   ├── building_classifier.onnx
│   ├── building_classifier_meta.json
│   └── ...
│
└── 🔧 パッチ・デバッグ用
    ├── patch_fp_cnn.ipynb                 FPV-CNN版セル3〜9
    ├── patch_dinov2.ipynb                 DINOv2版セル3・7パッチ
    ├── patch_cell4_organic_map.ipynb      有機的マップ生成
    ├── patch_cell9_train_persona.ipynb    Domain Randomization学習ループ
    ├── fix_onnx_singlefile.ipynb          ONNX外部データ問題の修正
    └── gpu_diagnosis.ipynb               GPU使用率診断
```

---

## 実行手順

### 事前準備

```bash
# ローカルでHTMLを開く場合 (file://では動かない)
python3 -m http.server 8000
# → http://localhost:8000/step3_persona_city_sim.html
```

---

### Step 1: ペルソナ定義 → 報酬パラメータ生成

**ファイル:** `step1_persona_reward_gen.ipynb`  
**環境:** Google Colab (CPU可)  
**所要時間:** 約5分

```python
# セル2: APIキーを設定
ANTHROPIC_API_KEY = 'sk-ant-...'
```

```python
# セル3: ペルソナを自由に編集
PERSONAS = [
    {
        "id": "A",
        "name": "探索者タロウ",
        "description": "20歳。新しい場所に積極的に訪れる。同じ道は通りたがらない。"
    },
    # B, C, D, E を定義...
]
```

**出力:** `persona_rewards.json`

APIキーがない場合はリポジトリ内の `persona_rewards.json` (サンプル) をそのまま使用できます。

---

### Step 1.5: 建物分類ヘッド学習

**ファイル:** `step1_5_building_classifier.ipynb`  
**環境:** Google Colab (T4 GPU 推奨、CPU可)  
**所要時間:** 約10分

```
セル1〜2: インストール・設定
セル3:    Unsplashから各クラス5枚を自動ダウンロード
          (自分の写真をDriveに置いてもOK)
セル4:    DINOv2で特徴抽出 → t-SNEで可視化
セル5:    分類ヘッド学習 (200epoch・数秒)
セル6:    混同行列で精度確認
セル7:    building_classifier.onnx をエクスポート
```

**精度の目安:**
| 各クラスの画像数 | 期待精度 |
|----------------|---------|
| 5枚 | 70〜80% |
| 20枚 | 85〜95% |
| 50枚 | 95%+ |

**自分の写真を追加する場合:**
```
/content/drive/MyDrive/mesa_persona/building_images/
  gyudon/
    my_photo_01.jpg   ← ここに追加
    my_photo_02.jpg
```
追加後にセル4〜7を再実行するだけです。

**出力:**
```
mesa_persona_onnx/
  building_classifier.onnx
  building_classifier_meta.json
```

> **Step 1.5 はスキップ可能です。** スキップした場合は建物分類スコアがゼロベクトルで代替され、Step 2の学習は動作します。

---

### Step 2: FPV + DINOv2 PPO学習

**ファイル:** `step2_persona_train.ipynb`  
**環境:** Google Colab **T4 GPU 必須**  
**所要時間:** 約1〜2時間/ペルソナ（×5ペルソナ）

#### 実行手順

```
1. Google Colab で step2_persona_train.ipynb を開く
2. ランタイム → T4 GPU に変更
3. セル1〜12 を順番に実行
```

#### 重要なセル

**セル3 — 定数設定:**
```python
IMG_W        = 224     # DINOv2の入力サイズ
DINO_MODEL   = 'dinov2_vits14'  # 384次元
N_ENVS       = 4096   # 並列環境数 (VRAMが足りなければ1024に)
STEPS_PER_PERSONA = 60_000_000
```

**セル7 — PolicyNet (自動的に分類ヘッドを読み込む):**
```python
# building_classifier.onnx が存在すれば自動読み込み
# なければゼロベクトルで代替 (Step 1.5スキップ時)

# 観測構造:
# DINOv2特徴 (384) + 建物分類スコア (8) = 392次元 → FC → 行動
```

**セル9 — Domain Randomization:**
```python
MAP_RANDOMIZE_EVERY = 20  # 20 update ごとにマップを変更
                    = 0   # 固定マップで学習したい場合
```

**セル11 — 学習対象ペルソナの選択:**
```python
TRAIN_PERSONAS = list(all_rewards.keys())  # 全ペルソナ
# TRAIN_PERSONAS = ['A']  # 1つだけテストする場合
```

#### VRAMが足りない場合

```python
# セル3で変更
N_ENVS  = 1024  # 4096 → 1024
ROLLOUT = 64    # 128  → 64
```

**出力:**
```
mesa_persona_onnx/
  persona_A.onnx + persona_A_meta.json
  persona_B.onnx + persona_B_meta.json
  persona_C.onnx + persona_C_meta.json
  persona_D.onnx + persona_D_meta.json
  persona_E.onnx + persona_E_meta.json
```

---

### Step 3: ブラウザで動かす

**ファイル:** `step3_persona_city_sim.html`  
**環境:** Chrome / Firefox (ローカルサーバー必須)

#### 起動方法

```bash
# HTMLと同じ階層で実行
python3 -m http.server 8000

# ブラウザで開く
http://localhost:8000/step3_persona_city_sim.html
```

#### データの配置

```bash
mesa/
├── step3_persona_city_sim.html
└── data/
    ├── persona_A.onnx
    ├── persona_A_meta.json
    ├── persona_B.onnx
    ├── persona_B_meta.json
    ├── ...
    ├── building_classifier.onnx
    └── building_classifier_meta.json
```

#### 起動モードの選択

| ボタン | 内容 |
|--------|------|
| **▶ Start Default ONNX** | `./data/` のONNXを自動読み込み（推奨） |
| **Start Simulation** | 手動でONNXファイルを選択 |
| **Skip (Random Mode)** | ONNXなしでランダム動作（動作確認用） |

#### DINOv2 の状態確認

```
右下に表示:
  ✓ DINOv2 Ready    → Transformers.js でDINOv2が起動済み
  ⚠ DINOv2 unavailable (CNN fallback) → フォールバックで動作中
```

DINOv2 の初回ロードには30秒〜1分かかります（2回目以降はキャッシュされます）。

---

## 操作方法

| 操作 | 内容 |
|------|------|
| ペルソナカードをクリック | そのペルソナのFPV画面に切り替え |
| **🎲 New Map** | 有機的マップをランダム生成（ONNX再読み込み不要） |
| **FP View** | 右下に一人称視点（推論に使う実際の画像）を表示 |
| **俯瞰 / Orbit** | カメラモード切り替え |
| **Trail** | エージェントの軌跡表示 |
| **Speed ×1/2/4** | シミュレーション速度変更 |
| **Pause / Reset** | 一時停止 / リセット |

---

## 技術スタック

| 技術 | 用途 |
|------|------|
| **PyTorch + PPO** | 強化学習アルゴリズム |
| **DINOv2 ViT-S/14** | 視覚特徴抽出（frozen） |
| **TorchScript JIT** | GPU並列FPVレンダリングの高速化 |
| **ONNX** | モデルのブラウザ推論用エクスポート |
| **onnxruntime-web** | ブラウザ上のONNX推論（WebAssembly） |
| **Transformers.js** | ブラウザ上のDINOv2推論（HuggingFace） |
| **Three.js** | 3Dビジュアライザー |
| **CanvasTexture / TextureLoader** | 実写テクスチャの建物 |
| **Claude API** | ペルソナ説明 → 報酬パラメータの自動生成 |

---

## 学習環境

| 環境 | 設定 | バッチサイズ |
|------|------|------------|
| Google Colab T4 | N_ENVS=4096 | 524,288 |
| Google Colab T4 (省メモリ) | N_ENVS=1024, ROLLOUT=64 | 65,536 |

---

## Python ↔ HTML 完全一致

学習とブラウザ推論で同じ画像を使います：

| 要素 | Python | HTML |
|------|--------|------|
| 画像サイズ | 224×224×3ch | 224×224×3ch |
| 視野角 | FOV=60° | FP_FOV=60° |
| レイ本数 | N_RAYS=224 | IMG_W=224 |
| 建物色 | CELL_RGB | FP_CELL_RGB |
| 配列形式 | CHW Tensor | CHW Float32Array |
| DINOv2モデル | dinov2_vits14 | Xenova/dinov2-small |

---

## ペルソナ設計の仕組み

LLMが以下のパラメータを自動設計します：

| パラメータ | 説明 | 探索A | 慎重B | 社交C | 効率D | 観光E |
|-----------|------|-------|-------|-------|-------|-------|
| `explore_bonus` | 未訪問セルボーナス | **4.5** | 0.1 | 0.5 | 0.0 | 2.0 |
| `building_bonus` | 建物滞在ボーナス | 0.1 | 0.5 | **1.2** | 0.3 | **2.0** |
| `forward_bias` | 前進への傾き | 0.3 | 0.6 | 0.2 | **0.9** | 0.1 |
| `revisit_penalty` | 再訪ペナルティ | **1.8** | 0.0 | 0.0 | 0.2 | 0.3 |
| `goal_reward` | 目的地到達報酬 | 10.0 | 28.0 | 15.0 | **30.0** | 20.0 |
| `step_penalty` | 毎ステップコスト | 0.05 | **0.4** | 0.08 | **0.5** | 0.02 |
| `social_bonus` | 他者近接ボーナス | 0.0 | 0.0 | **3.0** | 0.0 | 0.5 |
| `stall_penalty` | 滞留ペナルティ | 1.5 | 2.0 | 0.3 | 2.0 | **0.1** |

---

## 開発ロードマップ

```
Conway's Game of Life
  ↓ セルオートマトン → 自律エージェント
Grid RL → Continuous RL → Raycast RL
  ↓ ベクトル観測 → カメラ画像へ
FPV-CNN Persona Simulation
  ↓ CNN → DINOv2 大規模事前学習モデルへ
DINOv2 + 建物分類 Persona Simulation  ← 現在
  ↓
Hierarchical Agent:
  LLM (計画) × DINOv2 (認識) × PPO (行動)
  「お腹が空いた → 牛丼屋を探して入る」
  ↓
Sim2Real:
  実カメラ映像 → 同じDINOv2 → 実ロボット行動
```

---

## Sim2Real ロードマップ

```python
# 現在: シミュレーション内のFPV画像
self.vision = dinov2_vits14   # frozen

# 将来: 実カメラ映像でも同じモデルが動く
# エンコーダを差し替えるだけ
self.vision = dinov2_vitb14   # より高精度版
# または
self.vision = custom_finetuned_dino  # 実店舗写真でfine-tune済み
```

---

## ライセンス

MIT License

---

## 関連プロジェクト

- [DINOv2 (Meta AI)](https://github.com/facebookresearch/dinov2)
- [Transformers.js (HuggingFace)](https://huggingface.co/docs/transformers.js)
- [NVIDIA Cosmos WorldModel](https://www.nvidia.com/en-us/research/cosmos/)
- [LeRobot (HuggingFace)](https://github.com/huggingface/lerobot)
