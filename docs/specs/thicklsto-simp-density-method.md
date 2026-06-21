# Spec: thickLSTO_simp.ipynb の密度法化

## 前提

1. 対象は `examples/thickLSTO_simp.ipynb` とする。
2. `examples/thickLSTO.ipynb` は参照用として変更しない。
3. 主設計変数を `phi` から密度変数 `rho` へ変更する。
4. 初期仕様では厚みPDE入力を `chi = rho_bar` とする。
5. 目的は、厚み感度が境界だけでなく構造内部にも分布するかを検証することである。

## Objective

`thickLSTO_simp.ipynb` を、レベルセット法ベースから密度法ベースへ変更する。

現状:

```text
phi -> Heaviside -> material_scale -> FEM
-> compliance sensitivity -> reaction-diffusion update of phi
```

変更後:

```text
rho -> density_filter -> rho_tilde -> tanh projection -> rho_bar
-> SIMP material scale -> FEM -> density update
```

厚み制約:

```text
rho_bar -> chi -> maximum thickness PDE -> G_thick
-> thickness sensitivity
```

`chi = H(phi)` ではなく `chi = rho_bar` を使い、Heaviside微分による境界局在化を外す。

## Tech Stack

- Python 3.12
- NumPy / SciPy / Matplotlib / Jupyter Notebook
- 既存ノートブック内のFEM、厚みPDE、Helmholtz実装

## Commands

環境作成と確認:

```bash
uv venv .venv --python 3.12
.venv/bin/python -V
```

ノートブック構文確認は、各コードセルを `ast.parse` して行う。

```bash
.venv/bin/python -c 'import ast,json; nb=json.load(open("examples/thickLSTO_simp.ipynb")); [ast.parse("".join(c.get("source", []))) for c in nb["cells"] if c.get("cell_type") == "code"]; print("syntax ok")'
```

短縮実行確認は、`n_steps = 1`、必要なら小さい `nx, ny` に変更して行う。

## Project Structure

- `examples/thickLSTO.ipynb`: 参照用レベルセット版
- `examples/thickLSTO_simp.ipynb`: 密度法化する対象
- `docs/specs/`: 本仕様書
- `docs/plans/`: タスク分解
- `tests/`: 既存テスト

## Design

### 設計変数

主設計変数は `rho` とし、範囲は `rho_min <= rho <= 1` とする。
初期値は既存の帯状構造を密度で表し、構造部を `rho = 1`、空洞部を `rho = rho_min` とする。

### 密度フィルター

設計変数 `rho` に密度フィルターを適用する。

```text
rho_tilde = density_filter @ rho
```

目的はチェッカーボード抑制と密度場の平滑化である。

### tanh投影

`rho_tilde` を `rho_bar` へ投影する。

$$
rho_{bar}
=
\frac{\tanh(\beta \eta)+\tanh(\beta(rho_{tilde}-\eta))}
{\tanh(\beta \eta)+\tanh(\beta(1-\eta))}
$$

初期値は `beta = 1.0`、`eta = 0.5` とする。`beta` は継続法で増やせるが、初期検証では小さめに保つ。

### 剛性

剛性はSIMP型にする。

$$
E(rho_{bar}) = E_{min} + (1 - E_{min}) rho_{bar}^{penal}
$$

初期値は `penal = 3.0`、`E_min = material.void_ratio` とする。

### 体積制約

体積制約は初期実装では `rho_bar` で評価する。

$$
G_{vol} = \int_{\Omega} rho_{bar} d\Omega - V_{max}
$$

### 厚みPDE

厚みPDEへの入力は `chi = rho_bar` とする。現在の `phi -> smooth_characteristic(phi) -> chi` 経路は、厚み制約の主経路から外す。

このとき $d chi / d rho_{bar} = 1$ なので、Heaviside微分による境界局在化を外せる。

### 厚み感度

厚み感度は以下のチェーンルールで設計変数へ戻す。

$$
\frac{dG_{thick}}{d rho}
=
\frac{dG_{thick}}{d chi}
\frac{d rho_{bar}}{d rho_{tilde}}
\frac{d rho_{tilde}}{d rho}
$$

ここで `chi = rho_bar` のため、$d chi / d rho_{bar} = 1$ である。

### 更新方法

初期実装では、標準的な密度法としてMMA更新を採用する。

```text
objective: compliance
constraint 1: volume
constraint 2: maximum thickness
design variable: rho
bounds: rho_min <= rho <= 1
```

既存のreaction-diffusion更新は `phi` の符号境界を前提にしているため、密度法の主更新には使わない。

## Code Style

密度法用の変数名は `rho` 系に揃える。主経路は `rho_tilde = density_filter @ rho`、`rho_bar = density_projection(rho_tilde, beta, eta)`、`chi = rho_bar` とする。`phi` は比較や参照用に限定する。

## Testing Strategy

初期検証は短縮条件で行う。

- `n_steps = 1`
- 必要に応じて `nx, ny` を小さくする
- `rho`, `rho_tilde`, `rho_bar`, `chi`, `h_s`, 厚み感度を可視化する

確認項目:

- `rho` が `[rho_min, 1]` に収まる
- `rho_bar`, `G_vol`, `G_thick` が有限値
- 厚みPDEの `states`, `thickness`, `evaluation` が有限値
- 厚み感度が構造内部にも分布するか確認できる
- コンプライアンス感度と厚み感度の桁が極端に乖離しない

## Boundaries

- Always: `examples/thickLSTO.ipynb` は変更しない。
- Always: 初期実装は `examples/thickLSTO_simp.ipynb` 内で完結させる。
- Always: 厚み制約は最適化感度へ入れる。
- Ask first: `moto/src` 側へ共通化する。
- Ask first: 厚みPDEの式自体を変更する。
- Never: 厚み制約を評価だけにして、最適化感度から黙って外さない。
- Never: `chi = H(phi)` と `chi = rho_bar` の結果を同じ意味として扱わない。

## Success Criteria

- `rho` が主設計変数として使われる。
- 厚みPDE入力が `chi = rho_bar` になる。
- `phi -> smooth_characteristic(phi)` が厚み制約の主経路から外れる。
- `rho_bar`, `chi`, `h_s`, 厚み感度を可視化できる。
- 少なくとも短縮条件で1ステップ実行できる。
- 厚み感度が境界だけでなく構造内部にも分布するか確認できる。

## Open Questions

- MMAをノートブック内へ新規実装するか、既存実装を流用するか。
- 厚み用 `beta` をコンプライアンス用 `beta` と分けるか。
- Helmholtz感度拡張を密度法でも残すか。
- 最終的な二値構造の厚み検証を別セルで行うか。
