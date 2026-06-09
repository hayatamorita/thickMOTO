# Implementation Plan: MPM 最大厚み制約

## Overview

`examples/1_validation.ipynb` の密度法 MPM トポロジー最適化へ、`examples/thickLSTO.ipynb` の PDE 型最大厚み制約を MMA の第 2 制約として追加する。最も高リスクな離散随伴勾配を先に検証し、有限差分と一致してから Notebook と MMA へ接続する。

## Architecture Decisions

- 厚み PDE は `moto/src/thickness_constraint.py` に集約し、SciPy で JAX/JIT の外から評価する。
- PDE 座標は `L_ref=0.09` で無次元化し、領域を `2 x 1` とする。
- `rho_bar=0.5` を境界とし、C2 Heaviside を `w=0.1` で使う。
- 材料点から節点へは正規化 GIMP 投影、感度はその転置で戻す。
- `projection_volume_mode` は `reference` を既定値、`current` を実験的モードとする。
- `j_max=0.01`、終了後の可行性判定は両制約 `<=1e-4` とする。
- 数値テストは標準ライブラリの `unittest` から開始する。

## Dependency Graph

```text
Contracts
  ├── C2 Heaviside + dimensionless mesh precompute
  └── GIMP projection + transpose
        └── Thickness PDE value
              └── Discrete adjoint gradient
                    └── End-to-end finite-difference gate
                          ├── Notebook diagnostics
                          └── MMA integration
                                └── Optimization validation
```

## Phase 1: Numerical Foundation

### Task 1: Thickness module contracts and test harness

**Description:** パラメータ、戻り値、投影モード、前計算データを定義し、Notebook から独立したテスト構成を作る。

**Acceptance criteria:**
- [ ] `h0=0.1`, `j_max=0.01`, `L_ref=0.09`, `w=0.1`, `projection_volume_mode="reference"` を表現できる。
- [ ] 不正な正値パラメータと未対応モードを例外で拒否し、`tests/` から import できる。

**Verification:** `.venv/bin/python -m unittest discover -s tests -v`

**Dependencies:** None  
**Files:** `moto/src/thickness_constraint.py`, `tests/test_thickness_constraint.py`  
**Scope:** S

### Task 2: C2 Heaviside and dimensionless mesh precompute

**Description:** C2 Heaviside、その導関数、既存 `VoxelMesh` から無次元 Rect4 前計算データを構築する。

**Acceptance criteria:**
- [ ] `phi<=-w`, `phi=0`, `phi>=w` で `0`, `0.5`, `1` を返し、導関数が境界で連続する。
- [ ] `0.18 x 0.09` の座標を `2 x 1` に変換し、節点重みの総和が面積 `2.0` と一致する。

**Verification:** Heaviside 導関数を中心有限差分と比較する。  
**Dependencies:** Task 1  
**Files:** `moto/src/thickness_constraint.py`, `tests/test_thickness_constraint.py`  
**Scope:** M

### Task 3: Normalized GIMP projection and transpose

**Description:** `rho_bar -> phi_rho -> phi_node -> chi_node` の投影と、その転置作用を実装する。

**Acceptance criteria:**
- [ ] 一様な材料点場を一様な節点場へ投影し、分母ゼロ節点を空洞・ゼロ勾配にする。
- [ ] `reference/current` が指定された体積、座標、domain length で map を構築する。

**Verification:** 内積テスト `<P a,b>=<a,P^T b>` と reference 不変性を確認する。  
**Dependencies:** Task 1  
**Files:** `moto/src/thickness_constraint.py`, `tests/test_thickness_projection.py`  
**Scope:** M

## Checkpoint: Foundation

- [ ] 全 unit test が成功する。
- [ ] Heaviside の有限差分と GIMP 転置の内積テストが成功する。
- [ ] Notebook と MMA にはまだ接続していない。

## Phase 2: Thickness Constraint and Gradient

### Task 4: Geometric-feature PDE and constraint value

**Description:** state PDE、発散投影、厚み場、違反密度、`G_thick` を既存メッシュへ移植する。

**Acceptance criteria:**
- [ ] `chi_node` から state、厚み場、違反場、`G_thick` を返す。
- [ ] `G_thick=integral(chi*violation)-j_max` を無次元領域で評価し、solve 失敗を検出する。

**Verification:** 全空洞、一様構造、帯状構造が有限値となり、`thickLSTO` 相当値と比較できる。  
**Dependencies:** Tasks 2, 3  
**Files:** `moto/src/thickness_constraint.py`, `tests/test_thickness_pde.py`  
**Scope:** M

### Task 5: Canonical geometry regression tests

**Description:** 無次元 `2 x 1` 領域の既知形状で、厚み値と外周境界条件を回帰テストする。

**Acceptance criteria:**
- [ ] 薄帯より厚帯の厚み場と `G_thick` が大きくなる。
- [ ] 内部平行移動では大きく変化せず、外周接触時の影響を記録できる。

**Verification:** `.venv/bin/python -m unittest tests.test_thickness_pde -v`  
**Dependencies:** Task 4  
**Files:** `tests/test_thickness_pde.py`  
**Scope:** S

### Task 6: Discrete adjoint nodal gradient

**Description:** 離散残差から真の `dG_thick/dchi_node` を随伴法で実装する。shape-removal direction は流用しない。

**Acceptance criteria:**
- [ ] state と同じ離散行列を使い、直接項と state 依存項を含める。
- [ ] 形状 `(num_nodes,)` の有限な節点勾配を返す。

**Verification:** 遷移領域の複数節点で中心有限差分との値と符号を比較する。  
**Dependencies:** Task 4  
**Files:** `moto/src/thickness_constraint.py`, `tests/test_thickness_gradient.py`  
**Scope:** M

### Task 7: End-to-end design-variable gradient

**Description:** Heaviside 微分、GIMP 転置、threshold filter 微分、density filter 転置を通して `dG_thick/dx` を得る。

**Acceptance criteria:**
- [ ] `G_thick` と形状 `(num_design_var,)` の勾配を同時に返す。
- [ ] 分母ゼロ節点と Heaviside 遷移区間外から勾配が流れない。

**Verification:** 空洞、境界、構造領域の設計変数で中心有限差分と比較する。  
**Dependencies:** Tasks 3, 6  
**Files:** `moto/src/thickness_constraint.py`, `tests/test_thickness_gradient.py`  
**Scope:** M

## Checkpoint: Gradient Gate

- [ ] `G_thick` の回帰テストが成功する。
- [ ] 節点勾配と設計変数勾配が中心有限差分と一致する。
- [ ] 符号、形状、非有限値チェックが完了する。
- [ ] 通過するまで MMA へ接続しない。

## Phase 3: Notebook and MMA Integration

### Task 8: Evaluation-only Notebook integration

**Description:** 既存 `rho_phys` に対して厚み制約を評価し、最適化を変更せず診断できるセルを追加する。

**Acceptance criteria:**
- [ ] `G_thick` を単独評価し、`chi_node`、厚み場、違反場を可視化できる。
- [ ] 既存コンプライアンス最適化の計算経路を変更しない。

**Verification:** `.venv` で評価セルを実行し、全出力が有限値であることを確認する。  
**Dependencies:** Task 7  
**Files:** `examples/1_validation.ipynb`  
**Scope:** S

### Task 9: MMA second-constraint integration

**Description:** JAX/JIT の体積制約と JIT 外の SciPy 厚み制約を MMA ループで結合する。

**Acceptance criteria:**
- [ ] `num_cons=2`, `constr.shape==(2,1)`, `grad_cons.shape==(2,num_design_var)` となる。
- [ ] `thickness_cons` を履歴へ追加し、`reference` を既定値として最適化できる。

**Verification:** 1反復を実行し、形状・有限値を assertion で確認する。  
**Dependencies:** Tasks 7, 8  
**Files:** `examples/1_validation.ipynb`, 必要なら厚みモジュール  
**Scope:** M

### Task 10: Short optimization validation

**Description:** `max_iter=5` から `10` で方向性、安定性、追加計算時間を検証する。

**Acceptance criteria:**
- [ ] `G_thick` が低下するか可行域を維持し、全履歴が有限値となる。
- [ ] 違反領域に対応する設計変更を可視化で確認できる。

**Verification:** 反復履歴、最終可視化、1反復あたりの追加時間を保存する。  
**Dependencies:** Task 9  
**Files:** `examples/1_validation.ipynb`  
**Scope:** S

### Task 11: Full-run verification and documentation

**Description:** 収束設定で実行し、可行性、パラメータ、既知の制約を文書化する。

**Acceptance criteria:**
- [ ] `G_vol<=1e-4` かつ `G_thick<=1e-4` の成否を明示する。
- [ ] `reference/current` の挙動と `current` の感度制限を記録する。

**Verification:** 全 unit test と Notebook 実行を行い、`git diff` をレビューする。  
**Dependencies:** Task 10  
**Files:** Notebook、仕様書、必要な検証文書  
**Scope:** S

## Checkpoint: Complete

- [ ] 投影、PDE、随伴、チェーンルールのテストが成功する。
- [ ] 短時間・全体最適化で `G_thick` の履歴を確認できる。
- [ ] 最終可行性を `1e-4` 基準で判定できる。
- [ ] 実装と仕様書が一致する。

## Parallelization Opportunities

- Task 2 と Task 3 は Task 1 完了後に並行できる。
- Task 5 は Task 4 完了後、Task 6 と並行できる。
- Task 6, 7, 9 は同じ勾配契約に依存するため順次実施する。

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 離散随伴が有限差分と不一致 | High | MMA前に Gradient Gate を設ける |
| 外周ゼロ境界が厚み場を歪める | High | 薄帯・厚帯・平行移動を先行検証する |
| JAX tracer と SciPy solve が混在 | High | JIT 外へ分離し NumPy 配列で契約する |
| GIMP 投影の分母ゼロ | Medium | 空洞値とゼロ勾配を単体テストする |
| `current` の感度が不完全 | Medium | 実験的機能とし既定は `reference` にする |
| PDE の反復コストが大きい | Medium | 前計算再利用と短時間計測を行う |
| 2制約が競合する | Medium | 履歴を分離し move limit を段階調整する |

## Validation Results

- 15 unit tests が成功し、Heaviside、GIMP 転置、節点随伴、設計変数勾配が有限差分と一致した。
- 実メッシュでは `G_thick=0.21039`、勾配形状 `(1, 11024)`、節点場形状 `(3321,)` を確認した。
- 5反復で `G_thick` は `0.21039 -> -0.00624` へ低下し、`G_vol` も可行だった。
- `max_iter=200` の全実行と `current` モードの連成感度は未検証である。
