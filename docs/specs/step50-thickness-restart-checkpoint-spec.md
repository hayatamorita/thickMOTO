# Spec: Step 50 Checkpoint Restart for Thickness Constraint Test

## Assumptions
- 今回の実装対象ノートブックは `examples/1_validation_alt.ipynb` とする。
- 既存の `examples/1_validation.ipynb` と `examples/1_validation_BC.ipynb` は、このタスクでは直接編集しない。
- 50 step時点の形状は、密度画像ではなく `MMAState` を保存して復元する。
- 厚み制約ON後のテストでは、厚み違反量がstepごとに下がることを合格条件に含める。

## Objective
`examples/1_validation_alt.ipynb` に、50 step時点の最適化状態を保存し、その状態をロードして厚み制約ONで再開できる仕組みを入れる。

目的は、初期状態から厚み制約を入れたときの不安定性と切り分けて、既に構造が出た50 step形状から厚み制約が正常に働くかを検証すること。

## Target Files
- Edit target: `examples/1_validation_alt.ipynb`
- Test target: `tests/test_thickness_restart_from_checkpoint.py`
- Checkpoint output: `examples/checkpoints/1_validation_alt_step50.npz`

## Non-Targets
- `examples/1_validation.ipynb` の挙動変更はしない。
- `examples/1_validation_BC.ipynb` の案B/案Cロジックは変更しない。
- 50 stepの重い計算を通常の単体テスト内で毎回実行しない。
- 密度画像だけから最適化を再開する実装にはしない。

## Reference Thickness Logic
厚み制約の評価ロジックは `examples/thickLSTO.ipynb` を参照する。

参照する主なセル:

- cell 5: `characteristic()`, `heaviside_material_smooth()`, `heaviside_volume()`, `clip_phi()`
- cell 11: `MaximumThicknessParams`, `analyze_maximum_thickness()`
- cell 13: `step_min_compliance()` 内の厚み感度の合成
- cell 17: 厚み場、制約値、感度、`S_comp`, `S_thick`, `cos` の可視化

`thickLSTO.ipynb` での厚み評価の流れ:

1. `smooth_characteristic(phi, characteristic_width)` で $\chi$ を作る。
2. $\chi$ を使って幾何学的特徴PDEを解き、方向別状態量 `states` を得る。
3. `states` の発散から厚み場 $h_s$ を計算する。
4. ランプ関数で厚み違反密度 `evaluation` を作る。
5. $\chi \cdot evaluation$ を積分し、`j_max` を引いて厚み制約値を得る。
6. 随伴方程式から `shape_derivative` を計算する。
7. `shape_derivative` をHelmholtz拡張して `analysis.sensitivity` を得る。
8. 最適化には `lambda_thickness * analysis.sensitivity` を厚み感度として渡す。

厚み場は次で評価する。

$$
h_s = \frac{2}{\sqrt{a}\max(\nabla \cdot s, \epsilon_{\mathrm{div}})}
$$

厚み違反密度は次で評価する。

$$
q = 1 - \frac{1}{2}h_0\sqrt{a}\nabla \cdot s
$$

$$
R(q) = \frac{1}{2}\left(q + \sqrt{q^2 + \epsilon_{\mathrm{ramp}}}\right)
$$

厚み制約値は次で評価する。

$$
G_{\mathrm{thick}} = \int \chi R(q)\,d\Omega - j_{\max}
$$

`1_validation_alt.ipynb` ではMPM/MMAを使うため、`thickLSTO.ipynb` の反応拡散更新をそのまま移植しない。参照するのは、厚みPDE、$h_s$、ランプ評価、随伴感度、Helmholtz拡張、可視化指標のロジックとする。

## Checkpoint Contents
保存する主データは `MMAState.to_array()` とする。

最低限保存するもの:

- `mma_state_array`
- `num_design_var`
- `epoch`
- `max_vol_frac`
- `constraint_mode`
- `thickness_start_epoch`

確認用に保存してよいもの:

- `rho_phys`
- `rho_bar`
- `G_thick`
- `G_vol`

再開時は `MMAState.from_array(mma_state_array, num_design_var)` で復元する。

## Restart Behavior
`optimize_design()` は、初期化済みの `MMAState` を受け取れるようにする。

期待する呼び出し形:

```python
mma_state, mp_final, rho_phys, thickness_result, convg_history = optimize_design(
    initial_mma_state=checkpoint_mma_state,
    constraint_mode="volume_and_thickness",
    thickness_start_epoch=50,
    n_steps=5,
)
```

`initial_mma_state` が渡された場合は、内部で新しい一様な `design_var` から `init_mma()` し直さない。

`mma_state.epoch` が50で復元される場合、`thickness_start_epoch=50` によって再開直後から厚み制約が有効になる。

## Thickness Violation Metric
厚み違反量は次で評価する。

$$
v_k = \max(G_{\mathrm{thick}, k}, 0)
$$

テストでは、再開後の各stepで次を確認する。

$$
v_{k+1} \le v_k + \epsilon
$$

初期許容値は $\epsilon = 10^{-6}$ とする。

## Testing Strategy
TDDとして、まず失敗するテストを書く。

テスト名:

```python
def test_restart_from_step50_decreases_thickness_violation():
    ...
```

テストの流れ:

1. `examples/checkpoints/1_validation_alt_step50.npz` をロードする。
2. `MMAState.from_array()` で50 step時点のMMA状態を復元する。
3. 厚み制約ONで数stepだけ再開する。
4. `G_thick` が有限値であることを確認する。
5. `max(G_thick, 0)` がstepごとに下がることを確認する。

通常のテストでは、50 step計算そのものは実行しない。50 step checkpointがない場合は、明示的に失敗またはskipにする方針を選ぶ。

## Commands
環境作成:

```bash
uv venv .venv --python 3.12
```

テスト実行:

```bash
.venv/bin/python -m unittest tests.test_thickness_restart_from_checkpoint
```

ノートブック構造確認:

```bash
.venv/bin/python -c "import json; nb=json.load(open('examples/1_validation_alt.ipynb')); print(len(nb['cells']))"
```

## Implementation Notes
- 変更はまず `examples/1_validation_alt.ipynb` に閉じる。
- checkpoint保存用セルとcheckpointロード再開用セルを分ける。
- `optimize_design()` には `initial_mma_state=None` を追加する。
- `initial_mma_state is None` の場合は従来通り一様初期密度から開始する。
- `initial_mma_state is not None` の場合は、その `x` を設計変数として使う。
- `convg_history` には再開後の `G_thick` 履歴を残す。
- 厚み制約の式と感度処理は `examples/thickLSTO.ipynb` の `analyze_maximum_thickness()` を参照する。
- 生感度の確認には `shape_derivative`、最適化に入る厚み感度の確認にはHelmholtz後の `sensitivity` を使う。

## Success Criteria
- 50 step checkpointを `.npz` として保存できる。
- 保存した checkpoint をロードして `MMAState` を復元できる。
- 復元した状態から厚み制約ONで再開できる。
- 再開後の `G_thick` が有限値である。
- 再開後の厚み違反量 `max(G_thick, 0)` がstepごとに非増加になる。
- 既存の `examples/1_validation.ipynb` は変更しない。

## Open Questions
- checkpointが存在しない場合、テストをfailにするかskipにするか。
- checkpoint `.npz` をGit管理するか、生成手順のみGit管理するか。
- 再開後の検証step数を5にするか、より短い2から3 stepにするか。
