# Spec: 1_validation_BC density characteristic mode

## Objective

`examples/1_validation_BC.ipynb`で、厚みPDEへ入れる材料指示関数を切り替え可能にする。
目的は、厚み感度の斑点化が平滑化Heaviside微分に由来するかを検証することである。

現状A:

```text
rho_tilde
-> tanh projection
-> rho_bar
-> phi = 2(rho_bar - 0.5)
-> chi = H(phi)
-> thickness PDE
```

案B:

```text
rho_tilde
-> tanh projection
-> rho_bar
-> chi = GIMP(rho_bar)
-> thickness PDE
```

案Bでは次の局在化を除く。

$$
\frac{\partial \chi}{\partial \phi}
$$

ただし、tanh投影微分は残す。

$$
\frac{\partial \rho_{bar}}{\partial \rho_{tilde}}
$$

## Tech Stack

- Python 3.12
- JAX
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook
- 既存の`moto.src.thickness_constraint`

## Commands

Python確認:

```bash
.venv/bin/python -V
```

ノートブック構文確認:

```bash
.venv/bin/python - <<'PY'
import ast
import json
from pathlib import Path

nb = json.loads(Path("examples/1_validation_BC.ipynb").read_text())
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        ast.parse("".join(cell.get("source", [])), filename=f"cell-{i}")
print("syntax ok")
PY
```

厚み制約関数の形状確認:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl .venv/bin/python scripts/check_bc_mode.py
```

`scripts/check_bc_mode.py`は必要な場合だけ一時的に作成し、コミット対象にしない。

## Project Structure

```text
examples/1_validation.ipynb     比較用の元ノートブック
examples/1_validation_BC.ipynb  案B検証用ノートブック
docs/specs/                     本仕様書
moto/src/thickness_constraint.py 共有実装。初期実装では変更しない
```

## Code Style

切り替え変数はノートブックの厚みパラメータ付近に置く。

```python
thickness_characteristic_mode = "density"
```

有効値は2つに限定する。

```python
"heaviside"  # 現状A
"density"    # 案B
```

案Bの感度は次で戻す。

```python
gradient_rho_bar = projection.transpose(analysis.gradient_characteristic)
gradient_rho_tilde = gradient_rho_bar * threshold_derivative
gradient_design = density_filter.T @ gradient_rho_tilde
```

不正なモードは`ValueError`にする。

## Testing Strategy

最低限の確認:

- 全コードセルがPython構文として正しい
- `"density"`で`constraint.shape == (1, 1)`
- `"density"`で`gradient.shape == (1, num_design_var)`
- `"density"`で`gradient`が有限値
- `"heaviside"`へ戻しても同じ形状で動く

現在確認済みの形状:

```text
density   constraint=(1, 1), gradient=(1, 11024), finite=True
heaviside constraint=(1, 1), gradient=(1, 11024), finite=True
```

最適化結果の確認:

- `Density rho_bar`
- `Raw thickness dG_thick/dx`
- `MMA thickness lambda_thick dG_thick/dx`
- `G_vol`
- `G_thick`
- `S_comp`
- `S_thick`
- `cos`

## Boundaries

- Always: `examples/1_validation_BC.ipynb`だけで案Bを検証する。
- Always: `examples/1_validation.ipynb`は比較用として残す。
- Always: `moto/src/thickness_constraint.py`は初期実装では変更しない。
- Ask first: 案Cの`chi = GIMP(rho_tilde)`を追加する場合。
- Ask first: MMAの制約スケーリングやmove limitを変更する場合。
- Never: 厚み感度を黙ってゼロにしない。

## Success Criteria

- `thickness_characteristic_mode = "density"`で案Bが使える。
- `"heaviside"`へ戻せば現状Aと同じ経路を使える。
- 案Bの厚み感度が、境界だけでなく密度領域にも出ることを確認できる。
- 案Bで`G_thick`が低下するか確認できる。
- `G_vol`が大きく負になる場合は、対策検討へ進む。

## Current Observation

保存済み実行結果では、案Bで`G_thick`は低下している。

```text
epoch 0: G_thick = 1.998e-01
epoch 9: G_thick = 1.263e-01
```

一方で体積制約は大きく負になっている。

```text
epoch 9: G_vol = -3.601e-01
```

また、感度スケールは厚み側が支配的である。

```text
S_comp  = 3.733e-03
S_thick = 3.633e-01
cos     = -0.519
```

このため、案Bは厚み違反を下げる方向には働いているが、密度分布は構造形成より低密度化へ寄りやすい。

## Open Questions

- 案Bでは厚み制約のスケーリングをMMA前に調整するか。
- 体積制約が負側へ外れすぎる場合、体積下限制約を追加するか。
- `move_limit`を小さくするか。
- 案Cを追加して、tanh投影微分の影響まで切り分けるか。
