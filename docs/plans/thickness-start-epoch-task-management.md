# Task Management: 厚み制約の遅延開始

## Overview

`examples/1_validation.ipynb` に、MMA epoch $n$ から厚み制約を有効化する
`thickness_start_epoch` を追加する。開始前も厚み制約値、厚み場、違反場は評価して
履歴と可視化へ残すが、MMA には厚み制約を作用させない。

## Decisions

- `thickness_start_epoch` は0始まりの MMA epoch とする。
- `n=0` は現在と同じく最初から厚み制約を有効にする。
- `n=30` は epoch 0〜29を体積制約のみ、epoch 30以降を2制約とする。
- `constraint_mode="volume"` では全期間で体積制約のみとする。
- `constraint_mode="volume_and_thickness"` では `n` 以降のみ厚み制約を有効にする。
- MMA の `num_cons` は途中で変更せず、2制約モードでは最初から2に固定する。
- 開始前は MMA に厚み制約値 `-1.0`、厚み感度ゼロを渡す。
- 実際の `G_thick` は開始前も評価し、履歴、状態表示、可視化に使用する。
- 開始前の実効厚み感度は `S_thick=0` とする。

## Dependency Graph

```text
切替条件と入力検証
  -> MMA入力の遅延切替
    -> 履歴・診断表示の整合
      -> 境界epochの短時間検証
        -> 既存モードの回帰検証
```

## Task Management Table

| ID | Phase | Task | Status | Depends on | Scope |
|---|---|---|---|---|---|
| T1 | Foundation | 開始epochの引数と検証を追加 | Done | None | XS |
| T2 | Core | MMAへ渡す厚み制約をepochで切替 | Done | T1 | S |
| T3 | Diagnostics | 実値履歴と感度表示を切替状態へ対応 | Done | T2 | S |
| C1 | Checkpoint | 静的検査と既存単体テスト | Done | T1-T3 | XS |
| T4 | Validation | 開始境界の短時間実行を検証 | Done | C1 | S |
| T5 | Regression | `volume` と `n=0` の互換性を検証 | Done | T4 | S |
| C2 | Complete | 結果レビューと差分確認 | Done | T5 | XS |

## T1: 開始epochの引数と検証を追加

**Description:** `optimize_design()` に `thickness_start_epoch: int = 0` を追加し、
不正値を早期に拒否する。呼び出しセルから開始epochを指定できるようにする。

**Acceptance criteria:**

- [ ] `thickness_start_epoch=0` が既定値である。
- [ ] 負数、整数でない値、真偽値を拒否する。
- [ ] 呼び出しセルの1か所で開始epochを設定できる。

**Verification:**

- [ ] Notebook の全コードセルが構文解析できる。
- [ ] `0`、正整数、不正値について入力検証を確認する。

**Dependencies:** None

**Files likely touched:**

- `examples/1_validation.ipynb`

**Estimated scope:** XS

## T2: MMAへ渡す厚み制約をepochで切替

**Description:** 実際の厚み解析は毎epoch実行しつつ、MMA用の厚み制約値と感度だけを
`thickness_active` に応じて切り替える。

**Acceptance criteria:**

- [ ] `thickness_active` は2制約モードかつ `epoch >= n` の場合だけ真になる。
- [ ] 開始前は MMA に `G_thick=-1.0` とゼロ感度を渡す。
- [ ] 開始epoch以降は実際の `G_thick` と `grad_thickness` を渡す。
- [ ] `num_cons` と MMA state を途中で作り直さない。

**Verification:**

- [ ] `constr.shape==(2, 1)` と `grad_cons.shape==(2, num_design_var)` を維持する。
- [ ] 開始前後で MMA 用配列の第2行だけが期待どおり切り替わる。

**Dependencies:** T1

**Files likely touched:**

- `examples/1_validation.ipynb`

**Estimated scope:** S

## T3: 実値履歴と感度表示を切替状態へ対応

**Description:** MMA用のダミー値と、解析で得た実際の厚み制約値を混同しないよう、
履歴、ログ、`S_thick`、`cos` の計算を整理する。

**Acceptance criteria:**

- [ ] `convg_history["thickness_cons"]` には常に実際の `G_thick` を保存する。
- [ ] 状態表示に `thickness_active` と開始epochが分かる情報を含める。
- [ ] 開始前は `lambda_thick=0`、`S_thick=0`、`cos=nan` となる。
- [ ] 開始後は MMA が返した厚み乗数を感度診断に使用する。

**Verification:**

- [ ] 開始前も `chi_node`、厚み場、違反場を表示できる。
- [ ] ログ上の実際の `G_thick` と MMA用ダミー値が混同されない。

**Dependencies:** T2

**Files likely touched:**

- `examples/1_validation.ipynb`

**Estimated scope:** S

## C1: 静的検査と既存単体テスト

- [ ] Notebook を `nbformat` で読み込める。
- [ ] 全コードセルを `compile()` できる。
- [ ] `.venv/bin/python -m unittest discover -s tests -v` が成功する。
- [ ] `git diff` に無関係な変更や大きな出力差分がない。

## T4: 開始境界の短時間実行を検証

**Description:** 小さい反復数と `thickness_start_epoch=2` を使い、開始直前と開始時の
切替を実際のMMA更新で確認する。

**Acceptance criteria:**

- [ ] epoch 0、1では厚み制約が非アクティブである。
- [ ] epoch 2では厚み制約がアクティブになる。
- [ ] 全epochで実際の `G_thick` が評価・保存される。
- [ ] 制約値と感度に NaN、Inf がない。

**Verification:**

- [ ] `max_iter=3` 程度の短時間実行ログを確認する。
- [ ] epoch 1から2への切替でMMA stateが再初期化されていない。

**Dependencies:** C1

**Files likely touched:**

- `examples/1_validation.ipynb`

**Estimated scope:** S

## T5: 既存モードの回帰検証

**Description:** 既存の体積制約のみのモードと、最初から厚み制約を入れるモードが
変更前の意味を維持することを確認する。

**Acceptance criteria:**

- [ ] `constraint_mode="volume"` では `n` に関係なく体積制約だけをMMAへ渡す。
- [ ] `constraint_mode="volume_and_thickness", n=0` は現在の挙動と一致する。
- [ ] `n >= max_iter` では全反復が実質的に体積制約のみになる。

**Verification:**

- [ ] 各ケースを1〜3反復で実行し、制約数、乗数、ログを比較する。
- [ ] 既存15件の単体テストが成功する。

**Dependencies:** T4

**Files likely touched:**

- `examples/1_validation.ipynb`

**Estimated scope:** S

## C2: Complete

- [ ] 全Acceptance criteriaを満たす。
- [ ] Notebookの実行順依存を増やしていない。
- [ ] 開始epochの意味が呼び出しセルから明確である。
- [ ] 実行出力を必要以上にコミットしていない。
- [ ] ユーザーのレビュー後に実装完了とする。

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 開始前に正の実制約とゼロ感度をMMAへ渡す | High | MMA用には負のダミー制約を渡す |
| epoch途中で制約数を変えてMMA履歴を壊す | High | `num_cons=2` を全期間維持する |
| 実値とダミー値を履歴で混同する | Medium | 解析値とMMA入力を別変数にする |
| 開始時に厚み感度が急増する | Medium | 短時間検証後に開始epochとmove limitを調整する |
| `n` の0始まり/1始まりを誤解する | Medium | 0始まりを引数説明とログに明記する |
| Notebook出力で差分が肥大化する | Low | 実装コミット前に出力差分を確認する |

## Parallelization

- T1〜T3は同じ最適化ループを変更するため順次実施する。
- T4とT5も同じNotebook実行環境を使うため順次実施する。
- 実装規模が1ファイル中心のため、並列化の利点は小さい。

## Out of Scope

- Newton未収束時のMMAロールバック。
- `thresh_beta`、`characteristic_width`、`move_limit` のcontinuation変更。
- 厚み制約を徐々に重くするランプ係数。
- `moto/src/thickness_constraint.py` のPDE・感度式変更。
