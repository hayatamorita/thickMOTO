# Task Breakdown: Step 50 Checkpoint Restart

## Overview
`examples/1_validation_alt.ipynb` を今回の編集対象にして、50 step時点の `MMAState` を保存し、それをロードして厚み制約ONで再開する。

実装はTDD順に進める。先に失敗するテストを作り、その後にノートブック側へ再開APIとcheckpoint保存・ロード処理を追加する。

## Architecture Decisions
- 50 step形状は密度画像ではなく `MMAState.to_array()` で保存する。
- 再開は `MMAState.from_array()` で行う。
- 通常テスト内では50 step計算を毎回実行しない。
- 厚み制約の評価式は `examples/thickLSTO.ipynb` のPDEロジックを参照する。
- 実装対象は `examples/1_validation_alt.ipynb` に限定する。

## Dependency Graph
1. 現状把握とテスト条件の固定
2. checkpoint fixtureの仕様固定
3. 失敗するrestartテスト作成
4. `optimize_design()` の再開API追加
5. checkpoint保存・ロードセル追加
6. restartテストの合格化
7. 手動検証と仕様・タスク更新

## Task List

### Phase 1: Test Contract

- [o] Task 1: `1_validation_alt.ipynb` の現在構造を確認する
  - Acceptance: `optimize_design()` の引数、返り値、`convg_history` のキーを確認できている。
  - Acceptance: `MMAState` の保存・復元に必要な値が特定できている。
  - Verify: `.venv/bin/python -c "import json; nb=json.load(open('examples/1_validation_alt.ipynb')); print(len(nb['cells']))"`
  - Files: `examples/1_validation_alt.ipynb`
  - Dependencies: None
  - Scope: XS

- [o] Task 2: checkpoint fixtureの扱いを決める
  - Acceptance: `examples/checkpoints/1_validation_alt_step50.npz` のキー一覧が決まっている。
  - Acceptance: checkpointがない場合のテスト挙動を `skip` または `fail` のどちらかに固定する。
  - Verify: checkpoint仕様がテスト内の定数として読める。
  - Files: `tests/test_thickness_restart_from_checkpoint.py`
  - Dependencies: Task 1
  - Scope: XS

- [o] Task 3: 失敗するrestartテストを書く
  - Acceptance: `test_restart_from_step50_decreases_thickness_violation()` が存在する。
  - Acceptance: `MMAState.from_array()` でcheckpointを復元する想定になっている。
  - Acceptance: 再開後の `max(G_thick, 0)` 非増加を検証している。
  - Verify: `.venv/bin/python -m unittest tests.test_thickness_restart_from_checkpoint`
  - Files: `tests/test_thickness_restart_from_checkpoint.py`
  - Dependencies: Task 2
  - Scope: S

### Checkpoint: RED
- [o] Test fails for the expected reason: restart API or checkpoint loading path is missing.
- [o] Failure reason is not import error from unrelated modules.

### Phase 2: Notebook Restart API

- [o] Task 4: `optimize_design()` に `initial_mma_state` を追加する
  - Acceptance: `initial_mma_state=None` なら従来通り一様初期密度から開始する。
  - Acceptance: `initial_mma_state` が渡されたら `init_mma()` し直さない。
  - Acceptance: `num_design_var` と `initial_mma_state.x` の形状不一致を検出する。
  - Verify: 1 stepだけ再開呼び出しできる。
  - Files: `examples/1_validation_alt.ipynb`
  - Dependencies: Task 3
  - Scope: S

- [o] Task 5: `convg_history` にrestart検証用履歴を残す
  - Acceptance: `thickness_cons` に再開後の `G_thick` が記録される。
  - Acceptance: `epoch` がcheckpointのepochから継続する。
  - Acceptance: `constraint_mode="volume_and_thickness"` と `thickness_start_epoch=50` で厚み制約が有効化される。
  - Verify: 再開後の `convg_history["epoch"]` と `convg_history["thickness_cons"]` を表示する。
  - Files: `examples/1_validation_alt.ipynb`
  - Dependencies: Task 4
  - Scope: XS

### Checkpoint: Restart API
- [o] `initial_mma_state=None` の既存動作が維持されている。
- [o] `initial_mma_state` 指定時にcheckpoint epochから再開できる。

### Phase 3: Checkpoint Save and Load

- [o] Task 6: 50 step checkpoint保存セルを追加する
  - Acceptance: `examples/checkpoints/` を作成して `.npz` を保存できる。
  - Acceptance: `mma_state.to_array()`、`num_design_var`、`epoch`、`max_vol_frac` を保存する。
  - Acceptance: 確認用に `rho_phys`、`G_thick`、`G_vol` を保存できる。
  - Verify: `.npz` を `np.load()` で開き、必要キーが存在する。
  - Files: `examples/1_validation_alt.ipynb`
  - Dependencies: Task 5
  - Scope: S

- [o] Task 7: checkpointロード再開セルを追加する
  - Acceptance: `.npz` から `MMAState` を復元できる。
  - Acceptance: 復元状態から厚み制約ONで短いstep数だけ再開できる。
  - Acceptance: 再開後の `G_thick` と厚み違反量を表示できる。
  - Verify: 2から5 stepのrestartを手動実行する。
  - Files: `examples/1_validation_alt.ipynb`
  - Dependencies: Task 6
  - Scope: S

### Checkpoint: Manual Restart
- [] 50 step checkpointを保存できる。
- [] checkpointロード後、厚み制約ONで再開できる。
- [] 再開後の `G_thick` が有限値である。

### Phase 4: Thickness Reference and Diagnostics

- [o] Task 8: thickLSTO参照ロジックとの差分を明示する
  - Acceptance: `shape_derivative` は生感度、`sensitivity` はHelmholtz後感度として扱う。
  - Acceptance: `1_validation_alt.ipynb` では反応拡散更新を移植しない。
  - Acceptance: MMAに入る厚み感度がどれかをコメントまたは表示で確認できる。
  - Verify: notebook内で `S_comp`, `S_thick`, `cos` を確認する。
  - Files: `examples/1_validation_alt.ipynb`
  - Dependencies: Task 7
  - Scope: S

- [o] Task 9: restartテストをGREENにする
  - Acceptance: checkpointがある場合、restartテストが通る。
  - Acceptance: `max(G_thick, 0)` がstepごとに非増加である。
  - Acceptance: `G_thick`、感度、設計変数が有限値である。
  - Verify: `.venv/bin/python -m unittest tests.test_thickness_restart_from_checkpoint`
  - Files: `tests/test_thickness_restart_from_checkpoint.py`, `examples/1_validation_alt.ipynb`
  - Dependencies: Task 8
  - Scope: M

### Checkpoint: GREEN
- [o] restartテストが通る。
- [o] checkpointなしの場合の挙動が仕様通りである。

### Phase 5: Final Verification

- [o] Task 10: 既存厚みテストを実行する
  - Acceptance: 既存の厚みPDE、投影、勾配テストが壊れていない。
  - Verify: `.venv/bin/python -m unittest tests.test_thickness_pde tests.test_thickness_projection tests.test_thickness_gradient tests.test_thickness_constraint`
  - Files: None
  - Dependencies: Task 9
  - Scope: XS

- [o] Task 11: docsとタスク状態を更新する
  - Acceptance: 完了済みタスクを `[o]` に更新する。
  - Acceptance: 実装中に仕様変更があればspecへ反映する。
  - Verify: `wc -l docs/specs/step50-thickness-restart-checkpoint-spec.md docs/plans/step50-thickness-restart-checkpoint-tasks.md`
  - Files: `docs/specs/step50-thickness-restart-checkpoint-spec.md`, `docs/plans/step50-thickness-restart-checkpoint-tasks.md`
  - Dependencies: Task 10
  - Scope: XS

## Risks and Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| 50 step checkpointが大きい | Medium | `.npz` のGit管理は実装時に判断し、必要なら生成手順だけ管理する |
| restart APIがnotebook内に閉じていてテストから呼びにくい | Medium | 最初はnotebook内の最小API追加に留め、必要なら後でhelper化する |
| 厚み違反量が完全な単調減少にならない | High | 許容値 $\epsilon=10^{-6}$ を使い、失敗時は原因を記録して条件を再検討する |
| `thickLSTO.ipynb` とMMA側の感度スケールが異なる | High | `S_comp`, `S_thick`, `cos` を出してスケール差を確認する |

## Open Questions
- checkpointが存在しない場合、テストを `skip` にするか `fail` にするか。
- `examples/checkpoints/1_validation_alt_step50.npz` をGit管理するか。
- restart検証step数を5 stepにするか、2から3 stepに短縮するか。
