<!-- Instruction sheet as given (verbatim), except that the repository name is written as `$REPO` (= this repository at tag v1-language-sciences-2026-03). Paths are those of the working folder at the time; see docs/README.md for the current layout. -->

# 指示書：T1〜T3（CPG 改訂版、v1.0、2026-10-22）

`cpg_revision/` フォルダと、リポジトリ `$REPO` を使う。
実行順は **T3 → T1 → T2 → T4（統合分析）**。T1 と T2 は互いに独立なので並行してよい。

前提（既存ファイル）：
- `coding/blind_items.csv`（217項目、ブラインド表）
- `coding/results/codes_{gpt,gemini,claude}.csv`（第1回判定。`clean_host`, `components` を含む）
- `coding/results/consensus_codes.csv`, `coding/results/recording_consensus.csv`
- `out/items_with_consensus.csv`（213項目、pull 3種＋合議コード）
- `embed/nnpm_replicate.py`, `embed/analyze_replication.py`
- リポジトリの `results/69method/control_cache.json`, `results/69method/cpg_archaeology_results.csv`

共通ルール：
- 数値はすべてスクリプト出力から生成する。手で書き写さない。
- 推測で埋めない。判断できない項目は空欄か `UNRESOLVED` にして、理由を残す。
- 仕様から外れた判断は、すべて `DEVIATIONS_T1_T3.md` に記録する。
- 乱数は固定シード（20261022）。LLM と埋め込み API の呼び出しは JSONL に全件ログを残す。

---

## T3：除外リストの確定（最初にやる）

### やること
1. `SEA_012` を除外する（宿主語が仮置きの文字列 `[technique-level item …]`）。
2. ほかにも同じ種類の問題がないか、全217項目の `host_original` と `host_romanized` を機械的に点検する。
   次のどれかに当てはまる項目を**候補として一覧にする**（自動では除外しない）：
   - 角括弧 `[ ]` や丸括弧の中に説明文がある（例：`(visual diphrasis — no spoken form)`）
   - 英語の説明文になっている（例：`Ritual address to landscape entities`）
   - `/` や `;` で複数の形が並んでいる（例：`unalive / corn / … / grape`）
   - 歌や詩の数行まるごと（20文字を超える文、改行を含むもの）
3. 一覧 `t3/exclusion_candidates.csv` を作る。列：`item_id, host_original, host_romanized, problem_type, proposed_action`
   - `proposed_action` は `exclude` / `use_first_form` / `keep` / `ask_author` のどれか。
     迷うものは `ask_author` にする。
4. **確定した除外リスト** `t3/exclusions.csv`（`item_id, reason`）には、まず `SEA_012` だけを入れる。
   ほかの候補はボスの判断待ちとし、`ask_author` のまま報告する。

### 出力
- `t3/exclusion_candidates.csv`、`t3/exclusions.csv`、`t3/README.md`（点検ルールと件数）

---

## T1：対照語の掃除と、清書した宿主語での NNPM 再実行

### 背景
- 対照語リストに、宿主語そのもの（54件）や宿主語を含む語（15件、例：苹果 → 红苹果）が混ざっている。
- 宿主語の62件に英語の注釈が混ざっている（例：`pingguo_apple`、`Hodam (glutton / fat stomach)`）。
- 宿主語と対照語の文字体系が揃っていないと結果がゆがむ（bge-m3 で確認済み）。

### 1. 清書した宿主語（clean host）の決定 → `t1/clean_hosts.csv`
- 第1回判定の3モデルの `clean_host` を多数決（2/3以上の一致）で採用する。
- 一致しない場合は `blind_items.csv` の `host_original` を使い、`clean_host_source = fallback_original` と記録する。
- 列：`item_id, clean_host, clean_host_source, script`（`script` は Latin / CJK / Arabic / Cyrillic / Devanagari / Ethiopic / Javanese / Other を Unicode 範囲で判定）

### 2. 対照語の掃除 → `t1/controls_clean.csv`
`control_cache.json` の対照語について、項目ごとに次を**順に**適用する。

| 規則 | 内容 |
|---|---|
| R1 同一 | 対照語が clean_host・host_original・host_romanized のどれかと一致（大文字小文字・空白・声調記号を無視）→ 除外 |
| R2 部分一致 | 対照語が clean_host を含む、または clean_host が対照語を含む（どちらも2文字以上）→ 除外 |
| R3 文字体系 | 対照語の文字体系が clean_host と違う → 除外 |
| R4 重複 | 同じ対照語が2回以上 → 1つだけ残す |

- 列：`item_id, control, kept, rule_dropped`
- 項目ごとの残数を `t1/controls_summary.csv` に出す（`n_original, n_kept, n_dropped_R1..R4`）。
- 残数が5未満の項目は `low_control` フラグを立てる。**対照語を新しく作り直すことはしない**（今回の範囲外）。

### 3. NNPM の再実行（条件 C）
- `embed/nnpm_replicate.py` を拡張して、条件 C を足す：
  **宿主＝clean_host、対照＝掃除後の対照語（最大10個、元の並び順）**。
- 3モデルすべてで回す：
  - `text-embedding-3-large`（OpenAI、3072次元）
  - `BAAI/bge-m3`（ローカル）
  - `gemini-embedding-001`（`task_type = SEMANTIC_SIMILARITY`）
- k の感度：k = 3, 5, 10（残数が k 未満の項目は、その k の分析から外す）。
- 出力：`embed/out/nnpm_C_{model}.csv`（列：`item_id, tradition, clean_host, n_controls_kept, sim_host_target, pull_C_k3, pull_C_k5, pull_C_k10`）

### 4. 事前に固定する仮説（実行前にこのファイルの内容で固定）
- **H-C1**：T3 の除外を適用し、`low_control` でない項目で、条件 C の平均 pull（k = 10）が0より大きい。
  判定：3モデルそれぞれで混合モデル `pull ~ 1 + (1 | tradition)` の切片の片側検定。Holm 補正後 p < .05 のモデル数を報告し、
  **3モデルとも満たせば「支持」、1〜2モデルなら「一部支持」、0なら「不支持」**。
- **H-C2**：同じ検定を、第1回合議で `cpg_valid = yes` の項目だけで行う（判定基準は H-C1 と同じ）。

### 5. 報告する分析（すべて3モデル並べて）
- 項目数の内訳（全体 → T3 除外 → `low_control` 除外 → 解析数）。数が合うことを明記。
- 条件 B（前回）と条件 C の比較：平均、正の割合、t、混合モデル z。
- 記録状態（`recording_status`）別、方向（R/E/S）別、主チャネル別の記述統計。
- 伝統別（n ≥ 3）の検定と BH 補正。
- 感度分析：`low_control` を含めた場合、k = 3 / 5 の場合。
- モデル間の項目単位の順位相関（Spearman）。
- 出力：`out/t1_results.json`、`out/t1_results.md`

---

## T2：形と境界の再判定（コードブック v1.1）

### 背景
前回の判定では、ケニング・草泥马・五月三十五日まで「造語（novel_form）」になった。
CPG の定義を「**表の層（字面で読める意味）と裏の層（共同体に保存された意味）の2階建て**」に改めたので、それに合わせて判定し直す。
あわせて、語用論との境界のテストもデータにする（探索的）。

### 1. 新しいコードブック `coding/codebook_v1_1.md`（英語、判定者に渡す本文）

#### `form_type`（exactly one）
- `existing_form` — an ordinary word, phrase, number, emoji or fixed expression that already had its own surface meaning before the covert meaning attached (草 "grass", tea, 304, 海老, "Netflix and chill").
- `transparent_combination` — a new combination of existing elements whose **literal** meaning an outsider can still read, while the covert meaning needs community knowledge (whale-road, 草泥马 "grass-mud horse", 五月三十五日 "May 35th", メダカブス).
- `opaque_coinage` — a new form with no literal meaning an outsider can read (マジ卍, ズボラズギュン).
- `blend_or_abbreviation` — a new form whose meaning is just the sum of its recognisable parts, with no second layer (チョベリバ, サボる).
- `expressive_coinage` — a new form motivated by sound or feeling only, with no second layer (ぴえん).
- `respelling` — a deliberate misspelling or variant of an existing word that keeps the same meaning (seggs, unalive).

#### `two_storey`（`yes` / `no`）
`yes` only if the item has BOTH (a) a literal surface layer an outsider can read and (b) a different covert layer shared by a community.

#### `content_or_attitude`（exactly one）
- `content` — under the covert reading, the thing referred to changes (草: plant → laughter; 牡丹: flower → boar meat).
- `attitude` — the referent stays the same and only the speaker's evaluation or stance changes (いい度胸してるな; ご自由にどうぞ).
- `both`

#### `outsider_inference`（exactly one）
Imagine a fluent speaker of the language who is not a member of the community, given the full context of use.
- `inferable` — they could work out the covert meaning from context and general reasoning alone.
- `inferable_with_effort` — they could work it out, but it is a conventional shortcut (e.g. "Can you pass the salt?").
- `not_inferable` — they would need community or cultural knowledge; once told, the link makes sense.
- `not_motivated` — even once told, there is no understandable link.

#### `host_unit`（exactly one）
`word` / `phrase` / `utterance`（a whole sentence or saying, e.g. ぶぶ漬けでもどうどす）

#### `confidence`（1 / 2 / 3）と `rationale`（1〜2文）

出力キー：`item_id, form_type, two_storey, content_or_attitude, outsider_inference, host_unit, confidence, rationale`

### 2. 実行
- `coding/run_llm_coding.py` を流用して `coding/run_v1_1_coding.py` を作る（列順バグ修正済みの版を使う）。
- モデル：`gpt-5.6-sol`、`gemini-3.1-pro-preview`、`claude-opus-5-5`。前回と同じ設定。
- 対象：`blind_items.csv` の217項目全部（T3 の除外は分析時に適用）。
- 10項目の試運転 → 本番。拒否（refusal）が出たバッチは1件ずつ再実行し、その旨をログに残す。
- 出力：`coding/results/v1_1_{gpt,gemini,claude}.csv`（＋ `.jsonl`）

### 3. 集計 `coding/aggregate_v1_1.py`
- 各項目の Fleiss の κ と全員一致率。
- 合議ラベル（2/3以上の一致、なければ `UNRESOLVED`）→ `coding/results/v1_1_consensus.csv`
- 旧 `form_status` と新 `form_type` のクロス表（ケニング6件・草泥马・五月三十五日がどこに移ったかを明示）。
- `two_storey` と第1回 `cpg_valid` のクロス表。

### 4. 事前に固定する扱い
- **CPG の範囲**＝ `two_storey = yes` かつ `outsider_inference = not_inferable`。
- それ以外（語用、使い方の慣習、造語の各種）は境界例として**記述統計のみ**で報告する。
- 旧基準（`form_status = existing_form`）の結果も、比較のため並べて残す。

---

## T4：統合分析（T1・T2・T3 が終わってから）

- `out/items_final.csv` を作る：T3 除外を適用し、T1 の条件 C の pull（3モデル）、T2 の合議ラベル、
  第1回の合議コード、記録状態をすべて1つの表に。
- CPG の範囲（T2 の定義）で、H-C1 と同じ検定をもう一度行う（探索的として明記）。
- `outsider_inference` 別（inferable / inferable_with_effort / not_inferable）の pull の記述統計。
- `content_or_attitude` 別の記述統計。
- 出力：`out/final_results.json`、`out/final_results.md`。
- `verified_numbers.md` に第13節として要約を追記。

---

## 受け入れ条件
- 3モデル × 217項目の判定で `failed` が0件。全呼び出しが JSONL に記録されている。
- T1 の3モデルすべてで埋め込みが完了し、取得できなかった文字列は一覧にしてある。
- 項目数の内訳が、各段階で合っている（除外・低対照・UNRESOLVED の件数を明示）。
- LLM と埋め込みの呼び出し以外を最初から再実行し、出力が完全に一致する。
- 報告の最後に「判断が必要な点」を箇条書きで出す（`ask_author` の項目を含む）。
