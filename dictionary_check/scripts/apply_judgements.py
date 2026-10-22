"""Add the 大辞泉 judgement columns to dictionary_check/private/daijisen_text.csv (not published).

Usage: python3 dictionary_check/scripts/apply_judgements.py   (run after fetch_kotobank.py; then export_public.py)

判定 (where the covert meaning appears in the retrieved 大辞泉 text):
    番号つきの語義   it is a numbered sense (or sub-sense ㋐㋑…) of the entry
    補説・用例      not a sense, but stated in [補説], a usage note inside a sense, or the explanatory part of the
                    definition (e.g. why the food is used at celebrations)
    なし            the retrieved text does not state it
    (blank)         nothing was retrieved for the headword
一致度: 一致 = the text states the covert meaning itself; 近い = a closely related sense (see 判定メモ).

The judgements below were made by Claude (claude-opus-5-5) reading the extracted text on 2026-10-21; they are
not dictionary-editor judgements. Every 引用 is checked to be a verbatim substring of 本文; the script fails
otherwise. Several quotes for one item are joined with " / ".
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1] / "private"
G, H, N = "番号つきの語義", "補説・用例", "なし"

J = {  # item_id: (判定, [引用...], 一致度, 判定メモ)
    "T_001": (G, ["４ 《「獅子に牡丹」の「獅子」を「猪」にとりなしていう》イノシシの肉。"], "一致", ""),
    "T_002": (G, ["５ 鹿の肉。鹿には紅葉が取り合わせであるところからいう。"], "一致", "もみじ【紅葉】の語義5"),
    "T_003": (G, ["５ 《色が桜の花に似ているところから》馬肉の俗称。桜肉。"], "一致", ""),
    "T_004": (N, [], "", "柏（かしわ）の項に鶏肉の意はない。鶏肉の「かしわ」が別表記の見出しにあるかは未確認（取得範囲外）"),
    "T_005": (G, ["５ 夫婦になる。"], "近い",
              "「夫婦になる」（用例〈竹取〉「男は女に―・ふことをす」）。性的な逢瀬そのものの語義ではない"),
    "T_006": (N, [], "", ""),
    "T_007": (G, ["３ はかなく消えやすいこと。"], "一致", ""),
    "T_008": (G, ["２ 男女が情を通じる。情事を行う。"], "一致", ""),
    "T_009": (G, ["５ 共寝をする。同衾する。"], "一致", ""),
    "T_010": (N, [], "", "語義3「規律・けじめなどがなくなる」（風紀が乱れる）はあるが、性的な意の記述はない"),
    "T_011": (G, ["５ 相手の求めを退ける。はねつけて相手にしない。「女に―・られる」"], "近い",
              "「（恋人を）拒む」は語義5に該当。「雨が降る」は振るの項になく、降るは別見出し（未取得）"),
    "T_012": (H, ["和歌では多く「長雨」と掛けて用いる。"], "一致", "語義2（物思いにふけりながら見ること）の中の注記"),
    "T_013": (G, ["３ ちりぢりに消えてなくなる。", "８ 人がいさぎよく死ぬ。"], "近い",
              "「喪失」そのものの語義はなく、消滅（3）と死（8）が近い"),
    "T_014": (N, [], "", ""),
    "T_055": (N, [], "", "正月料理の煮豆とする記述はあるが、勤勉の意はない"),
    "T_056": (H, ["「数の多い子」と子孫繁栄の意にとって、新年・婚礼などの祝儀に用いる。"], "一致", "単一語義の定義文中の説明"),
    "T_057": (N, [], "", "長寿の記述はない"),
    "T_058": (H, ["[補説]「よろこぶ」に通じることから祝い事に用いられ"], "一致", ""),
    "T_059": (H, ["「めでたい」に通じるところから縁起のよい魚とされ"], "一致", "単一語義の定義文中の説明"),
    "T_060": (N, [], "", "語義2は拳銃の隠語。見通しの記述はない"),
    "T_061": (N, [], "", ""),
    "T_062": (N, [], "", ""),
    "T_063": (N, [], "", "紅白・蒲鉾とも祝いの意の記述はない（「紅白の幕」は色の用例）"),
    "T_064": (N, [], "", "語義2に「昔、田の肥料にしたことからの名という。正月の祝い肴にする。」とあるが、豊作の意は書かれていない"),
    "T_065": (N, [], "", ""),
    "T_066": (N, [], "", ""),
    "T_067": (N, [], "", ""),
    "T_076": (G, ["２ 男女の共寝。"], "一致", "とこ【床】の語義2（「床を共にする」の句そのものは見出しにない）"),
    "T_077": (G, ["５ 女性を誘惑する。"], "近い", "慣用句「手を出す」の項の語義5。性的接触そのものではなく誘惑"),
    "T_078": (N, [], "", "慣用句「一線を越える」の項は「守るべきことを破る。してはならないことをする。」のみ"),
    "T_079": (G, ["㋑性的な対象としてみた身体。肉体。「―を許す」「―の関係がある」", "４ 性的に交わること。"], "一致",
              "からだ【体】3㋑と関係【関係】の語義4"),
    "T_080": (N, [], "", "「深い関係」の見出しはない。深いの語義2に用例「―・い仲」があるが性的な意の記述はない"),
    "T_081": (N, [], "", "「大人の関係」の見出しはない"),
    "T_082": (N, [], "", "「最後まで」の見出しはない"),
    "T_083": (N, [], "", "「お泊まり」の見出しはなく、とまり【泊（ま）り】で照合"),
    "T_090": (G, ["２ ちょっと酒を飲むこと。「帰りに一杯やりませんか」"], "一致", ""),
    "T_091": (G, ["㋑酒をからだに入れる。「同僚と―・む」"], "一致", "「飲みに行く」の見出しはなく、飲むの1㋑"),
    "T_092": (N, [], "", "語義2「義理や社交上の必要から、行動をともにする」（用例「食事を―・う」）。飲酒の記述はない"),
    "T_093": (N, [], "", "「二軒目」の見出しはない"),
    "T_094": (G, ["２ ちょっと酒を飲むこと。"], "近い", "飲酒の語義はあるが「継続」の意はない"),
    "T_095": (N, [], "", "乾杯の語義自体が飲酒を含むが、「飲み会開始」の意は書かれていない"),
}

df = pd.read_csv(HERE / "daijisen_text.csv", encoding="utf-8-sig").fillna("")
assert set(J) == set(df.item_id), set(df.item_id) ^ set(J)
cols = {"判定": [], "引用": [], "一致度": [], "判定メモ": []}
for _, r in df.iterrows():
    if not r["本文"]:  # nothing retrieved: leave blank, never guess
        for k in cols:
            cols[k].append("")
        continue
    verdict, quotes, match, memo = J[r.item_id]
    for q in quotes:
        assert q in r["本文"], f"{r.item_id}: quote not found verbatim: {q}"
    assert (verdict == N) == (not quotes), r.item_id
    for k, v in zip(cols, [verdict, " / ".join(quotes), match, memo]):
        cols[k].append(v)
for k, v in cols.items():
    df[k] = v
df.to_csv(HERE / "daijisen_text.csv", index=False, encoding="utf-8-sig")
print(df["判定"].value_counts().to_string())
print(df.groupby(["判定", "一致度"]).size().to_string())
