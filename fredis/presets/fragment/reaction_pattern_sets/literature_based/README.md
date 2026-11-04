# 文献に基づく reaction pattern 集（literature_based）

ESI-CID（偶電子イオン）で報告されている開裂を、FREDIS の reaction pattern として書き出したものです。各パターンについて、根拠にした文献と、参照化合物で文献のイオンを再現できたかを記載します。

- 作成スクリプト：`fredis_projects/reaction_patterns/literature_based/scripts/build_literature_sets.py`
- 検証スクリプト：`fredis_projects/reaction_patterns/literature_based/scripts/verify_literature_sets.py`
  - fredis リポジトリのルートで `PYTHONPATH=. python …/verify_literature_sets.py` を実行します。
  - 現在の結果は「0 failure(s)」です。

## 1. FREDIS で表現できる反応・できない反応

FREDIS の reaction pattern は「reactant SMARTS の結合のうち、product に含まれない結合を切り、product の原子を含む連結成分を残す」という形で解釈されます。

| 項目 | FREDIS での扱い | このセットでの対応 |
|---|---|---|
| 結合の開裂（複数でもよい） | 可能 | すべてこの形で記述 |
| 結合次数の変更（単結合⇔二重結合） | 可能。ただし芳香族結合の次数は変えられない | RDA は `=`、芳香環は `~`（元の次数のまま） |
| **新しい結合の生成**（転位で R1–R2 がつながる、環縮小など） | **不可**。その product はエラーも出さずに捨てられる | `requires_bond_formation` に分け、エンジンが対応するまで何も生成しない |
| マップ番号のない reactant 原子 | その原子の結合はすべて切られる | 文脈は `$()`（再帰 SMARTS）で書き、すべての原子にマップ番号を付ける（検証スクリプトで確認） |
| 水素の移動 | fragment-ion の水素シフト（M_H_plus / M_H_minus では −5〜0）で扱う | 環を2本切って開く形で環縮小（H が 2 多い）を表現（例：CO の脱離） |
| 反応を適用する対象 | 元の化合物（Source）だけ。生成した断片に改めてパターンをマッチさせることはない | 2段階目の反応も元の化合物の構造で書く（例：coumarin_double_co_loss） |

`functional_group_cleavage` には、`nonring_single_bond`（環外の単結合をすべて切る）と同じ開裂が含まれます。両方を同じツリーに入れると同じ反応が二重に登録されるので、**どちらか一方だけを使ってください**。このセットは、切る結合を化学的に意味のあるものに限るので、組み合わせの数が少なくなります。

## 2. 既存パターンのレビュー（ファイルは変更していません）

| ファイル | 判定 | 内容 |
|---|---|---|
| `cyclic_carbonyl_co_elimination` | **何も生成しない** | product の `[*:4]-[*:7]` が新しい結合（環縮小）なので捨てられる。シクロヘキサノンでも 0 件。代わりに `ring_cleavage/cyclic_ketone_co_loss`（2本切って開環、H−2 で同じ m/z）を使う。 |
| `aromatic_elimination` | **何も生成しない** | product の `[*:1]=[*:8]`（電子環状反応による環化）が新しい結合なので捨てられる。オクタテトラエンでも 0 件。 |
| `pericyclic_cycloreversion_of_cyclohexane` | 化学的に疑問 | 二重結合のない六員環（シクロヘキサン、ピペリジン、テトラヒドロピランなど）すべてにマッチし、2原子の「アルケン」側だけを残す。CID の retro-Diels–Alder（RDA）は二重結合を含むシクロヘキセン型の環で起こる [5]。`ring_cleavage/retro_diels_alder_*` を推奨。 |
| `McLafferty_rearrangement`（2種） | 妥当 | γ水素の移動とα–β開裂。enol 側・alkene 側とも生成される [3, 4]。 |
| `nonring_single_bond/nonring_other_single_bond` | 片側だけ | product が `fragment_1` しかないので、S–S、P–O などの開裂で片側の断片しか残らない。`[*:2]` 側の product を追加することを推奨。 |
| `flavonoid/*` | 前回レビュー済み | `flavonoid3.md` を参照。 |

## 3. パターン一覧

根拠レベル：
- **A**：文献に機構とイオンがあり、参照化合物で FREDIS が同じイオンを再現した。
- **B**：総説や教科書に記載があり、参照化合物で再現した。
- **C**：文献に記載はあるが、参照化合物での確認をしていない。

### 3-1. functional_group_cleavage（官能基の開裂。汎用）
| パターン | 切る結合（残る断片） | 確認したイオン（参照化合物） | 文献 | 根拠 |
|---|---|---|---|---|
| alcohol_water_loss | C–OH（炭素側） | [M+H−H2O]+ 271（testosterone） | [1, 2] | B |
| carboxylic_acid_water_loss | C(=O)–OH（アシリウム） | 同じ開裂の class_quinolone パターンで [M+H−H2O]+ 314（ciprofloxacin）を確認 | [1, 13, 14] | B |
| carboxylic_acid_co2_loss | C–COOH（脱炭酸側） | 同じ開裂の class_quinolone パターンで [M+H−CO2]+ 288 を確認。負イオンでも典型 | [1, 13, 14] | B |
| primary_alkylamine_ammonia_loss | C–NH2 | 1件の生成のみ確認（文献イオンでは未確認） | [1] | C |
| alkylamine_loss | C–N（アミド・スルホンアミド・アリールアミンを除く） | [M+H−(CH3)2NH]+ 144（DMT） | [1, 2] | B |
| methyl_ester_methanol_loss | C(=O)–OCH3 | [M+H−CH3OH]+ 121（methyl salicylate） | [1] | B |
| ester_alkyl_oxygen_cleavage | O–C(alkyl)（酸側／アルキル側） | caffeate 179（chlorogenic acid、負イオン） | [1, 12] | A |
| ester_acyl_oxygen_cleavage | C(=O)–O（アシリウム／アルコール側） | quinate 191（chlorogenic acid、負イオン） | [1, 12] | A |
| ether_c_o_cleavage | O–C(sp3)（エステルを除く） | benzhydryl 167（diphenhydramine） | [1] | B |
| amide_c_n_cleavage | C(=O)–N（b型／y型） | b4 425、b3 278（Leu-enkephalin） | [9, 10] | A |
| acyl_carbon_monoxide_loss | Cα–C(=O)–X の2本（CO を失う） | a4 397 = b4 − CO（Leu-enkephalin、2反応） | [9] | A |
| aryl_methyl_ether_methyl_loss | O–CH3（CH3• を失う） | [M+H−CH3]+• 270（acacetin） | [2, 15] | B |
| aryl_methyl_ether_methanol_loss | Ar–OCH3（CH3OH を失う。オルト効果） | 未確認 | [2] | C |
| o_glycosidic_cleavage | アグリコン–O–C1（Y/Z、B/C 型） | Y0+ 303（isoquercitrin） | [7, 15] | A |
| sulfate_ester_cleavage | O–SO3（SO3 の脱離／HSO4−） | [M−H−SO3]− 269、HSO4− 97（estrone 3-sulfate） | [1, 11] | A |
| phosphate_ester_cleavage | C–O、O–P（H3PO4・HPO3 の脱離、H2PO4−、PO3−） | H2PO4− 97、PO3− 79（glucose 6-phosphate） | [1] | B |
| sulfonamide_s_n_cleavage | S–N、Ar–S | 156、99（sulfamethoxazole） | [16, 17] | A |
| sulfone_c_s_cleavage | C–SO2 | 未確認 | [1] | C |
| nitro_no2_loss | C–NO2 | [M−H−NO2]− 92（4-nitrophenol） | [18] | A |
| carbon_halogen_cleavage | C–Cl/Br/I（HX の脱離） | 未確認 | [1] | C |

### 3-2. ring_cleavage（環の開裂。汎用）
| パターン | 内容 | 確認したイオン | 文献 | 根拠 |
|---|---|---|---|---|
| retro_diels_alder_cyclohexene | 非芳香族シクロヘキセン → diene ＋ dienophile | butadiene+H 55（4-vinylcyclohexene） | [5, 2] | B |
| retro_diels_alder_benzo_fused | ベンゼン環に縮環した六員環（tetralin、chroman、tetrahydroisoquinoline など）の RDA | 192、165（tetrahydropalmatine） | [5, 20] | A |
| lactone_co2_loss | ラクトンから CO2 の脱離（2本切って開環） | [M+H−CO2]+ 103（coumarin） | [19] | A |
| lactone_co_loss | ラクトンから CO の脱離（環縮小を開環で表現） | [M+H−CO]+ 119（coumarin）、135（umbelliferone） | [19] | A |
| cyclic_ketone_co_loss | 環状ケトン・キノン型カルボニルから CO の脱離（環縮小を開環で表現） | [M+H−CO]+ 69（cyclohexanone） | [3, 2] | B |
| phenol_co_loss | フェノールの C–OH 炭素を CO として脱離（環縮小を開環で表現） | 107 = −2CO（umbelliferone、lactone_co_loss との2反応） | [3, 2] | B |

### 3-3. クラス限定
| セット / パターン | 内容 | 確認したイオン | 文献 | 根拠 |
|---|---|---|---|---|
| class_coumarin / coumarin_double_co_loss | ラクトンの CO と、ベンゼン環側の C–O の CO を両方失う | [M+H−2CO]+ 91（coumarin） | [19] | A |
| class_sulfonamide / sulfanilamide_s_n_cleavage | 4-アミノベンゼンスルホンアミドの S–N（m/z 156）と Ar–S（m/z 92） | 156、92、99（sulfamethoxazole） | [16, 17] | A |
| class_quinolone / quinolone_carboxylic_acid_losses | 4-キノロン-3-カルボン酸の H2O・CO2 の脱離 | 314、288（ciprofloxacin） | [13, 14] | A |
| class_quinolone / arylpiperazine_c2h5n_loss | アリールピペラジンの CH2–NH–CH2（C2H5N、43 Da）の脱離 | 245 = 288 − C2H5N（ciprofloxacin、2反応）。C2H5N という帰属は質量差からの推定 | [13, 14] | B |
| class_glucosinolate / glucosinolate_sulfate_cleavage | N–O–SO3 → HSO4−（m/z 97）／脱スルホ体 | 97（sinigrin、負イオン） | [21] | A |
| class_glucosinolate / glucosinolate_thioglucose_cleavage | C(=N)–S → チオグルコース陰イオン（m/z 195） | 195（sinigrin） | [21] | A |
| class_nucleoside / n_glycosidic_cleavage | N-グリコシド結合の開裂 → プロトン化塩基（糖 132/116 Da の脱離） | adenine+H 136（adenosine） | [22] | A |
| class_phospholipid / phosphocholine_head_group | ホスホコリン（m/z 184）／[M+H−183]+ | 184、577（POPC） | [23] | A |
| class_phospholipid / acyl_chain_loss | 脂肪酸（RCOOH）・ケテンの脱離、アシリウム | 生成のみ確認 | [23] | C |
| class_isoquinoline_alkaloid / benzylisoquinoline_benzyl_cleavage | C1–Cα（ベンジル）の開裂 | isoquinolinium 192、benzyl 137（reticuline） | [24] | A |
| class_isoquinoline_alkaloid / tetrahydroisoquinoline_amine_loss | 環の N を CH3NH2 などとして失う（開環） | [M+H−CH3NH2]+ 299（reticuline） | [24] | A |
| class_isoquinoline_alkaloid / tetrahydroisoquinoline_retro_diels_alder | テトラヒドロイソキノリン環の RDA（iminium ＋ o-キノジメタン） | 生成は4件。tetrahydropalmatine の 192 は汎用の benzo_fused RDA で説明 | [5, 20] | B |

### 3-4. requires_bond_formation（新しい結合が必要なため、現在は何も生成しない）
| パターン | 反応 | 文献で報告されているイオン | 文献 |
|---|---|---|---|
| sulfonyl_so2_extrusion | R1–SO2–R2 → R1–R2（ipso 置換を伴う転位） | sulfonamide の [M+H−SO2]+、負イオンでの SO2 の脱離 | [16, 17] |
| nitroarene_no_loss | Ar–NO2 → Ar–O（ニトロ–亜硝酸エステル転位）＋ NO | [M−H−NO]− 108（4-nitrophenol） | [18] |
| sulfanilyl_so_loss_m108 | Ar–SO2 → Ar–O（SO の脱離） | m/z 108（sulfonamide 抗菌薬） | [17] |
| cyclic_ketone_ring_contraction | 環縮小による CO の脱離（正確な形） | 同じ m/z は ring_cleavage/cyclic_ketone_co_loss で近似できる | [3] |

参照化合物でも、これらのイオンは「説明されないこと」を確認しています（sulfamethoxazole の 190・108、4-nitrophenol の 108、sinigrin の glucose sulfate 259）。使えるようにするには、エンジンに次の対応を加える必要があります。
- product の新しい結合を、反応の更新（bond update）として持つ。
- 残す原子を求めるとき、新しい結合でもつながっているとみなす。

## 4. まだ入れていない候補（今後の追加先）
- **ペプチド**：immonium イオン、側鎖の脱離（Met の CH3SOH など）[9]
- **トリテルペンサポニン、ステロイド配糖体**：糖鎖の逐次脱離。o_glycosidic_cleavage で一部は表現できる。
- **アントシアニジン（flavylium）**：flavonoid_exhaustive にあり。
- **ウレア・カルバメート**：HNCO・イソシアネートの脱離（C–N 開裂で一部は表現できる）。
- **Smiles 転位、ortho 効果**：結合の生成が必要。

## 参考文献
- [1] Holčapek M, Jirásko R, Lísa M. Basic rules for the interpretation of atmospheric pressure ionization mass spectra of small molecules. *J Chromatogr A* 2010;1217:3908–3921.
- [2] Demarque DP, Crotti AEM, Vessecchi R, Lopes JLC, Lopes NP. Fragmentation reactions using electrospray ionization mass spectrometry: an important tool for the structural elucidation and characterization of synthetic and natural products. *Nat Prod Rep* 2016;33:432–455. doi:10.1039/C5NP00073D
- [3] McLafferty FW, Tureček F. *Interpretation of Mass Spectra*, 4th ed. University Science Books, 1993.
- [4] McLafferty FW. Mass spectrometric analysis. Molecular rearrangements. *Anal Chem* 1959;31:82–87.
- [5] Tureček F, Hanuš V. Retro-Diels–Alder reaction in mass spectrometry. *Mass Spectrom Rev* 1984;3:85–152.
- [7] Domon B, Costello CE. A systematic nomenclature for carbohydrate fragmentations in FAB-MS/MS spectra of glycoconjugates. *Glycoconj J* 1988;5:397–409.
- [9] Paizs B, Suhai S. Fragmentation pathways of protonated peptides. *Mass Spectrom Rev* 2005;24:508–548. doi:10.1002/mas.20024
- [10] Roepstorff P, Fohlman J. Proposal for a common nomenclature for sequence ions in mass spectra of peptides. *Biomed Mass Spectrom* 1984;11:601.
- [11] Levsen K, Schiebel HM, Behnke B, Dötzer R, Dreher W, Elend M, Thiele H. Structure elucidation of phase II metabolites by tandem mass spectrometry: an overview. *J Chromatogr A* 2005;1067:55–72. doi:10.1016/j.chroma.2004.08.165
- [12] Clifford MN, Johnston KL, Knight S, Kuhnert N. Hierarchical scheme for LC-MSn identification of chlorogenic acids. *J Agric Food Chem* 2003;51:2900–2911. doi:10.1021/jf026187q
- [13] Volmer DA, Mansoori B, Locke SJ. Study of 4-quinolone antibiotics in biological samples by short-column liquid chromatography coupled with electrospray ionization tandem mass spectrometry. *Anal Chem* 1997;69:4143–4155.
- [14] Kovačević B, Schorr P, Qi Y, Volmer DA. *J Am Soc Mass Spectrom* 2014（piperazinyl quinolone の2種のプロトン化異性体と、カルボキシ基からの H2O・CO2 の脱離）。Neta P. et al. *Rapid Commun Mass Spectrom* 2010（H2O と CO2 の脱離比の構造・衝突エネルギー依存性）。**題名・頁は未確認**。
- [15] Cuyckens F, Claeys M. Mass spectrometry in the structural analysis of flavonoids. *J Mass Spectrom* 2004;39:1–15. doi:10.1002/jms.585
- [16] Sun M, Dai W, Liu DQ. Fragmentation of aromatic sulfonamides in electrospray ionization mass spectrometry: elimination of SO2 via rearrangement. *J Mass Spectrom* 2008;43:383–393. doi:10.1002/jms.1335
- [17] Hu N, Liu P, Jiang K, Zhou Y, Pan Y. Mechanism study of SO2 elimination from sulfonamides by negative electrospray ionization mass spectrometry. *Rapid Commun Mass Spectrom* 2008;22:2715–2722. doi:10.1002/rcm.3672
- [18] Schmidt AC, Herzschuh R, Matysik FM, Engewald W. Investigation of the ionisation and fragmentation behaviour of different nitroaromatic compounds occurring as polar metabolites of explosives using electrospray ionisation tandem mass spectrometry. *Rapid Commun Mass Spectrom* 2006;20:2293–2302. doi:10.1002/rcm.2591
- [19] Concannon S, Ramachandran VN, Smyth WF. A study of the electrospray ionisation of selected coumarin derivatives and their subsequent fragmentation using an ion trap mass spectrometer. *Rapid Commun Mass Spectrom* 2000;14:1157–1166.
- [20] Shim HJ, Lee JY. General fragmentations of alkaloids in electrospray ionization tandem mass spectrometry. *Mass Spectrom Lett* 2013;4:79–82. doi:10.5478/MSL.2013.4.4.79（tetrahydroprotoberberine の RDA）
- [21] Fabre N, Poinsot V, Debrauwer L, et al. Characterisation of glucosinolates using electrospray ion trap and electrospray quadrupole time-of-flight mass spectrometry. *Phytochem Anal* 2007;18. doi:10.1002/pca.983
- [22] Dudley E, Bond L. Mass spectrometry analysis of nucleosides and nucleotides. *Mass Spectrom Rev* 2014;33:302–331. doi:10.1002/mas.21388
- [23] Hsu FF, Turk J. Electrospray ionization/tandem quadrupole mass spectrometric studies on phosphatidylcholines: the fragmentation processes. *J Am Soc Mass Spectrom* 2003;14:352–363. doi:10.1016/S1044-0305(03)00064-3
- [24] Schmidt J, Raith K, Boettcher C, Zenk MH. Analysis of benzylisoquinoline-type alkaloids by electrospray tandem mass spectrometry and atmospheric pressure photoionization. *Eur J Mass Spectrom* 2005;11:325–333. doi:10.1255/ejms.745

**注意**：参考文献のうち 13、14、20、21 の頁や著者の一部は、要旨と書誌データベースでしか確認できていません。各パターンの「確認したイオン」は、文献の値から計算した理論 m/z を FREDIS が説明できるかの確認です。文献のスペクトルを一本ずつ照合したものではありません。
