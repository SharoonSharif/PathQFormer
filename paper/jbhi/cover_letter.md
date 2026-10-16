# Cover letter (paste as plain text into the cover-letter box of the IEEE Author Portal, https://ieee.atyponrex.com/journal/jbhi-embs; the box does not render Markdown)

To the Editor-in-Chief, IEEE Journal of Biomedical and Health Informatics

Title: One Checkpoint for Complete, Slide-Only and RNA-Only Patients: Auditing and Training Histology–Transcriptomics Survival Models for Missing Inputs

Article type: Regular paper (8 main-text pages in the IEEEtran journal format, plus one 5-page supplementary PDF uploaded as a supporting document; 13 pages in total, within the 14-page limit)

Authors: Sharoon Sharif (independent researcher; ORCID 0009-0003-2669-1209), corresponding author; Ha Can (independent researcher, biomedical sciences), second author

Summary. Survival models that fuse whole-slide images with transcriptomics are evaluated on patients who have both inputs, whereas a hospital cohort also holds slide-only and RNA-only patients, so the paper asks whether a fused model actually uses both inputs and what happens when one is missing at test time. On five TCGA cohorts under one equal-budget, three-seed protocol, an input-use audit (removal, permutation across patients, risk-score correlation) finds that the official SurvPath implementation, as trained under this protocol, ranks patients the same way when its RNA input is permuted (C-index change +0.001) but falls to a C-index of 0.47–0.52 without the slide, whereas a query-bottleneck fusion model trained with learned null codes, modality dropout and auxiliary unimodal heads responds to RNA, keeps a C-index of 0.54–0.62 with either input removed and moves at most 0.025 when half the patients lack a modality, so one trained checkpoint serves complete, slide-only and RNA-only patients. With both inputs present it does not detectably beat an RNA-only MLP or a late-fusion ensemble, which the paper states plainly: the recipe buys coverage of incomplete patients, not accuracy.

What is new. (1) An input-use audit of a published histology–transcriptomics co-attention model under an equal-budget protocol, showing that a high fused C-index does not demonstrate that both modalities are used; the audit (removal, permutation across patients, risk-score correlation) can be reported with any multimodal survival C-index. (2) Matched-variant evidence on where missing-modality robustness comes from: null codes with modality dropout remove the collapse without the slide, auxiliary unimodal heads make the fused prediction respond to RNA, and re-evaluating the same checkpoints under mean imputation instead of the null code restores a SurvPath-sized loss without the slide (0.030 to 0.069 for the variant without heads). (3) A one-checkpoint training recipe, assembled from existing components, that lets a single trained model accept complete, slide-only and RNA-only patients, with its accuracy against unimodal and late-fusion baselines reported in full. (4) Protocol evidence this literature rarely reports: seed-aware statistics (seed-averaged 25-fold paired tests with bootstrap confidence intervals and Holm correction) and a replay of checkpoint selection on the stored training histories, which shows that validation-loss early stopping would have selected epoch 1–3 checkpoints and cost the late-learning models 0.06–0.09 C-index on these event-poor folds.

Known gaps, stated in the paper. The variant without auxiliary heads was not evaluated under permutation (its removed-input rows are compared with the permuted rows of the other two models); the RNA-permuted loss that supports the auxiliary-head claim has p = 0.075 and 0.063 over 25 folds; SurvPath was not re-tuned under our protocol; and all results are on TCGA with frozen UNI2-h features.

Fit with JBHI. The paper concerns how models that join whole-slide images with transcriptomics behave on the incomplete patient records a hospital cohort holds, and it offers an audit that any such model reported to this readership can include; it falls within the journal's pathology-informatics and bioinformatics scope. All results are on public TCGA data and the paper makes no claim of clinical readiness. Code, configurations, archived per-run results and all trained checkpoints are public, so every table can be regenerated.

Originality. The manuscript is entirely original; it has not been copyrighted, published, submitted or accepted for publication elsewhere, and it is not under consideration by any journal, conference or online venue. For completeness: an earlier version was submitted to Transactions on Machine Learning Research and desk-rejected on 30 September 2026 without being sent to reviewers; it has since been revised and re-framed for a health-informatics readership, and no version has been posted as a preprint or published in any form.

Data and code availability. TCGA data are public and de-identified; the UNI2-h patch features and the SurvPath RNA-seq matrices, metadata and splits are available from their authors. The code (MIT license), configuration files, archived per-run results and the script that regenerates every table are public at https://github.com/SharoonSharif/PathQFormer (release v0.3.0) and archived at Zenodo, https://doi.org/10.5281/zenodo.23245611 (version DOI; concept DOI 10.5281/zenodo.22900123). The 275 trained fold models are on the Hugging Face Hub at https://huggingface.co/sharoonsharif1/PathQFormer-checkpoints. The repository, its v0.3.0 release and Zenodo archive, and the checkpoint repository carry the model's earlier name, PathQ-Former; the GitHub README records the change.

Use of AI tools. A large language model (Claude, Anthropic) assisted with drafting and editing the text and LaTeX source and with writing analysis and plotting code, under the authors' direction; S. Sharif designed the study, ran the experiments and checked every reported number against the archived results, H. Can contributed biomedical domain expertise, and both authors are responsible for all content. This is also stated in the Acknowledgment.

Conflicts of interest and funding. The authors declare no competing interests. The work received no dedicated funding; cloud compute was paid for by S. Sharif. Author contributions: S. Sharif designed the study, ran the experiments and drafted the manuscript; H. Can contributed biomedical domain expertise to the study design and the interpretation of the results.

Suggested reviewer expertise (areas, not names).
1. Computational pathology and multiple-instance learning on whole-slide images.
2. Multimodal fusion of histology and genomics for survival prediction (co-attention models such as MCAT and SurvPath).
3. Missing-modality and robust multimodal learning (modality dropout, shared–specific representations).
4. Survival-analysis methodology and evaluation (concordance indices, IPCW metrics, Brier score, event-poor cross-validation).
5. Reproducibility and benchmarking practice in medical machine learning.

Thank you for considering the manuscript.

Sharoon Sharif
Independent researcher
sharoonsharif@gmail.com
