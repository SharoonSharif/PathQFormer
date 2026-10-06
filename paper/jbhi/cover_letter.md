# Cover letter (for the ScholarOne cover-letter box)

To the Editor-in-Chief, IEEE Journal of Biomedical and Health Informatics

**Title:** One Checkpoint for Complete, Slide-Only and RNA-Only Patients: Auditing and Training Histology–Transcriptomics Survival Models for Missing Inputs

**Article type:** Regular paper (9 main-text pages plus a 5-page supplementary PDF)

**Author:** Sharoon Sharif (Alabama A&M University; Faith Comes By Hearing), sole and corresponding author

**Summary.** Survival models that fuse whole-slide images with transcriptomics are evaluated on patients who have both inputs, whereas in a hospital cohort some patients have a slide, some RNA and some both. The paper asks two questions such a cohort raises and that an accuracy table cannot answer: does a fused model actually use both inputs, and what happens when one is missing at test time? On five TCGA cohorts under one equal-budget, three-seed protocol, an input-use audit (removal, permutation across patients, risk-score correlation) shows that the official SurvPath implementation, as trained under this protocol, ranks patients the same way when its RNA input is permuted (C-index change +0.001) but collapses to 0.47–0.52 without the slide. A query-bottleneck fusion model trained with learned null codes and modality dropout no longer collapses without the slide, and adding auxiliary unimodal heads makes its fused prediction respond to RNA; the resulting single checkpoint keeps a C-index of 0.54–0.62 with either input removed and moves at most 0.025 when half the patients lack a modality, so one trained model can serve complete, slide-only and RNA-only patients. With both inputs it does not detectably beat an RNA-only MLP or a late-fusion ensemble, which the paper states plainly: the recipe buys coverage of incomplete patients, not accuracy. A replay of checkpoint selection on the stored training histories shows that validation-loss early stopping would have selected epoch 1–3 checkpoints and cost the late-learning models 0.06–0.09 C-index on these event-poor folds.

**What is new.** (1) An input-use audit of a published co-attention model under an equal-budget protocol, with removal, permutation and risk-correlation tests that any multimodal survival model can report. (2) Matched-variant evidence on where missing-modality robustness comes from: null codes and modality dropout remove the collapse without the slide, auxiliary unimodal heads make the fused output respond to RNA, and mean imputation on the same checkpoints reproduces the collapse. (3) A one-checkpoint training recipe, assembled from existing components, that lets a single trained model accept complete, slide-only and RNA-only patients, with its accuracy against unimodal and late-fusion baselines reported honestly. (4) Seed-aware statistics (seed-averaged 25-fold paired tests with bootstrap confidence intervals and Holm correction) and a post-hoc replay of checkpoint selection from stored histories, both rarely reported in this literature.

**Originality.** The manuscript is not published and is not under review elsewhere. A previous version was submitted to Transactions on Machine Learning Research and was desk-rejected on 30 September 2026; the manuscript has since been revised and re-framed for a health-informatics readership, and no part of it has appeared in any venue.

**Data and code availability.** TCGA data are public and de-identified; the UNI2-h patch features and the SurvPath RNA-seq matrices, metadata and splits are available from their authors. The code (MIT licence), configuration files, archived per-run results and the script that regenerates every table are public at https://github.com/SharoonSharif/PathQFormer and archived at Zenodo (DOI 10.5281/zenodo.22900123); the 275 trained fold models are on the Hugging Face Hub at https://huggingface.co/sharoonsharif1/PathQFormer-checkpoints.

**Use of AI tools.** A large language model (Claude, Anthropic) assisted with drafting and editing the text and LaTeX source and with writing analysis and plotting code, under the author's direction; the author designed the study, ran the experiments, checked every reported number against the archived results, and is responsible for all content. This is also stated in the Acknowledgment.

**Conflicts of interest and funding.** The author declares no competing interests. The work received no dedicated funding; cloud compute was paid for by the author.

**Suggested reviewer expertise (areas, not names).**
1. Computational pathology and multiple-instance learning on whole-slide images.
2. Multimodal fusion of histology and genomics for survival prediction (co-attention models such as MCAT/SurvPath and their successors).
3. Missing-modality and robust multimodal learning (modality dropout, shared–specific representations).
4. Survival analysis methodology and evaluation (concordance indices, IPCW metrics, Brier score, event-poor cross-validation).
5. Reproducibility and benchmarking practice in medical machine learning.

Thank you for considering the manuscript.

Sharoon Sharif
