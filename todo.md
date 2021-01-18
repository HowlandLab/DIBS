# TODO: Wishlist of things to implement (unordered)
- [ ] *** Pipeline: do a check to see how many NAN values get passed from stage to stage...im guessing the binning helps us solve that, but does adaptive filtering actually solve it first?
- [ ] *** Further container-ize steps in build process s.t. TSNE can be evaluated without training classifier so evaluate clustering
- [ ] In the diagnostics section, have a button that produces a *HISTORGRAM* of the lengths of actions distributions (1 histogram per action...? Seems that way for now...)
- [ ] GMM plot should modify labels to match behaviours instead of just assignment number (WHEN POSSIBLE)
- [ ] Easy: Killian: rename internal vars for Pipelines from _, scaled to unscaled, _
- [ ] Add a button for exporting (all) data
- [ ] Test: ensure new/variable features selected
- [ ] Confusion matrix for Streamlit diagnostics section (Aaron)
- [ ] Add PCA plotting for training data for BasePipeline to Streamlit diagnostics section
- [ ] Aaron: add math for snout/tail body angle in feature_engi
- [ ] Lower priority: Multiple features select on the Streamlit app model building section

# Done
- [x] Add colour to video assignment
- [x] Add binning feature slider
- [x] Add FPS input into params
- [x] Add colour to text in example behaviour videos
- [x] Add sidebar checkbox for extra, explanatory text
