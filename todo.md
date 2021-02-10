# TODO: Wishlist of things to implement (unordered)
- [ ] Easy: Add a button for exporting (all) data
- [ ] Med/easy: *** Pipeline: do a check to see how many NAN values get passed from stage to stage...im guessing the binning helps us solve that, but does adaptive filtering actually solve it first?
- [ ] Hard: *** Further container-ize steps in build process s.t. TSNE can be evaluated without training classifier so evaluate clustering
- [ ] Med: In the diagnostics section, have a button that produces a *HISTOGRAM* of the lengths of actions distributions (1 histogram per action...? Seems that way for now...)
- [ ] Med/easy: GMM plot should modify labels to match behaviours instead of just assignment number (WHEN POSSIBLE)
- [ ] Easy: rename internal vars for Pipelines from _, scaled to unscaled, _
- [ ] Med: Confusion matrix for Streamlit diagnostics section
- [ ] Med: Add PCA plotting for training data for BasePipeline to Streamlit diagnostics section
- [ ] Hard: Lower priority: Multiple features select on the Streamlit app model building section

# Done
- [x] Add math for vector(part1,part2) delta angle
- [x] Add colour to video assignment
- [x] Add binning feature slider
- [x] Add FPS input into params
- [x] Add colour to text in example behaviour videos
- [x] Add sidebar checkbox for extra, explanatory text
- [x] Change min-max scaling from gaussian to 0-1
