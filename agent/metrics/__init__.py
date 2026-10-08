"""Quantitative metrics for the software-generation pipeline.

* ``style_diversity`` / ``blueprint_tree`` / ``vocab`` — structural heterogeneity
  of the SubStyleDesigner UI blueprints (Avalonia/XAML), via normalized "region
  skeleton" trees compared with Tree Edit Distance.
* ``built_diversity`` — the same structural machinery run on the *built* apps'
  ``.axaml`` views, plus blueprint→built structural fidelity.
* ``visual_fidelity`` — *visual* fidelity: parses the blueprint ``AppStyle.axaml``
  and the built app's resource dictionaries and scores how faithfully the build
  kept the design tokens (palette, primary, corner radii, fonts, token-binding
  vs hardcoded colors). Automated counterpart of the Coder 5c/8b visual contract.
"""
