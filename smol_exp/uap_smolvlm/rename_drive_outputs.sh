#!/bin/bash
# Run inside the folder that holds the saved results (the uap_smolvlm folder from the zip / Drive).
# Renames the old result files so the cleaned notebooks find them.
mv_if() { [ -e "$1" ] && mv -v "$1" "$2"; }
mv_if week1_results.json uap_training_results.json
mv_if week1_v.png uap_training_perturbation.png
mv_if week1_examples.png uap_training_examples.png
mv_if week2_step1_results.json pipeline_prompts_results.json
mv_if week2_partA_prompts.png prompt_dependence.png
mv_if week2_partB_pipeline.png real_pipeline.png
mv_if week2_partB_examples.png real_pipeline_examples.png
mv_if week2_step2_results.json tiles_transfer_results.json
mv_if week2_step2_figures.png tiles_transfer_figures.png
mv_if week2_step3_eps_sweep.json eps_sweep_results.json
mv_if week2_eps_sweep.png eps_sweep.png
mv_if week2_eps_visibility.png eps_visibility.png
mv_if week3_defenses_testtime.json test_time_defenses_results.json
mv_if week3_defenses_testtime.png test_time_defenses.png
mv_if week3_strong_adaptive.json strong_adaptive_results.json
mv_if week3_strong_adaptive.png strong_adaptive.png
mv_if week3_visibility_flat.png visibility_flat_region.png
mv_if week3_finetune_and_rrp.json finetune_and_rrp_results.json
mv_if week3_defense_ranking.png defense_ranking.png
