# jevmlx eval report

## Environment

| key | value |
| --- | --- |
| chip | Apple M5 Max |
| git_sha | dbb1ff1f04a0bce4b20817e4c09eae2b5a3d2ad3 |
| jevmlx_version | 0.1.0 |
| machine_model | Mac17,7 |
| macos_version | 26.6.2 |
| mlx_lm_version | 0.31.3 |
| mlx_version | 0.32.2 |
| python_version | 3.12.14 |
| ram_gb | 128.0000 |
| timestamp_utc | 2026-09-21T04:20:46+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.6322 [0.6083, 0.6456] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.8601 |
| exact record | 0.1136 |
| case_exact_match | 0.1111 |
| brier | 0.8337 [0.8142, 0.8785] (case_cluster_bootstrap) |
| correctness_auroc | 0.7506 |
| ece_5bin_equal_mass | 0.1200 |
| tie_rate | 0.0025 |
| agreement[agreement_common_subset] | 0.6322 |
| agreement[n_cases] | 44 |
| agreement[n_fields] | 14847 |
| agreement[overall] | 0.6322 |
| any_flip_rate[activity_ongoing] | 0.0000 |
| any_flip_rate[adjustment_duplicates_line] | 0.0000 |
| any_flip_rate[affected_scope] | 0.2500 |
| any_flip_rate[already_compensated] | 0.0909 |
| any_flip_rate[amount_vs_record] | 0.3182 |
| any_flip_rate[approval_0] | 0.0764 |
| any_flip_rate[attack_type] | 0.3750 |
| any_flip_rate[attacker_modified_configuration] | 0.0000 |
| any_flip_rate[attacker_persistence_present] | 0.0000 |
| any_flip_rate[attribution] | 0.7500 |
| any_flip_rate[bank_change_claimed_in_comms] | 0.0000 |
| any_flip_rate[billed_above_basis] | 0.0000 |
| any_flip_rate[cancellation_reason] | 0.2857 |
| any_flip_rate[changes_terms] | 0.0000 |
| any_flip_rate[churn_risk] | 0.5000 |
| any_flip_rate[claims_agent_error] | 0.0000 |
| any_flip_rate[context_explains_activity] | 0.0000 |
| any_flip_rate[credentials_exposed] | 0.0000 |
| any_flip_rate[desired_outcome] | 0.2353 |
| any_flip_rate[different_entity] | 0.0000 |
| any_flip_rate[evidence_strength] | 0.6000 |
| any_flip_rate[expressed_satisfaction] | 0.8667 |
| any_flip_rate[first_bad_step] | 0.2500 |
| any_flip_rate[frustration] | 0.2235 |
| any_flip_rate[hardship] | 0.4773 |
| any_flip_rate[in_scope] | 0.0000 |
| any_flip_rate[intent] | 0.2353 |
| any_flip_rate[intent_pair] | 0.0000 |
| any_flip_rate[is_true_positive] | 0.1200 |
| any_flip_rate[issue_resolved] | 0.0588 |
| any_flip_rate[line_0_completion] | 0.3827 |
| any_flip_rate[line_0_kind] | 0.1204 |
| any_flip_rate[line_0_owner_declined] | 0.0000 |
| any_flip_rate[line_0_rate_differs] | 0.0000 |
| any_flip_rate[line_0_rebilled] | 0.0741 |
| any_flip_rate[line_0_scope] | 0.3302 |
| any_flip_rate[line_0_unexplained_fee] | 0.0000 |
| any_flip_rate[line_1_completion] | 0.5401 |
| any_flip_rate[line_1_kind] | 0.1265 |
| any_flip_rate[line_1_owner_declined] | 0.0000 |
| any_flip_rate[line_1_rate_differs] | 0.0000 |
| any_flip_rate[line_1_rebilled] | 0.0154 |
| any_flip_rate[line_1_scope] | 0.4352 |
| any_flip_rate[line_1_unexplained_fee] | 0.0062 |
| any_flip_rate[line_2_completion] | 0.3878 |
| any_flip_rate[line_2_kind] | 0.1510 |
| any_flip_rate[line_2_owner_declined] | 0.0000 |
| any_flip_rate[line_2_rate_differs] | 0.0163 |
| any_flip_rate[line_2_rebilled] | 0.0163 |
| any_flip_rate[line_2_scope] | 0.4571 |
| any_flip_rate[line_2_unexplained_fee] | 0.0327 |
| any_flip_rate[line_3_completion] | 0.4571 |
| any_flip_rate[line_3_kind] | 0.1388 |
| any_flip_rate[line_3_owner_declined] | 0.0000 |
| any_flip_rate[line_3_rate_differs] | 0.0041 |
| any_flip_rate[line_3_rebilled] | 0.0082 |
| any_flip_rate[line_3_scope] | 0.4367 |
| any_flip_rate[line_3_unexplained_fee] | 0.0449 |
| any_flip_rate[line_4_completion] | 0.3523 |
| any_flip_rate[line_4_kind] | 0.0909 |
| any_flip_rate[line_4_owner_declined] | 0.0000 |
| any_flip_rate[line_4_rate_differs] | 0.0000 |
| any_flip_rate[line_4_rebilled] | 0.0000 |
| any_flip_rate[line_4_scope] | 0.7955 |
| any_flip_rate[line_4_unexplained_fee] | 0.0455 |
| any_flip_rate[malicious_content_in_mailboxes] | 0.0000 |
| any_flip_rate[malicious_process_running] | 0.0625 |
| any_flip_rate[offers_evidence] | 0.0000 |
| any_flip_rate[open_to_offer] | 0.0000 |
| any_flip_rate[outbound_channel_active] | 0.0000 |
| any_flip_rate[price_basis] | 0.0988 |
| any_flip_rate[prior_0] | 0.0625 |
| any_flip_rate[prior_1] | 0.1458 |
| any_flip_rate[prior_2] | 0.2082 |
| any_flip_rate[prior_3] | 0.1000 |
| any_flip_rate[proposal_reply] | 1.0000 |
| any_flip_rate[refund_reason] | 0.2273 |
| any_flip_rate[reports_unauthorized] | 0.0824 |
| any_flip_rate[reports_unresolved] | 0.0000 |
| any_flip_rate[request_specificity] | 1.0000 |
| any_flip_rate[requests_human] | 0.0118 |
| any_flip_rate[sender_0] | 0.0311 |
| any_flip_rate[session_in_attacker_hands] | 0.0000 |
| any_flip_rate[shares_credentials] | 0.0000 |
| any_flip_rate[spread_beyond_initial_entity] | 0.0000 |
| any_flip_rate[statement_not_invoice] | 0.0000 |
| any_flip_rate[tax_two_rates] | 0.0000 |
| any_flip_rate[threat_chargeback_or_public] | 0.0000 |
| any_flip_rate[threat_legal_regulatory] | 0.0000 |
| any_flip_rate[too_ambiguous] | 0.0000 |
| any_flip_rate[unexplained_charges] | 0.0000 |
| any_flip_rate[unusual_urgency] | 0.0278 |
| any_flip_rate[urgency] | 0.2471 |
| balanced_accuracy[activity_ongoing] | 1.0000 |
| balanced_accuracy[adjustment_duplicates_line] | 0.0000 |
| balanced_accuracy[affected_scope] | 0.4444 |
| balanced_accuracy[already_compensated] | 0.9167 |
| balanced_accuracy[amount_vs_record] | 0.0625 |
| balanced_accuracy[approval_0] | 0.4406 |
| balanced_accuracy[attack_type] | 0.1111 |
| balanced_accuracy[attacker_modified_configuration] | 0.0000 |
| balanced_accuracy[attacker_persistence_present] | 0.5000 |
| balanced_accuracy[attribution] | 0.2262 |
| balanced_accuracy[bank_change_claimed_in_comms] | 0.5000 |
| balanced_accuracy[billed_above_basis] | 0.5000 |
| balanced_accuracy[cancellation_reason] | 0.3750 |
| balanced_accuracy[changes_terms] | 1.0000 |
| balanced_accuracy[churn_risk] | 0.1875 |
| balanced_accuracy[claims_agent_error] | 0.7500 |
| balanced_accuracy[claims_supported] | 0.2500 |
| balanced_accuracy[context_explains_activity] | 0.2667 |
| balanced_accuracy[credentials_exposed] | 0.5000 |
| balanced_accuracy[desired_outcome] | 0.4136 |
| balanced_accuracy[different_entity] | 0.5000 |
| balanced_accuracy[evidence_strength] | 0.4583 |
| balanced_accuracy[expressed_satisfaction] | 0.1562 |
| balanced_accuracy[first_bad_step] | 0.4048 |
| balanced_accuracy[frustration] | 0.4667 |
| balanced_accuracy[handed_off] | 0.8750 |
| balanced_accuracy[handoff_required] | 0.7500 |
| balanced_accuracy[hardship] | 0.1528 |
| balanced_accuracy[in_scope] | 1.0000 |
| balanced_accuracy[instructed_by_tool_output] | 1.0000 |
| balanced_accuracy[intent] | 0.4444 |
| balanced_accuracy[intent_pair] | 0.0000 |
| balanced_accuracy[is_true_positive] | 0.2330 |
| balanced_accuracy[issue_resolved] | 0.8556 |
| balanced_accuracy[left_undone] | 0.5833 |
| balanced_accuracy[line_0_completion] | 0.0543 |
| balanced_accuracy[line_0_kind] | 0.0426 |
| balanced_accuracy[line_0_owner_declined] | 1.0000 |
| balanced_accuracy[line_0_rate_differs] | 1.0000 |
| balanced_accuracy[line_0_rebilled] | 0.9271 |
| balanced_accuracy[line_0_scope] | 0.2857 |
| balanced_accuracy[line_0_unexplained_fee] | 1.0000 |
| balanced_accuracy[line_1_completion] | 0.1780 |
| balanced_accuracy[line_1_kind] | 0.0578 |
| balanced_accuracy[line_1_owner_declined] | 1.0000 |
| balanced_accuracy[line_1_rate_differs] | 1.0000 |
| balanced_accuracy[line_1_rebilled] | 0.9848 |
| balanced_accuracy[line_1_scope] | 0.2614 |
| balanced_accuracy[line_1_unexplained_fee] | 0.9939 |
| balanced_accuracy[line_2_completion] | 0.0766 |
| balanced_accuracy[line_2_kind] | 0.0282 |
| balanced_accuracy[line_2_owner_declined] | 1.0000 |
| balanced_accuracy[line_2_rate_differs] | 0.9839 |
| balanced_accuracy[line_2_rebilled] | 0.9839 |
| balanced_accuracy[line_2_scope] | 0.0766 |
| balanced_accuracy[line_2_unexplained_fee] | 0.6492 |
| balanced_accuracy[line_3_completion] | 0.0898 |
| balanced_accuracy[line_3_kind] | 0.0281 |
| balanced_accuracy[line_3_owner_declined] | 1.0000 |
| balanced_accuracy[line_3_rate_differs] | 0.9960 |
| balanced_accuracy[line_3_rebilled] | 0.9919 |
| balanced_accuracy[line_3_scope] | 0.3306 |
| balanced_accuracy[line_3_unexplained_fee] | 0.2741 |
| balanced_accuracy[line_4_completion] | 0.0337 |
| balanced_accuracy[line_4_kind] | 0.0337 |
| balanced_accuracy[line_4_owner_declined] | 1.0000 |
| balanced_accuracy[line_4_rate_differs] | 1.0000 |
| balanced_accuracy[line_4_rebilled] | 1.0000 |
| balanced_accuracy[line_4_scope] | 0.1798 |
| balanced_accuracy[line_4_unexplained_fee] | 0.9551 |
| balanced_accuracy[malicious_content_in_mailboxes] | 1.0000 |
| balanced_accuracy[malicious_process_running] | 0.9444 |
| balanced_accuracy[offers_evidence] | 0.5000 |
| balanced_accuracy[open_to_offer] | 1.0000 |
| balanced_accuracy[outbound_channel_active] | 1.0000 |
| balanced_accuracy[price_basis] | 0.0340 |
| balanced_accuracy[prior_0] | 0.8048 |
| balanced_accuracy[prior_1] | 0.8562 |
| balanced_accuracy[prior_2] | 0.7944 |
| balanced_accuracy[prior_3] | 0.9012 |
| balanced_accuracy[proposal_reply] | 0.6667 |
| balanced_accuracy[refund_done_msg_0] | 1.0000 |
| balanced_accuracy[refund_done_msg_1] | 0.5000 |
| balanced_accuracy[refund_reason] | 0.3333 |
| balanced_accuracy[reports_unauthorized] | 0.7222 |
| balanced_accuracy[reports_unresolved] | 0.8333 |
| balanced_accuracy[request_fulfilled] | 0.3333 |
| balanced_accuracy[request_specificity] | 0.5000 |
| balanced_accuracy[requests_human] | 0.7407 |
| balanced_accuracy[safety_1__instructed_by_tool_output__1] | 1.0000 |
| balanced_accuracy[safety_1__user_asked_or_agreed__1] | 1.0000 |
| balanced_accuracy[safety_1__within_grant__1] | 1.0000 |
| balanced_accuracy[safety_2__instructed_by_tool_output__2] | 1.0000 |
| balanced_accuracy[safety_2__user_asked_or_agreed__2] | 1.0000 |
| balanced_accuracy[safety_2__within_grant__2] | 1.0000 |
| balanced_accuracy[secured_msg_0] | 1.0000 |
| balanced_accuracy[secured_msg_1] | 1.0000 |
| balanced_accuracy[sender_0] | 0.8337 |
| balanced_accuracy[session_in_attacker_hands] | 0.0000 |
| balanced_accuracy[shares_credentials] | 1.0000 |
| balanced_accuracy[spread_beyond_initial_entity] | 0.5000 |
| balanced_accuracy[statement_not_invoice] | 1.0000 |
| balanced_accuracy[tax_two_rates] | 1.0000 |
| balanced_accuracy[threat_chargeback_or_public] | 0.6000 |
| balanced_accuracy[threat_legal_regulatory] | 1.0000 |
| balanced_accuracy[too_ambiguous] | 0.5000 |
| balanced_accuracy[unexplained_charges] | 0.5000 |
| balanced_accuracy[unusual_urgency] | 0.9726 |
| balanced_accuracy[urgency] | 0.1111 |
| balanced_accuracy[user_asked_or_agreed] | 1.0000 |
| balanced_accuracy[within_grant] | 1.0000 |
| macro_f1[activity_ongoing] | 1.0000 |
| macro_f1[adjustment_duplicates_line] | 0.0000 |
| macro_f1[affected_scope] | 0.3953 |
| macro_f1[already_compensated] | 0.9565 |
| macro_f1[amount_vs_record] | 0.1176 |
| macro_f1[approval_0] | 0.4953 |
| macro_f1[attack_type] | 0.2000 |
| macro_f1[attacker_modified_configuration] | 0.0000 |
| macro_f1[attacker_persistence_present] | 0.3333 |
| macro_f1[attribution] | 0.2967 |
| macro_f1[bank_change_claimed_in_comms] | 0.4328 |
| macro_f1[billed_above_basis] | 0.3827 |
| macro_f1[cancellation_reason] | 0.5455 |
| macro_f1[changes_terms] | 1.0000 |
| macro_f1[churn_risk] | 0.3158 |
| macro_f1[claims_agent_error] | 0.7619 |
| macro_f1[claims_supported] | 0.1667 |
| macro_f1[context_explains_activity] | 0.4211 |
| macro_f1[credentials_exposed] | 0.3333 |
| macro_f1[desired_outcome] | 0.4602 |
| macro_f1[different_entity] | 0.4702 |
| macro_f1[evidence_strength] | 0.5401 |
| macro_f1[expressed_satisfaction] | 0.1627 |
| macro_f1[first_bad_step] | 0.5060 |
| macro_f1[frustration] | 0.4026 |
| macro_f1[handed_off] | 0.7619 |
| macro_f1[handoff_required] | 0.7619 |
| macro_f1[hardship] | 0.2069 |
| macro_f1[in_scope] | 1.0000 |
| macro_f1[instructed_by_tool_output] | 1.0000 |
| macro_f1[intent] | 0.3398 |
| macro_f1[intent_pair] | 0.0000 |
| macro_f1[is_true_positive] | 0.1657 |
| macro_f1[issue_resolved] | 0.9222 |
| macro_f1[left_undone] | 0.5833 |
| macro_f1[line_0_completion] | 0.1027 |
| macro_f1[line_0_kind] | 0.0816 |
| macro_f1[line_0_owner_declined] | 1.0000 |
| macro_f1[line_0_rate_differs] | 1.0000 |
| macro_f1[line_0_rebilled] | 0.9621 |
| macro_f1[line_0_scope] | 0.4444 |
| macro_f1[line_0_unexplained_fee] | 1.0000 |
| macro_f1[line_1_completion] | 0.2702 |
| macro_f1[line_1_kind] | 0.1092 |
| macro_f1[line_1_owner_declined] | 1.0000 |
| macro_f1[line_1_rate_differs] | 1.0000 |
| macro_f1[line_1_rebilled] | 0.9923 |
| macro_f1[line_1_scope] | 0.4145 |
| macro_f1[line_1_unexplained_fee] | 0.9970 |
| macro_f1[line_2_completion] | 0.1423 |
| macro_f1[line_2_kind] | 0.0549 |
| macro_f1[line_2_owner_declined] | 1.0000 |
| macro_f1[line_2_rate_differs] | 0.9919 |
| macro_f1[line_2_rebilled] | 0.9919 |
| macro_f1[line_2_scope] | 0.1423 |
| macro_f1[line_2_unexplained_fee] | 0.7873 |
| macro_f1[line_3_completion] | 0.1224 |
| macro_f1[line_3_kind] | 0.0516 |
| macro_f1[line_3_owner_declined] | 1.0000 |
| macro_f1[line_3_rate_differs] | 0.9980 |
| macro_f1[line_3_rebilled] | 0.9960 |
| macro_f1[line_3_scope] | 0.4970 |
| macro_f1[line_3_unexplained_fee] | 0.2756 |
| macro_f1[line_4_completion] | 0.0652 |
| macro_f1[line_4_kind] | 0.0652 |
| macro_f1[line_4_owner_declined] | 1.0000 |
| macro_f1[line_4_rate_differs] | 1.0000 |
| macro_f1[line_4_rebilled] | 1.0000 |
| macro_f1[line_4_scope] | 0.3048 |
| macro_f1[line_4_unexplained_fee] | 0.9770 |
| macro_f1[malicious_content_in_mailboxes] | 1.0000 |
| macro_f1[malicious_process_running] | 0.9714 |
| macro_f1[offers_evidence] | 0.4286 |
| macro_f1[open_to_offer] | 1.0000 |
| macro_f1[outbound_channel_active] | 1.0000 |
| macro_f1[price_basis] | 0.0592 |
| macro_f1[prior_0] | 0.8918 |
| macro_f1[prior_1] | 0.9225 |
| macro_f1[prior_2] | 0.8854 |
| macro_f1[prior_3] | 0.9481 |
| macro_f1[proposal_reply] | 0.8000 |
| macro_f1[refund_done_msg_0] | 1.0000 |
| macro_f1[refund_done_msg_1] | 0.3333 |
| macro_f1[refund_reason] | 0.3450 |
| macro_f1[reports_unauthorized] | 0.8387 |
| macro_f1[reports_unresolved] | 0.8000 |
| macro_f1[request_fulfilled] | 0.2857 |
| macro_f1[request_specificity] | 0.6667 |
| macro_f1[requests_human] | 0.7513 |
| macro_f1[safety_1__instructed_by_tool_output__1] | 1.0000 |
| macro_f1[safety_1__user_asked_or_agreed__1] | 1.0000 |
| macro_f1[safety_1__within_grant__1] | 1.0000 |
| macro_f1[safety_2__instructed_by_tool_output__2] | 1.0000 |
| macro_f1[safety_2__user_asked_or_agreed__2] | 1.0000 |
| macro_f1[safety_2__within_grant__2] | 1.0000 |
| macro_f1[secured_msg_0] | 1.0000 |
| macro_f1[secured_msg_1] | 1.0000 |
| macro_f1[sender_0] | 0.8112 |
| macro_f1[session_in_attacker_hands] | 0.0000 |
| macro_f1[shares_credentials] | 1.0000 |
| macro_f1[spread_beyond_initial_entity] | 0.3333 |
| macro_f1[statement_not_invoice] | 1.0000 |
| macro_f1[tax_two_rates] | 1.0000 |
| macro_f1[threat_chargeback_or_public] | 0.7500 |
| macro_f1[threat_legal_regulatory] | 1.0000 |
| macro_f1[too_ambiguous] | 0.4298 |
| macro_f1[unexplained_charges] | 0.4328 |
| macro_f1[unusual_urgency] | 0.9861 |
| macro_f1[urgency] | 0.0941 |
| macro_f1[user_asked_or_agreed] | 1.0000 |
| macro_f1[within_grant] | 1.0000 |
| mean_tvd[activity_ongoing] | 0.0337 |
| mean_tvd[adjustment_duplicates_line] | 0.0007 |
| mean_tvd[affected_scope] | 0.2563 |
| mean_tvd[already_compensated] | 0.0716 |
| mean_tvd[amount_vs_record] | 0.2554 |
| mean_tvd[approval_0] | 0.0879 |
| mean_tvd[attack_type] | 0.1555 |
| mean_tvd[attacker_modified_configuration] | 0.0163 |
| mean_tvd[attacker_persistence_present] | 0.0113 |
| mean_tvd[attribution] | 0.7011 |
| mean_tvd[bank_change_claimed_in_comms] | 0.0075 |
| mean_tvd[billed_above_basis] | 0.0092 |
| mean_tvd[cancellation_reason] | 0.3292 |
| mean_tvd[changes_terms] | 0.0108 |
| mean_tvd[churn_risk] | 0.4896 |
| mean_tvd[claims_agent_error] | 0.0109 |
| mean_tvd[context_explains_activity] | 0.0517 |
| mean_tvd[credentials_exposed] | 0.0032 |
| mean_tvd[desired_outcome] | 0.2137 |
| mean_tvd[different_entity] | 0.0100 |
| mean_tvd[evidence_strength] | 0.5120 |
| mean_tvd[expressed_satisfaction] | 0.5374 |
| mean_tvd[first_bad_step] | 0.1677 |
| mean_tvd[frustration] | 0.2027 |
| mean_tvd[hardship] | 0.4063 |
| mean_tvd[in_scope] | 0.0864 |
| mean_tvd[intent] | 0.2751 |
| mean_tvd[intent_pair] | 0.0006 |
| mean_tvd[is_true_positive] | 0.0532 |
| mean_tvd[issue_resolved] | 0.0267 |
| mean_tvd[line_0_completion] | 0.1904 |
| mean_tvd[line_0_kind] | 0.1164 |
| mean_tvd[line_0_owner_declined] | 0.0017 |
| mean_tvd[line_0_rate_differs] | 0.0138 |
| mean_tvd[line_0_rebilled] | 0.0267 |
| mean_tvd[line_0_scope] | 0.1503 |
| mean_tvd[line_0_unexplained_fee] | 0.0129 |
| mean_tvd[line_1_completion] | 0.1882 |
| mean_tvd[line_1_kind] | 0.1151 |
| mean_tvd[line_1_owner_declined] | 0.0041 |
| mean_tvd[line_1_rate_differs] | 0.0183 |
| mean_tvd[line_1_rebilled] | 0.0283 |
| mean_tvd[line_1_scope] | 0.1495 |
| mean_tvd[line_1_unexplained_fee] | 0.0233 |
| mean_tvd[line_2_completion] | 0.1888 |
| mean_tvd[line_2_kind] | 0.1396 |
| mean_tvd[line_2_owner_declined] | 0.0036 |
| mean_tvd[line_2_rate_differs] | 0.0198 |
| mean_tvd[line_2_rebilled] | 0.0249 |
| mean_tvd[line_2_scope] | 0.1713 |
| mean_tvd[line_2_unexplained_fee] | 0.0357 |
| mean_tvd[line_3_completion] | 0.1913 |
| mean_tvd[line_3_kind] | 0.1331 |
| mean_tvd[line_3_owner_declined] | 0.0037 |
| mean_tvd[line_3_rate_differs] | 0.0195 |
| mean_tvd[line_3_rebilled] | 0.0265 |
| mean_tvd[line_3_scope] | 0.1769 |
| mean_tvd[line_3_unexplained_fee] | 0.0338 |
| mean_tvd[line_4_completion] | 0.2003 |
| mean_tvd[line_4_kind] | 0.1132 |
| mean_tvd[line_4_owner_declined] | 0.0037 |
| mean_tvd[line_4_rate_differs] | 0.0216 |
| mean_tvd[line_4_rebilled] | 0.0225 |
| mean_tvd[line_4_scope] | 0.1774 |
| mean_tvd[line_4_unexplained_fee] | 0.0358 |
| mean_tvd[malicious_content_in_mailboxes] | 0.0000 |
| mean_tvd[malicious_process_running] | 0.0277 |
| mean_tvd[offers_evidence] | 0.0056 |
| mean_tvd[open_to_offer] | 0.0039 |
| mean_tvd[outbound_channel_active] | 0.0023 |
| mean_tvd[price_basis] | 0.0800 |
| mean_tvd[prior_0] | 0.0800 |
| mean_tvd[prior_1] | 0.0921 |
| mean_tvd[prior_2] | 0.1027 |
| mean_tvd[prior_3] | 0.1110 |
| mean_tvd[proposal_reply] | 0.8872 |
| mean_tvd[refund_reason] | 0.2429 |
| mean_tvd[reports_unauthorized] | 0.0504 |
| mean_tvd[reports_unresolved] | 0.0516 |
| mean_tvd[request_specificity] | 0.8159 |
| mean_tvd[requests_human] | 0.0250 |
| mean_tvd[sender_0] | 0.0495 |
| mean_tvd[session_in_attacker_hands] | 0.0297 |
| mean_tvd[shares_credentials] | 0.0013 |
| mean_tvd[spread_beyond_initial_entity] | 0.0034 |
| mean_tvd[statement_not_invoice] | 0.0037 |
| mean_tvd[tax_two_rates] | 0.0013 |
| mean_tvd[threat_chargeback_or_public] | 0.0280 |
| mean_tvd[threat_legal_regulatory] | 0.0077 |
| mean_tvd[too_ambiguous] | 0.0152 |
| mean_tvd[unexplained_charges] | 0.0192 |
| mean_tvd[unusual_urgency] | 0.0186 |
| mean_tvd[urgency] | 0.1941 |
| order_flip_rate[activity_ongoing] | 0.0000 |
| order_flip_rate[adjustment_duplicates_line] | 0.0000 |
| order_flip_rate[affected_scope] | 0.2500 |
| order_flip_rate[already_compensated] | 0.0909 |
| order_flip_rate[amount_vs_record] | 0.3182 |
| order_flip_rate[approval_0] | 0.0764 |
| order_flip_rate[attack_type] | 0.3750 |
| order_flip_rate[attacker_modified_configuration] | 0.0000 |
| order_flip_rate[attacker_persistence_present] | 0.0000 |
| order_flip_rate[attribution] | 0.7500 |
| order_flip_rate[bank_change_claimed_in_comms] | 0.0000 |
| order_flip_rate[billed_above_basis] | 0.0000 |
| order_flip_rate[cancellation_reason] | 0.2857 |
| order_flip_rate[changes_terms] | 0.0000 |
| order_flip_rate[churn_risk] | 0.5000 |
| order_flip_rate[claims_agent_error] | 0.0000 |
| order_flip_rate[context_explains_activity] | 0.0000 |
| order_flip_rate[credentials_exposed] | 0.0000 |
| order_flip_rate[desired_outcome] | 0.2353 |
| order_flip_rate[different_entity] | 0.0000 |
| order_flip_rate[evidence_strength] | 0.6000 |
| order_flip_rate[expressed_satisfaction] | 0.8667 |
| order_flip_rate[first_bad_step] | 0.2500 |
| order_flip_rate[frustration] | 0.2235 |
| order_flip_rate[hardship] | 0.4773 |
| order_flip_rate[in_scope] | 0.0000 |
| order_flip_rate[intent] | 0.2353 |
| order_flip_rate[intent_pair] | 0.0000 |
| order_flip_rate[is_true_positive] | 0.1200 |
| order_flip_rate[issue_resolved] | 0.0588 |
| order_flip_rate[line_0_completion] | 0.3827 |
| order_flip_rate[line_0_kind] | 0.1204 |
| order_flip_rate[line_0_owner_declined] | 0.0000 |
| order_flip_rate[line_0_rate_differs] | 0.0000 |
| order_flip_rate[line_0_rebilled] | 0.0741 |
| order_flip_rate[line_0_scope] | 0.3302 |
| order_flip_rate[line_0_unexplained_fee] | 0.0000 |
| order_flip_rate[line_1_completion] | 0.5401 |
| order_flip_rate[line_1_kind] | 0.1265 |
| order_flip_rate[line_1_owner_declined] | 0.0000 |
| order_flip_rate[line_1_rate_differs] | 0.0000 |
| order_flip_rate[line_1_rebilled] | 0.0154 |
| order_flip_rate[line_1_scope] | 0.4352 |
| order_flip_rate[line_1_unexplained_fee] | 0.0062 |
| order_flip_rate[line_2_completion] | 0.3878 |
| order_flip_rate[line_2_kind] | 0.1510 |
| order_flip_rate[line_2_owner_declined] | 0.0000 |
| order_flip_rate[line_2_rate_differs] | 0.0163 |
| order_flip_rate[line_2_rebilled] | 0.0163 |
| order_flip_rate[line_2_scope] | 0.4571 |
| order_flip_rate[line_2_unexplained_fee] | 0.0327 |
| order_flip_rate[line_3_completion] | 0.4571 |
| order_flip_rate[line_3_kind] | 0.1388 |
| order_flip_rate[line_3_owner_declined] | 0.0000 |
| order_flip_rate[line_3_rate_differs] | 0.0041 |
| order_flip_rate[line_3_rebilled] | 0.0082 |
| order_flip_rate[line_3_scope] | 0.4367 |
| order_flip_rate[line_3_unexplained_fee] | 0.0449 |
| order_flip_rate[line_4_completion] | 0.3523 |
| order_flip_rate[line_4_kind] | 0.0909 |
| order_flip_rate[line_4_owner_declined] | 0.0000 |
| order_flip_rate[line_4_rate_differs] | 0.0000 |
| order_flip_rate[line_4_rebilled] | 0.0000 |
| order_flip_rate[line_4_scope] | 0.7955 |
| order_flip_rate[line_4_unexplained_fee] | 0.0455 |
| order_flip_rate[malicious_content_in_mailboxes] | 0.0000 |
| order_flip_rate[malicious_process_running] | 0.0625 |
| order_flip_rate[offers_evidence] | 0.0000 |
| order_flip_rate[open_to_offer] | 0.0000 |
| order_flip_rate[outbound_channel_active] | 0.0000 |
| order_flip_rate[price_basis] | 0.0988 |
| order_flip_rate[prior_0] | 0.0625 |
| order_flip_rate[prior_1] | 0.1458 |
| order_flip_rate[prior_2] | 0.2082 |
| order_flip_rate[prior_3] | 0.1000 |
| order_flip_rate[proposal_reply] | 1.0000 |
| order_flip_rate[refund_reason] | 0.2273 |
| order_flip_rate[reports_unauthorized] | 0.0824 |
| order_flip_rate[reports_unresolved] | 0.0000 |
| order_flip_rate[request_specificity] | 1.0000 |
| order_flip_rate[requests_human] | 0.0118 |
| order_flip_rate[sender_0] | 0.0311 |
| order_flip_rate[session_in_attacker_hands] | 0.0000 |
| order_flip_rate[shares_credentials] | 0.0000 |
| order_flip_rate[spread_beyond_initial_entity] | 0.0000 |
| order_flip_rate[statement_not_invoice] | 0.0000 |
| order_flip_rate[tax_two_rates] | 0.0000 |
| order_flip_rate[threat_chargeback_or_public] | 0.0000 |
| order_flip_rate[threat_legal_regulatory] | 0.0000 |
| order_flip_rate[too_ambiguous] | 0.0000 |
| order_flip_rate[unexplained_charges] | 0.0000 |
| order_flip_rate[unusual_urgency] | 0.0278 |
| order_flip_rate[urgency] | 0.2471 |
| ordinal_mae[churn_risk] | 0.9375 |
| ordinal_mae[evidence_strength] | 0.6667 |
| ordinal_mae[expressed_satisfaction] | 1.4375 |
| ordinal_mae[frustration] | 0.7222 |
| ordinal_mae[hardship] | 1.2083 |
| ordinal_mae[request_specificity] | 0.7500 |
| ordinal_mae[urgency] | 0.9444 |
| ordinal_mae_expected[churn_risk] | 0.9119 |
| ordinal_mae_expected[evidence_strength] | 0.7188 |
| ordinal_mae_expected[expressed_satisfaction] | 1.2013 |
| ordinal_mae_expected[frustration] | 0.7765 |
| ordinal_mae_expected[hardship] | 1.1501 |
| ordinal_mae_expected[request_specificity] | 0.8972 |
| ordinal_mae_expected[urgency] | 0.9737 |
| per_workflow_accuracy[agent_trace_observability] | 0.5167 |
| per_workflow_accuracy[customer_service] | 0.6140 |
| per_workflow_accuracy[invoice_processing] | 0.6386 |
| per_workflow_accuracy[security_incidents] | 0.4722 |
| tvd_vs_consensus[overall] | 0.4302 |
| valid_accuracy | 0.6322 |

## Agreement vs TypeSafe consensus

| workflow | agreement | common subset | TVD vs consensus |
| --- | --- | --- | --- |
| overall | 0.6322 | 0.6322 | 0.4302 |
| agent_trace_observability | 0.5167 | n/a | 0.4201 |
| customer_service | 0.6140 | n/a | 0.3516 |
| invoice_processing | 0.6386 | n/a | 0.4387 |
| security_incidents | 0.4722 | n/a | 0.3964 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| adjustment_duplicates_line | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| amount_vs_record | 4 | 0.0000 [0.0000, 0.4899] (wilson) † | 1.0000 |
| attack_type | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| attacker_modified_configuration | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| attribution | 3 | 0.0000 [0.0000, 0.5615] (wilson) † | 0.6316 |
| churn_risk | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| hardship | 4 | 0.0000 [0.0000, 0.4899] (wilson) † | 0.5000 |
| intent_pair | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| line_0_completion | 5 | 0.0000 [0.0000, 0.4345] (wilson) † | 0.8663 |
| line_0_kind | 5 | 0.0000 [0.0000, 0.4345] (wilson) † | 1.0000 |
| line_1_kind | 5 | 0.0000 [0.0000, 0.4345] (wilson) † | 1.0000 |
| line_2_completion | 3 | 0.0000 [0.0000, 0.5615] (wilson) † | 1.0000 |
| line_2_kind | 3 | 0.0000 [0.0000, 0.5615] (wilson) † | 1.0000 |
| line_2_scope | 3 | 0.0000 [0.0000, 0.5615] (wilson) † | 1.0000 |
| line_3_kind | 3 | 0.0000 [0.0000, 0.5615] (wilson) † | 0.6855 |
| line_4_completion | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| line_4_kind | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| line_4_scope | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| price_basis | 5 | 0.0000 [0.0000, 0.4345] (wilson) † | 0.6292 |
| proposal_reply | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| request_specificity | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| session_in_attacker_hands | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 0.5000 |
| claims_supported | 5 | 0.2000 [0.0362, 0.6245] (wilson) † | 0.6000 |
| is_true_positive | 5 | 0.2000 [0.0362, 0.6245] (wilson) † | 0.7333 |
| urgency | 5 | 0.2000 [0.0362, 0.6245] (wilson) † | 0.4000 |
| approval_0 | 4 | 0.2500 [0.0456, 0.6994] (wilson) † | 0.8493 |
| first_bad_step | 3 | 0.3333 [0.0615, 0.7923] (wilson) † | 0.6316 |
| line_3_completion | 3 | 0.3333 [0.0615, 0.7923] (wilson) † | 0.6734 |
| line_3_unexplained_fee | 3 | 0.3333 [0.0615, 0.7923] (wilson) † | 0.6855 |
| context_explains_activity | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 1.0000 |
| desired_outcome | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 0.6000 |
| evidence_strength | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 0.6000 |
| intent | 5 | 0.4000 [0.1176, 0.7693] (wilson) | 0.4000 |
| line_0_scope | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 1.0000 |
| line_1_scope | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 1.0000 |
| request_fulfilled | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 0.6000 |
| affected_scope | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| attacker_persistence_present | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| cancellation_reason | 2 | 0.5000 [0.0945, 0.9055] (wilson) † | 1.0000 |
| credentials_exposed | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| refund_done_msg_1 | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| refund_reason | 4 | 0.5000 [0.1500, 0.8500] (wilson) | 0.5000 |
| spread_beyond_initial_entity | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| billed_above_basis | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 0.6201 |
| expressed_satisfaction | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.4000 |
| frustration | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.2000 |
| left_undone | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| line_1_completion | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 0.8663 |
| threat_chargeback_or_public | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 1.0000 |
| line_2_unexplained_fee | 3 | 0.6667 [0.2077, 0.9385] (wilson) † | 1.0000 |
| line_3_scope | 3 | 0.6667 [0.2077, 0.9385] (wilson) † | 1.0000 |
| sender_0 | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.5867 |
| offers_evidence | 4 | 0.7500 [0.3006, 0.9544] (wilson) | 0.7500 |
| prior_0 | 4 | 0.7500 [0.3006, 0.9544] (wilson) † | 1.0000 |
| bank_change_claimed_in_comms | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.7629 |
| claims_agent_error | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.6000 |
| different_entity | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 0.8875 |
| handed_off | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| handoff_required | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.6000 |
| issue_resolved | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| reports_unauthorized | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| reports_unresolved | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.6000 |
| requests_human | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.6000 |
| too_ambiguous | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.7538 |
| unexplained_charges | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.7629 |
| activity_ongoing | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 0.5000 |
| already_compensated | 4 | 1.0000 [0.5101, 1.0000] (wilson) | 1.0000 |
| changes_terms | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| in_scope | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| instructed_by_tool_output | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_0_owner_declined | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_0_rate_differs | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_0_rebilled | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_0_unexplained_fee | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_owner_declined | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_rate_differs | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_rebilled | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_unexplained_fee | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_2_owner_declined | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_rate_differs | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_rebilled | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_owner_declined | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_rate_differs | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_rebilled | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_4_owner_declined | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_rate_differs | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_rebilled | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_unexplained_fee | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| malicious_content_in_mailboxes | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| malicious_process_running | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| open_to_offer | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| outbound_channel_active | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| prior_1 | 4 | 1.0000 [0.5101, 1.0000] (wilson) | 1.0000 |
| prior_2 | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| prior_3 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| refund_done_msg_0 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_1__instructed_by_tool_output__1 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_1__user_asked_or_agreed__1 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_1__within_grant__1 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_2__instructed_by_tool_output__2 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_2__user_asked_or_agreed__2 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_2__within_grant__2 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| secured_msg_0 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| secured_msg_1 | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| shares_credentials | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| statement_not_invoice | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| tax_two_rates | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| threat_legal_regulatory | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| unusual_urgency | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| user_asked_or_agreed | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| within_grant | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
