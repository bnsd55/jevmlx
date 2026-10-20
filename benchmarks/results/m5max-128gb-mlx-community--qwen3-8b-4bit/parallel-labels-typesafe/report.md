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
| timestamp_utc | 2026-09-21T12:20:25+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.7500 [0.6561, 0.8160] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.8577 |
| exact record | 0.2444 |
| case_exact_match | 0.2444 |
| brier | 0.8227 [0.7563, 0.9091] (case_cluster_bootstrap) |
| correctness_auroc | 0.7323 |
| ece_5bin_equal_mass | 0.2025 |
| tie_rate | 0.0055 |
| agreement[agreement_common_subset] | 0.7500 |
| agreement[n_cases] | 45 |
| agreement[n_fields] | 368 |
| agreement[overall] | 0.7500 |
| any_flip_rate[activity_ongoing] | 0.0000 |
| any_flip_rate[adjustment_duplicates_line] | 0.0000 |
| any_flip_rate[affected_scope] | 0.0000 |
| any_flip_rate[already_compensated] | 0.0000 |
| any_flip_rate[amount_vs_record] | 0.1136 |
| any_flip_rate[approval_0] | 0.0035 |
| any_flip_rate[attack_type] | 0.0625 |
| any_flip_rate[attacker_modified_configuration] | 0.0000 |
| any_flip_rate[attacker_persistence_present] | 0.0000 |
| any_flip_rate[attribution] | 0.6875 |
| any_flip_rate[bank_change_claimed_in_comms] | 0.0000 |
| any_flip_rate[billed_above_basis] | 0.0000 |
| any_flip_rate[cancellation_reason] | 0.0000 |
| any_flip_rate[changes_terms] | 0.0000 |
| any_flip_rate[churn_risk] | 0.0000 |
| any_flip_rate[claims_agent_error] | 0.0000 |
| any_flip_rate[context_explains_activity] | 0.0000 |
| any_flip_rate[credentials_exposed] | 0.0000 |
| any_flip_rate[desired_outcome] | 0.2824 |
| any_flip_rate[different_entity] | 0.0000 |
| any_flip_rate[evidence_strength] | 0.3200 |
| any_flip_rate[expressed_satisfaction] | 0.0000 |
| any_flip_rate[first_bad_step] | 0.1875 |
| any_flip_rate[frustration] | 0.0118 |
| any_flip_rate[hardship] | 0.0909 |
| any_flip_rate[in_scope] | 0.0000 |
| any_flip_rate[intent] | 0.1059 |
| any_flip_rate[intent_pair] | 0.0000 |
| any_flip_rate[is_true_positive] | 0.0000 |
| any_flip_rate[issue_resolved] | 0.0000 |
| any_flip_rate[line_0_completion] | 0.0370 |
| any_flip_rate[line_0_kind] | 0.0093 |
| any_flip_rate[line_0_owner_declined] | 0.0000 |
| any_flip_rate[line_0_rate_differs] | 0.0000 |
| any_flip_rate[line_0_rebilled] | 0.0000 |
| any_flip_rate[line_0_scope] | 0.0648 |
| any_flip_rate[line_0_unexplained_fee] | 0.0000 |
| any_flip_rate[line_1_completion] | 0.0247 |
| any_flip_rate[line_1_kind] | 0.0031 |
| any_flip_rate[line_1_owner_declined] | 0.0000 |
| any_flip_rate[line_1_rate_differs] | 0.0000 |
| any_flip_rate[line_1_rebilled] | 0.0000 |
| any_flip_rate[line_1_scope] | 0.0648 |
| any_flip_rate[line_1_unexplained_fee] | 0.0000 |
| any_flip_rate[line_2_completion] | 0.0122 |
| any_flip_rate[line_2_kind] | 0.0408 |
| any_flip_rate[line_2_owner_declined] | 0.0000 |
| any_flip_rate[line_2_rate_differs] | 0.0000 |
| any_flip_rate[line_2_rebilled] | 0.0000 |
| any_flip_rate[line_2_scope] | 0.0449 |
| any_flip_rate[line_2_unexplained_fee] | 0.0000 |
| any_flip_rate[line_3_completion] | 0.0122 |
| any_flip_rate[line_3_kind] | 0.0000 |
| any_flip_rate[line_3_owner_declined] | 0.0000 |
| any_flip_rate[line_3_rate_differs] | 0.0000 |
| any_flip_rate[line_3_rebilled] | 0.0000 |
| any_flip_rate[line_3_scope] | 0.0694 |
| any_flip_rate[line_3_unexplained_fee] | 0.0000 |
| any_flip_rate[line_4_completion] | 0.0227 |
| any_flip_rate[line_4_kind] | 0.0000 |
| any_flip_rate[line_4_owner_declined] | 0.0000 |
| any_flip_rate[line_4_rate_differs] | 0.0000 |
| any_flip_rate[line_4_rebilled] | 0.0000 |
| any_flip_rate[line_4_scope] | 0.0227 |
| any_flip_rate[line_4_unexplained_fee] | 0.0000 |
| any_flip_rate[malicious_content_in_mailboxes] | 0.0000 |
| any_flip_rate[malicious_process_running] | 0.0000 |
| any_flip_rate[offers_evidence] | 0.1136 |
| any_flip_rate[open_to_offer] | 0.0000 |
| any_flip_rate[outbound_channel_active] | 0.0000 |
| any_flip_rate[price_basis] | 0.0154 |
| any_flip_rate[prior_0] | 0.1076 |
| any_flip_rate[prior_1] | 0.0139 |
| any_flip_rate[prior_2] | 0.3061 |
| any_flip_rate[prior_3] | 0.0000 |
| any_flip_rate[proposal_reply] | 0.0000 |
| any_flip_rate[refund_reason] | 0.3182 |
| any_flip_rate[reports_unauthorized] | 0.0000 |
| any_flip_rate[reports_unresolved] | 0.0000 |
| any_flip_rate[request_specificity] | 0.6667 |
| any_flip_rate[requests_human] | 0.0941 |
| any_flip_rate[sender_0] | 0.0000 |
| any_flip_rate[session_in_attacker_hands] | 0.0000 |
| any_flip_rate[shares_credentials] | 0.0000 |
| any_flip_rate[spread_beyond_initial_entity] | 0.0000 |
| any_flip_rate[statement_not_invoice] | 0.0000 |
| any_flip_rate[tax_two_rates] | 0.0000 |
| any_flip_rate[threat_chargeback_or_public] | 0.0000 |
| any_flip_rate[threat_legal_regulatory] | 0.0000 |
| any_flip_rate[too_ambiguous] | 0.0000 |
| any_flip_rate[unexplained_charges] | 0.0000 |
| any_flip_rate[unusual_urgency] | 0.0000 |
| any_flip_rate[urgency] | 0.0353 |
| balanced_accuracy[activity_ongoing] | 0.5000 |
| balanced_accuracy[adjustment_duplicates_line] | 0.0000 |
| balanced_accuracy[affected_scope] | 0.0000 |
| balanced_accuracy[already_compensated] | 1.0000 |
| balanced_accuracy[amount_vs_record] | 0.7500 |
| balanced_accuracy[approval_0] | 0.5000 |
| balanced_accuracy[attack_type] | 1.0000 |
| balanced_accuracy[attacker_modified_configuration] | 0.0000 |
| balanced_accuracy[attacker_persistence_present] | 0.5000 |
| balanced_accuracy[attribution] | 0.2500 |
| balanced_accuracy[bank_change_claimed_in_comms] | 0.5000 |
| balanced_accuracy[billed_above_basis] | 0.5000 |
| balanced_accuracy[cancellation_reason] | 0.5000 |
| balanced_accuracy[changes_terms] | 1.0000 |
| balanced_accuracy[churn_risk] | 1.0000 |
| balanced_accuracy[claims_agent_error] | 0.5000 |
| balanced_accuracy[claims_supported] | 0.5000 |
| balanced_accuracy[context_explains_activity] | 0.0000 |
| balanced_accuracy[credentials_exposed] | 0.5000 |
| balanced_accuracy[desired_outcome] | 0.2222 |
| balanced_accuracy[different_entity] | 0.5000 |
| balanced_accuracy[evidence_strength] | 0.5000 |
| balanced_accuracy[expressed_satisfaction] | 0.3750 |
| balanced_accuracy[first_bad_step] | 0.7500 |
| balanced_accuracy[frustration] | 0.4000 |
| balanced_accuracy[handed_off] | 0.7500 |
| balanced_accuracy[handoff_required] | 0.5000 |
| balanced_accuracy[hardship] | 1.0000 |
| balanced_accuracy[in_scope] | 1.0000 |
| balanced_accuracy[instructed_by_tool_output] | 1.0000 |
| balanced_accuracy[intent] | 0.6667 |
| balanced_accuracy[intent_pair] | 0.0000 |
| balanced_accuracy[is_true_positive] | 0.5000 |
| balanced_accuracy[issue_resolved] | 1.0000 |
| balanced_accuracy[left_undone] | 0.5000 |
| balanced_accuracy[line_0_completion] | 1.0000 |
| balanced_accuracy[line_0_kind] | 1.0000 |
| balanced_accuracy[line_0_owner_declined] | 1.0000 |
| balanced_accuracy[line_0_rate_differs] | 1.0000 |
| balanced_accuracy[line_0_rebilled] | 1.0000 |
| balanced_accuracy[line_0_scope] | 0.6000 |
| balanced_accuracy[line_0_unexplained_fee] | 1.0000 |
| balanced_accuracy[line_1_completion] | 0.5000 |
| balanced_accuracy[line_1_kind] | 1.0000 |
| balanced_accuracy[line_1_owner_declined] | 1.0000 |
| balanced_accuracy[line_1_rate_differs] | 1.0000 |
| balanced_accuracy[line_1_rebilled] | 1.0000 |
| balanced_accuracy[line_1_scope] | 0.4000 |
| balanced_accuracy[line_1_unexplained_fee] | 1.0000 |
| balanced_accuracy[line_2_completion] | 1.0000 |
| balanced_accuracy[line_2_kind] | 1.0000 |
| balanced_accuracy[line_2_owner_declined] | 1.0000 |
| balanced_accuracy[line_2_rate_differs] | 1.0000 |
| balanced_accuracy[line_2_rebilled] | 1.0000 |
| balanced_accuracy[line_2_scope] | 0.6667 |
| balanced_accuracy[line_2_unexplained_fee] | 1.0000 |
| balanced_accuracy[line_3_completion] | 0.5000 |
| balanced_accuracy[line_3_kind] | 0.5000 |
| balanced_accuracy[line_3_owner_declined] | 1.0000 |
| balanced_accuracy[line_3_rate_differs] | 1.0000 |
| balanced_accuracy[line_3_rebilled] | 1.0000 |
| balanced_accuracy[line_3_scope] | 0.6667 |
| balanced_accuracy[line_3_unexplained_fee] | 0.5000 |
| balanced_accuracy[line_4_completion] | 1.0000 |
| balanced_accuracy[line_4_kind] | 1.0000 |
| balanced_accuracy[line_4_owner_declined] | 1.0000 |
| balanced_accuracy[line_4_rate_differs] | 1.0000 |
| balanced_accuracy[line_4_rebilled] | 1.0000 |
| balanced_accuracy[line_4_scope] | 1.0000 |
| balanced_accuracy[line_4_unexplained_fee] | 1.0000 |
| balanced_accuracy[malicious_content_in_mailboxes] | 1.0000 |
| balanced_accuracy[malicious_process_running] | 1.0000 |
| balanced_accuracy[offers_evidence] | 0.5000 |
| balanced_accuracy[open_to_offer] | 0.0000 |
| balanced_accuracy[outbound_channel_active] | 1.0000 |
| balanced_accuracy[price_basis] | 0.6667 |
| balanced_accuracy[prior_0] | 0.2500 |
| balanced_accuracy[prior_1] | 0.7500 |
| balanced_accuracy[prior_2] | 1.0000 |
| balanced_accuracy[prior_3] | 1.0000 |
| balanced_accuracy[proposal_reply] | 1.0000 |
| balanced_accuracy[refund_done_msg_0] | 1.0000 |
| balanced_accuracy[refund_done_msg_1] | 0.5000 |
| balanced_accuracy[refund_reason] | 0.5000 |
| balanced_accuracy[reports_unauthorized] | 1.0000 |
| balanced_accuracy[reports_unresolved] | 0.6667 |
| balanced_accuracy[request_fulfilled] | 0.5000 |
| balanced_accuracy[request_specificity] | 0.0000 |
| balanced_accuracy[requests_human] | 0.5000 |
| balanced_accuracy[safety_1__instructed_by_tool_output__1] | 1.0000 |
| balanced_accuracy[safety_1__user_asked_or_agreed__1] | 1.0000 |
| balanced_accuracy[safety_1__within_grant__1] | 1.0000 |
| balanced_accuracy[safety_2__instructed_by_tool_output__2] | 1.0000 |
| balanced_accuracy[safety_2__user_asked_or_agreed__2] | 1.0000 |
| balanced_accuracy[safety_2__within_grant__2] | 1.0000 |
| balanced_accuracy[secured_msg_0] | 1.0000 |
| balanced_accuracy[secured_msg_1] | 1.0000 |
| balanced_accuracy[sender_0] | 0.5000 |
| balanced_accuracy[session_in_attacker_hands] | 0.5000 |
| balanced_accuracy[shares_credentials] | 1.0000 |
| balanced_accuracy[spread_beyond_initial_entity] | 0.5000 |
| balanced_accuracy[statement_not_invoice] | 1.0000 |
| balanced_accuracy[tax_two_rates] | 1.0000 |
| balanced_accuracy[threat_chargeback_or_public] | 1.0000 |
| balanced_accuracy[threat_legal_regulatory] | 1.0000 |
| balanced_accuracy[too_ambiguous] | 0.5000 |
| balanced_accuracy[unexplained_charges] | 0.5000 |
| balanced_accuracy[unusual_urgency] | 1.0000 |
| balanced_accuracy[urgency] | 0.3333 |
| balanced_accuracy[user_asked_or_agreed] | 1.0000 |
| balanced_accuracy[within_grant] | 1.0000 |
| macro_f1[activity_ongoing] | 0.3333 |
| macro_f1[adjustment_duplicates_line] | 0.0000 |
| macro_f1[affected_scope] | 0.0000 |
| macro_f1[already_compensated] | 1.0000 |
| macro_f1[amount_vs_record] | 0.8571 |
| macro_f1[approval_0] | 0.5000 |
| macro_f1[attack_type] | 1.0000 |
| macro_f1[attacker_modified_configuration] | 0.0000 |
| macro_f1[attacker_persistence_present] | 0.3333 |
| macro_f1[attribution] | 0.3333 |
| macro_f1[bank_change_claimed_in_comms] | 0.4444 |
| macro_f1[billed_above_basis] | 0.3750 |
| macro_f1[cancellation_reason] | 0.6667 |
| macro_f1[changes_terms] | 1.0000 |
| macro_f1[churn_risk] | 1.0000 |
| macro_f1[claims_agent_error] | 0.3750 |
| macro_f1[claims_supported] | 0.2857 |
| macro_f1[context_explains_activity] | 0.0000 |
| macro_f1[credentials_exposed] | 0.3333 |
| macro_f1[desired_outcome] | 0.1905 |
| macro_f1[different_entity] | 0.4444 |
| macro_f1[evidence_strength] | 0.2857 |
| macro_f1[expressed_satisfaction] | 0.2500 |
| macro_f1[first_bad_step] | 0.8333 |
| macro_f1[frustration] | 0.2800 |
| macro_f1[handed_off] | 0.5833 |
| macro_f1[handoff_required] | 0.3750 |
| macro_f1[hardship] | 1.0000 |
| macro_f1[in_scope] | 1.0000 |
| macro_f1[instructed_by_tool_output] | 1.0000 |
| macro_f1[intent] | 0.5556 |
| macro_f1[intent_pair] | 0.0000 |
| macro_f1[is_true_positive] | 0.2857 |
| macro_f1[issue_resolved] | 1.0000 |
| macro_f1[left_undone] | 0.3750 |
| macro_f1[line_0_completion] | 1.0000 |
| macro_f1[line_0_kind] | 1.0000 |
| macro_f1[line_0_owner_declined] | 1.0000 |
| macro_f1[line_0_rate_differs] | 1.0000 |
| macro_f1[line_0_rebilled] | 1.0000 |
| macro_f1[line_0_scope] | 0.7500 |
| macro_f1[line_0_unexplained_fee] | 1.0000 |
| macro_f1[line_1_completion] | 0.5000 |
| macro_f1[line_1_kind] | 1.0000 |
| macro_f1[line_1_owner_declined] | 1.0000 |
| macro_f1[line_1_rate_differs] | 1.0000 |
| macro_f1[line_1_rebilled] | 1.0000 |
| macro_f1[line_1_scope] | 0.5714 |
| macro_f1[line_1_unexplained_fee] | 1.0000 |
| macro_f1[line_2_completion] | 1.0000 |
| macro_f1[line_2_kind] | 1.0000 |
| macro_f1[line_2_owner_declined] | 1.0000 |
| macro_f1[line_2_rate_differs] | 1.0000 |
| macro_f1[line_2_rebilled] | 1.0000 |
| macro_f1[line_2_scope] | 0.8000 |
| macro_f1[line_2_unexplained_fee] | 1.0000 |
| macro_f1[line_3_completion] | 0.4000 |
| macro_f1[line_3_kind] | 0.4000 |
| macro_f1[line_3_owner_declined] | 1.0000 |
| macro_f1[line_3_rate_differs] | 1.0000 |
| macro_f1[line_3_rebilled] | 1.0000 |
| macro_f1[line_3_scope] | 0.8000 |
| macro_f1[line_3_unexplained_fee] | 0.4000 |
| macro_f1[line_4_completion] | 1.0000 |
| macro_f1[line_4_kind] | 1.0000 |
| macro_f1[line_4_owner_declined] | 1.0000 |
| macro_f1[line_4_rate_differs] | 1.0000 |
| macro_f1[line_4_rebilled] | 1.0000 |
| macro_f1[line_4_scope] | 1.0000 |
| macro_f1[line_4_unexplained_fee] | 1.0000 |
| macro_f1[malicious_content_in_mailboxes] | 1.0000 |
| macro_f1[malicious_process_running] | 1.0000 |
| macro_f1[offers_evidence] | 0.4286 |
| macro_f1[open_to_offer] | 0.0000 |
| macro_f1[outbound_channel_active] | 1.0000 |
| macro_f1[price_basis] | 0.6667 |
| macro_f1[prior_0] | 0.4000 |
| macro_f1[prior_1] | 0.8571 |
| macro_f1[prior_2] | 1.0000 |
| macro_f1[prior_3] | 1.0000 |
| macro_f1[proposal_reply] | 1.0000 |
| macro_f1[refund_done_msg_0] | 1.0000 |
| macro_f1[refund_done_msg_1] | 0.3333 |
| macro_f1[refund_reason] | 0.5556 |
| macro_f1[reports_unauthorized] | 1.0000 |
| macro_f1[reports_unresolved] | 0.5833 |
| macro_f1[request_fulfilled] | 0.3750 |
| macro_f1[request_specificity] | 0.0000 |
| macro_f1[requests_human] | 0.3750 |
| macro_f1[safety_1__instructed_by_tool_output__1] | 1.0000 |
| macro_f1[safety_1__user_asked_or_agreed__1] | 1.0000 |
| macro_f1[safety_1__within_grant__1] | 1.0000 |
| macro_f1[safety_2__instructed_by_tool_output__2] | 1.0000 |
| macro_f1[safety_2__user_asked_or_agreed__2] | 1.0000 |
| macro_f1[safety_2__within_grant__2] | 1.0000 |
| macro_f1[secured_msg_0] | 1.0000 |
| macro_f1[secured_msg_1] | 1.0000 |
| macro_f1[sender_0] | 0.4000 |
| macro_f1[session_in_attacker_hands] | 0.3333 |
| macro_f1[shares_credentials] | 1.0000 |
| macro_f1[spread_beyond_initial_entity] | 0.3333 |
| macro_f1[statement_not_invoice] | 1.0000 |
| macro_f1[tax_two_rates] | 1.0000 |
| macro_f1[threat_chargeback_or_public] | 1.0000 |
| macro_f1[threat_legal_regulatory] | 1.0000 |
| macro_f1[too_ambiguous] | 0.4444 |
| macro_f1[unexplained_charges] | 0.4444 |
| macro_f1[unusual_urgency] | 1.0000 |
| macro_f1[urgency] | 0.1905 |
| macro_f1[user_asked_or_agreed] | 1.0000 |
| macro_f1[within_grant] | 1.0000 |
| mean_tvd[activity_ongoing] | 0.0016 |
| mean_tvd[adjustment_duplicates_line] | 0.0000 |
| mean_tvd[affected_scope] | 0.0000 |
| mean_tvd[already_compensated] | 0.0000 |
| mean_tvd[amount_vs_record] | 0.1174 |
| mean_tvd[approval_0] | 0.0056 |
| mean_tvd[attack_type] | 0.0768 |
| mean_tvd[attacker_modified_configuration] | 0.0002 |
| mean_tvd[attacker_persistence_present] | 0.0000 |
| mean_tvd[attribution] | 0.6156 |
| mean_tvd[bank_change_claimed_in_comms] | 0.0000 |
| mean_tvd[billed_above_basis] | 0.0009 |
| mean_tvd[cancellation_reason] | 0.0001 |
| mean_tvd[changes_terms] | 0.0001 |
| mean_tvd[churn_risk] | 0.0142 |
| mean_tvd[claims_agent_error] | 0.0009 |
| mean_tvd[context_explains_activity] | 0.0010 |
| mean_tvd[credentials_exposed] | 0.0001 |
| mean_tvd[desired_outcome] | 0.2259 |
| mean_tvd[different_entity] | 0.0000 |
| mean_tvd[evidence_strength] | 0.2659 |
| mean_tvd[expressed_satisfaction] | 0.0503 |
| mean_tvd[first_bad_step] | 0.2012 |
| mean_tvd[frustration] | 0.0457 |
| mean_tvd[hardship] | 0.0956 |
| mean_tvd[in_scope] | 0.0000 |
| mean_tvd[intent] | 0.1138 |
| mean_tvd[intent_pair] | 0.0032 |
| mean_tvd[is_true_positive] | 0.0164 |
| mean_tvd[issue_resolved] | 0.0067 |
| mean_tvd[line_0_completion] | 0.0456 |
| mean_tvd[line_0_kind] | 0.0068 |
| mean_tvd[line_0_owner_declined] | 0.0000 |
| mean_tvd[line_0_rate_differs] | 0.0000 |
| mean_tvd[line_0_rebilled] | 0.0000 |
| mean_tvd[line_0_scope] | 0.0782 |
| mean_tvd[line_0_unexplained_fee] | 0.0000 |
| mean_tvd[line_1_completion] | 0.0302 |
| mean_tvd[line_1_kind] | 0.0052 |
| mean_tvd[line_1_owner_declined] | 0.0000 |
| mean_tvd[line_1_rate_differs] | 0.0001 |
| mean_tvd[line_1_rebilled] | 0.0002 |
| mean_tvd[line_1_scope] | 0.0762 |
| mean_tvd[line_1_unexplained_fee] | 0.0000 |
| mean_tvd[line_2_completion] | 0.0131 |
| mean_tvd[line_2_kind] | 0.0529 |
| mean_tvd[line_2_owner_declined] | 0.0000 |
| mean_tvd[line_2_rate_differs] | 0.0001 |
| mean_tvd[line_2_rebilled] | 0.0002 |
| mean_tvd[line_2_scope] | 0.0516 |
| mean_tvd[line_2_unexplained_fee] | 0.0000 |
| mean_tvd[line_3_completion] | 0.0157 |
| mean_tvd[line_3_kind] | 0.0059 |
| mean_tvd[line_3_owner_declined] | 0.0000 |
| mean_tvd[line_3_rate_differs] | 0.0000 |
| mean_tvd[line_3_rebilled] | 0.0001 |
| mean_tvd[line_3_scope] | 0.0753 |
| mean_tvd[line_3_unexplained_fee] | 0.0000 |
| mean_tvd[line_4_completion] | 0.0188 |
| mean_tvd[line_4_kind] | 0.0026 |
| mean_tvd[line_4_owner_declined] | 0.0000 |
| mean_tvd[line_4_rate_differs] | 0.0001 |
| mean_tvd[line_4_rebilled] | 0.0002 |
| mean_tvd[line_4_scope] | 0.0313 |
| mean_tvd[line_4_unexplained_fee] | 0.0000 |
| mean_tvd[malicious_content_in_mailboxes] | 0.0000 |
| mean_tvd[malicious_process_running] | 0.0294 |
| mean_tvd[offers_evidence] | 0.0979 |
| mean_tvd[open_to_offer] | 0.0006 |
| mean_tvd[outbound_channel_active] | 0.0016 |
| mean_tvd[price_basis] | 0.0343 |
| mean_tvd[prior_0] | 0.0867 |
| mean_tvd[prior_1] | 0.0525 |
| mean_tvd[prior_2] | 0.1229 |
| mean_tvd[prior_3] | 0.0502 |
| mean_tvd[proposal_reply] | 0.0001 |
| mean_tvd[refund_reason] | 0.2837 |
| mean_tvd[reports_unauthorized] | 0.0003 |
| mean_tvd[reports_unresolved] | 0.0010 |
| mean_tvd[request_specificity] | 0.2391 |
| mean_tvd[requests_human] | 0.0731 |
| mean_tvd[sender_0] | 0.0438 |
| mean_tvd[session_in_attacker_hands] | 0.0008 |
| mean_tvd[shares_credentials] | 0.0000 |
| mean_tvd[spread_beyond_initial_entity] | 0.0001 |
| mean_tvd[statement_not_invoice] | 0.0000 |
| mean_tvd[tax_two_rates] | 0.0000 |
| mean_tvd[threat_chargeback_or_public] | 0.0005 |
| mean_tvd[threat_legal_regulatory] | 0.0002 |
| mean_tvd[too_ambiguous] | 0.0000 |
| mean_tvd[unexplained_charges] | 0.0000 |
| mean_tvd[unusual_urgency] | 0.0000 |
| mean_tvd[urgency] | 0.0499 |
| order_flip_rate[activity_ongoing] | 0.0000 |
| order_flip_rate[adjustment_duplicates_line] | 0.0000 |
| order_flip_rate[affected_scope] | 0.0000 |
| order_flip_rate[already_compensated] | 0.0000 |
| order_flip_rate[amount_vs_record] | 0.1136 |
| order_flip_rate[approval_0] | 0.0035 |
| order_flip_rate[attack_type] | 0.0625 |
| order_flip_rate[attacker_modified_configuration] | 0.0000 |
| order_flip_rate[attacker_persistence_present] | 0.0000 |
| order_flip_rate[attribution] | 0.6875 |
| order_flip_rate[bank_change_claimed_in_comms] | 0.0000 |
| order_flip_rate[billed_above_basis] | 0.0000 |
| order_flip_rate[cancellation_reason] | 0.0000 |
| order_flip_rate[changes_terms] | 0.0000 |
| order_flip_rate[churn_risk] | 0.0000 |
| order_flip_rate[claims_agent_error] | 0.0000 |
| order_flip_rate[context_explains_activity] | 0.0000 |
| order_flip_rate[credentials_exposed] | 0.0000 |
| order_flip_rate[desired_outcome] | 0.2824 |
| order_flip_rate[different_entity] | 0.0000 |
| order_flip_rate[evidence_strength] | 0.3200 |
| order_flip_rate[expressed_satisfaction] | 0.0000 |
| order_flip_rate[first_bad_step] | 0.1875 |
| order_flip_rate[frustration] | 0.0118 |
| order_flip_rate[hardship] | 0.0909 |
| order_flip_rate[in_scope] | 0.0000 |
| order_flip_rate[intent] | 0.1059 |
| order_flip_rate[intent_pair] | 0.0000 |
| order_flip_rate[is_true_positive] | 0.0000 |
| order_flip_rate[issue_resolved] | 0.0000 |
| order_flip_rate[line_0_completion] | 0.0370 |
| order_flip_rate[line_0_kind] | 0.0093 |
| order_flip_rate[line_0_owner_declined] | 0.0000 |
| order_flip_rate[line_0_rate_differs] | 0.0000 |
| order_flip_rate[line_0_rebilled] | 0.0000 |
| order_flip_rate[line_0_scope] | 0.0648 |
| order_flip_rate[line_0_unexplained_fee] | 0.0000 |
| order_flip_rate[line_1_completion] | 0.0247 |
| order_flip_rate[line_1_kind] | 0.0031 |
| order_flip_rate[line_1_owner_declined] | 0.0000 |
| order_flip_rate[line_1_rate_differs] | 0.0000 |
| order_flip_rate[line_1_rebilled] | 0.0000 |
| order_flip_rate[line_1_scope] | 0.0648 |
| order_flip_rate[line_1_unexplained_fee] | 0.0000 |
| order_flip_rate[line_2_completion] | 0.0122 |
| order_flip_rate[line_2_kind] | 0.0408 |
| order_flip_rate[line_2_owner_declined] | 0.0000 |
| order_flip_rate[line_2_rate_differs] | 0.0000 |
| order_flip_rate[line_2_rebilled] | 0.0000 |
| order_flip_rate[line_2_scope] | 0.0449 |
| order_flip_rate[line_2_unexplained_fee] | 0.0000 |
| order_flip_rate[line_3_completion] | 0.0122 |
| order_flip_rate[line_3_kind] | 0.0000 |
| order_flip_rate[line_3_owner_declined] | 0.0000 |
| order_flip_rate[line_3_rate_differs] | 0.0000 |
| order_flip_rate[line_3_rebilled] | 0.0000 |
| order_flip_rate[line_3_scope] | 0.0694 |
| order_flip_rate[line_3_unexplained_fee] | 0.0000 |
| order_flip_rate[line_4_completion] | 0.0227 |
| order_flip_rate[line_4_kind] | 0.0000 |
| order_flip_rate[line_4_owner_declined] | 0.0000 |
| order_flip_rate[line_4_rate_differs] | 0.0000 |
| order_flip_rate[line_4_rebilled] | 0.0000 |
| order_flip_rate[line_4_scope] | 0.0227 |
| order_flip_rate[line_4_unexplained_fee] | 0.0000 |
| order_flip_rate[malicious_content_in_mailboxes] | 0.0000 |
| order_flip_rate[malicious_process_running] | 0.0000 |
| order_flip_rate[offers_evidence] | 0.1136 |
| order_flip_rate[open_to_offer] | 0.0000 |
| order_flip_rate[outbound_channel_active] | 0.0000 |
| order_flip_rate[price_basis] | 0.0154 |
| order_flip_rate[prior_0] | 0.1076 |
| order_flip_rate[prior_1] | 0.0139 |
| order_flip_rate[prior_2] | 0.3061 |
| order_flip_rate[prior_3] | 0.0000 |
| order_flip_rate[proposal_reply] | 0.0000 |
| order_flip_rate[refund_reason] | 0.3182 |
| order_flip_rate[reports_unauthorized] | 0.0000 |
| order_flip_rate[reports_unresolved] | 0.0000 |
| order_flip_rate[request_specificity] | 0.6667 |
| order_flip_rate[requests_human] | 0.0941 |
| order_flip_rate[sender_0] | 0.0000 |
| order_flip_rate[session_in_attacker_hands] | 0.0000 |
| order_flip_rate[shares_credentials] | 0.0000 |
| order_flip_rate[spread_beyond_initial_entity] | 0.0000 |
| order_flip_rate[statement_not_invoice] | 0.0000 |
| order_flip_rate[tax_two_rates] | 0.0000 |
| order_flip_rate[threat_chargeback_or_public] | 0.0000 |
| order_flip_rate[threat_legal_regulatory] | 0.0000 |
| order_flip_rate[too_ambiguous] | 0.0000 |
| order_flip_rate[unexplained_charges] | 0.0000 |
| order_flip_rate[unusual_urgency] | 0.0000 |
| order_flip_rate[urgency] | 0.0353 |
| ordinal_mae[churn_risk] | 0.0000 |
| ordinal_mae[evidence_strength] | 1.0000 |
| ordinal_mae[expressed_satisfaction] | 0.7500 |
| ordinal_mae[frustration] | 0.7778 |
| ordinal_mae[hardship] | 0.1250 |
| ordinal_mae[request_specificity] | 0.5000 |
| ordinal_mae[urgency] | 0.7000 |
| ordinal_mae_expected[churn_risk] | 0.0127 |
| ordinal_mae_expected[evidence_strength] | 1.0157 |
| ordinal_mae_expected[expressed_satisfaction] | 0.7980 |
| ordinal_mae_expected[frustration] | 0.8199 |
| ordinal_mae_expected[hardship] | 0.1285 |
| ordinal_mae_expected[request_specificity] | 0.5850 |
| ordinal_mae_expected[urgency] | 0.7025 |
| per_workflow_accuracy[agent_trace_observability] | 0.6154 |
| per_workflow_accuracy[customer_service] | 0.7391 |
| per_workflow_accuracy[invoice_processing] | 0.8641 |
| per_workflow_accuracy[security_incidents] | 0.4250 |
| tvd_vs_consensus[overall] | 0.2511 |
| valid_accuracy | 0.7562 |

## Agreement vs TypeSafe consensus

| workflow | agreement | common subset | TVD vs consensus |
| --- | --- | --- | --- |
| overall | 0.7500 | 0.7500 | 0.2511 |
| agent_trace_observability | 0.6154 | n/a | 0.3902 |
| customer_service | 0.7391 | n/a | 0.2701 |
| invoice_processing | 0.8641 | n/a | 0.1485 |
| security_incidents | 0.4250 | n/a | 0.5186 |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| adjustment_duplicates_line | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| affected_scope | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 0.5000 |
| attacker_modified_configuration | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| context_explains_activity | 6 | 0.0000 [0.0000, 0.3903] (wilson) † | 1.0000 |
| intent_pair | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| open_to_offer | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| request_specificity | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| prior_0 | 4 | 0.2500 [0.0456, 0.6994] (wilson) † | 1.0000 |
| attribution | 3 | 0.3333 [0.0615, 0.7923] (wilson) † | 0.6667 |
| evidence_strength | 6 | 0.3333 [0.0968, 0.7000] (wilson) † | 0.6000 |
| is_true_positive | 6 | 0.3333 [0.0968, 0.7000] (wilson) † | 0.6000 |
| claims_supported | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 0.6000 |
| desired_outcome | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 0.6000 |
| expressed_satisfaction | 5 | 0.4000 [0.1176, 0.7693] (wilson) | 0.4000 |
| frustration | 5 | 0.4000 [0.1176, 0.7693] (wilson) | 0.2000 |
| line_1_scope | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 1.0000 |
| urgency | 5 | 0.4000 [0.1176, 0.7693] (wilson) | 0.4000 |
| activity_ongoing | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| attacker_persistence_present | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| cancellation_reason | 2 | 0.5000 [0.0945, 0.9055] (wilson) † | 1.0000 |
| credentials_exposed | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| refund_done_msg_1 | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| refund_reason | 4 | 0.5000 [0.1500, 0.8500] (wilson) | 0.5000 |
| session_in_attacker_hands | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| spread_beyond_initial_entity | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| billed_above_basis | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| claims_agent_error | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| handed_off | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 0.8000 |
| handoff_required | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| intent | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.4000 |
| left_undone | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| line_0_scope | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 1.0000 |
| reports_unresolved | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| request_fulfilled | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| requests_human | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| first_bad_step | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.6667 |
| line_2_scope | 3 | 0.6667 [0.2077, 0.9385] (wilson) † | 1.0000 |
| line_3_completion | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.6667 |
| line_3_kind | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.6667 |
| line_3_scope | 3 | 0.6667 [0.2077, 0.9385] (wilson) † | 1.0000 |
| line_3_unexplained_fee | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.6667 |
| sender_0 | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.6667 |
| amount_vs_record | 4 | 0.7500 [0.3006, 0.9544] (wilson) † | 1.0000 |
| approval_0 | 4 | 0.7500 [0.3006, 0.9544] (wilson) | 0.7500 |
| offers_evidence | 4 | 0.7500 [0.3006, 0.9544] (wilson) | 0.7500 |
| prior_1 | 4 | 0.7500 [0.3006, 0.9544] (wilson) † | 1.0000 |
| bank_change_claimed_in_comms | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| different_entity | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| line_1_completion | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| price_basis | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.6000 |
| too_ambiguous | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| unexplained_charges | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| already_compensated | 4 | 1.0000 [0.5101, 1.0000] (wilson) | 1.0000 |
| attack_type | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| changes_terms | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| churn_risk | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| hardship | 4 | 1.0000 [0.5101, 1.0000] (wilson) | 0.5000 |
| in_scope | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| instructed_by_tool_output | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| issue_resolved | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_0_completion | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 0.8000 |
| line_0_kind | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_0_owner_declined | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_0_rate_differs | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_0_rebilled | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_0_unexplained_fee | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_kind | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_owner_declined | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_rate_differs | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_rebilled | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_1_unexplained_fee | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| line_2_completion | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_kind | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_owner_declined | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_rate_differs | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_rebilled | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_unexplained_fee | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_owner_declined | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_rate_differs | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_rebilled | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_4_completion | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_kind | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_owner_declined | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_rate_differs | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_rebilled | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_scope | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_unexplained_fee | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| malicious_content_in_mailboxes | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| malicious_process_running | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| outbound_channel_active | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| prior_2 | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| prior_3 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| proposal_reply | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| refund_done_msg_0 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| reports_unauthorized | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
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
| threat_chargeback_or_public | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| threat_legal_regulatory | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| unusual_urgency | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| user_asked_or_agreed | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| within_grant | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
