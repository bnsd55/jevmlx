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
| timestamp_utc | 2026-09-21T06:36:05+00:00 |

## Metrics

| metric | value |
| --- | --- |
| accuracy | 0.6767 [0.5179, 0.7853] (case_cluster_bootstrap) |
| majority baseline (mean over fields) | 0.8577 |
| exact record | 0.2500 |
| case_exact_match | 0.2500 |
| agreement[agreement_common_subset] | 0.6767 |
| agreement[n_cases] | 44 |
| agreement[n_fields] | 365 |
| agreement[overall] | 0.6767 |
| balanced_accuracy[activity_ongoing] | 0.0000 |
| balanced_accuracy[adjustment_duplicates_line] | 0.0000 |
| balanced_accuracy[affected_scope] | 0.0000 |
| balanced_accuracy[already_compensated] | 1.0000 |
| balanced_accuracy[amount_vs_record] | 1.0000 |
| balanced_accuracy[approval_0] | 0.3333 |
| balanced_accuracy[attack_type] | 1.0000 |
| balanced_accuracy[attacker_modified_configuration] | 0.5000 |
| balanced_accuracy[attacker_persistence_present] | 1.0000 |
| balanced_accuracy[attribution] | 0.0000 |
| balanced_accuracy[bank_change_claimed_in_comms] | 0.3750 |
| balanced_accuracy[billed_above_basis] | 0.5000 |
| balanced_accuracy[cancellation_reason] | 0.0000 |
| balanced_accuracy[changes_terms] | 1.0000 |
| balanced_accuracy[churn_risk] | 0.0000 |
| balanced_accuracy[claims_agent_error] | 0.5833 |
| balanced_accuracy[claims_supported] | 0.5000 |
| balanced_accuracy[context_explains_activity] | 0.0000 |
| balanced_accuracy[credentials_exposed] | 0.5000 |
| balanced_accuracy[desired_outcome] | 0.2222 |
| balanced_accuracy[different_entity] | 0.3750 |
| balanced_accuracy[evidence_strength] | 0.5000 |
| balanced_accuracy[expressed_satisfaction] | 0.1250 |
| balanced_accuracy[first_bad_step] | 0.2500 |
| balanced_accuracy[frustration] | 0.6000 |
| balanced_accuracy[handed_off] | 0.8750 |
| balanced_accuracy[handoff_required] | 0.3333 |
| balanced_accuracy[hardship] | 0.6667 |
| balanced_accuracy[in_scope] | 0.0000 |
| balanced_accuracy[instructed_by_tool_output] | 1.0000 |
| balanced_accuracy[intent] | 0.5000 |
| balanced_accuracy[intent_pair] | 0.0000 |
| balanced_accuracy[is_true_positive] | 0.5000 |
| balanced_accuracy[issue_resolved] | 0.8000 |
| balanced_accuracy[left_undone] | 0.7500 |
| balanced_accuracy[line_0_completion] | 0.5000 |
| balanced_accuracy[line_0_kind] | 0.8000 |
| balanced_accuracy[line_0_owner_declined] | 0.8000 |
| balanced_accuracy[line_0_rate_differs] | 0.8000 |
| balanced_accuracy[line_0_rebilled] | 0.8000 |
| balanced_accuracy[line_0_scope] | 0.8000 |
| balanced_accuracy[line_0_unexplained_fee] | 0.8000 |
| balanced_accuracy[line_1_completion] | 0.5000 |
| balanced_accuracy[line_1_kind] | 0.8000 |
| balanced_accuracy[line_1_owner_declined] | 0.8000 |
| balanced_accuracy[line_1_rate_differs] | 0.8000 |
| balanced_accuracy[line_1_rebilled] | 0.8000 |
| balanced_accuracy[line_1_scope] | 0.8000 |
| balanced_accuracy[line_1_unexplained_fee] | 0.8000 |
| balanced_accuracy[line_2_completion] | 1.0000 |
| balanced_accuracy[line_2_kind] | 1.0000 |
| balanced_accuracy[line_2_owner_declined] | 1.0000 |
| balanced_accuracy[line_2_rate_differs] | 1.0000 |
| balanced_accuracy[line_2_rebilled] | 1.0000 |
| balanced_accuracy[line_2_scope] | 1.0000 |
| balanced_accuracy[line_2_unexplained_fee] | 1.0000 |
| balanced_accuracy[line_3_completion] | 0.5000 |
| balanced_accuracy[line_3_kind] | 0.5000 |
| balanced_accuracy[line_3_owner_declined] | 1.0000 |
| balanced_accuracy[line_3_rate_differs] | 1.0000 |
| balanced_accuracy[line_3_rebilled] | 1.0000 |
| balanced_accuracy[line_3_scope] | 1.0000 |
| balanced_accuracy[line_3_unexplained_fee] | 0.5000 |
| balanced_accuracy[line_4_completion] | 1.0000 |
| balanced_accuracy[line_4_kind] | 1.0000 |
| balanced_accuracy[line_4_owner_declined] | 1.0000 |
| balanced_accuracy[line_4_rate_differs] | 1.0000 |
| balanced_accuracy[line_4_rebilled] | 1.0000 |
| balanced_accuracy[line_4_scope] | 1.0000 |
| balanced_accuracy[line_4_unexplained_fee] | 1.0000 |
| balanced_accuracy[malicious_content_in_mailboxes] | 1.0000 |
| balanced_accuracy[malicious_process_running] | 0.0000 |
| balanced_accuracy[offers_evidence] | 1.0000 |
| balanced_accuracy[open_to_offer] | 0.0000 |
| balanced_accuracy[outbound_channel_active] | 1.0000 |
| balanced_accuracy[price_basis] | 0.3333 |
| balanced_accuracy[prior_0] | 0.0000 |
| balanced_accuracy[prior_1] | 0.0000 |
| balanced_accuracy[prior_2] | 0.0000 |
| balanced_accuracy[prior_3] | 0.0000 |
| balanced_accuracy[proposal_reply] | 1.0000 |
| balanced_accuracy[refund_done_msg_0] | 1.0000 |
| balanced_accuracy[refund_done_msg_1] | 1.0000 |
| balanced_accuracy[refund_reason] | 0.6667 |
| balanced_accuracy[reports_unauthorized] | 0.8000 |
| balanced_accuracy[reports_unresolved] | 0.0000 |
| balanced_accuracy[request_fulfilled] | 1.0000 |
| balanced_accuracy[request_specificity] | 0.0000 |
| balanced_accuracy[requests_human] | 0.7500 |
| balanced_accuracy[safety_1__instructed_by_tool_output__1] | 1.0000 |
| balanced_accuracy[safety_1__user_asked_or_agreed__1] | 1.0000 |
| balanced_accuracy[safety_1__within_grant__1] | 1.0000 |
| balanced_accuracy[safety_2__instructed_by_tool_output__2] | 1.0000 |
| balanced_accuracy[safety_2__user_asked_or_agreed__2] | 1.0000 |
| balanced_accuracy[safety_2__within_grant__2] | 1.0000 |
| balanced_accuracy[secured_msg_0] | 1.0000 |
| balanced_accuracy[secured_msg_1] | 1.0000 |
| balanced_accuracy[sender_0] | 0.2500 |
| balanced_accuracy[session_in_attacker_hands] | 0.5000 |
| balanced_accuracy[shares_credentials] | 1.0000 |
| balanced_accuracy[spread_beyond_initial_entity] | 0.5000 |
| balanced_accuracy[statement_not_invoice] | 0.8000 |
| balanced_accuracy[tax_two_rates] | 0.8000 |
| balanced_accuracy[threat_chargeback_or_public] | 1.0000 |
| balanced_accuracy[threat_legal_regulatory] | 1.0000 |
| balanced_accuracy[too_ambiguous] | 0.3750 |
| balanced_accuracy[unexplained_charges] | 0.3750 |
| balanced_accuracy[unusual_urgency] | 0.8000 |
| balanced_accuracy[urgency] | 0.3333 |
| balanced_accuracy[user_asked_or_agreed] | 1.0000 |
| balanced_accuracy[within_grant] | 0.0000 |
| macro_f1[activity_ongoing] | 0.0000 |
| macro_f1[adjustment_duplicates_line] | 0.0000 |
| macro_f1[affected_scope] | 0.0000 |
| macro_f1[already_compensated] | 1.0000 |
| macro_f1[amount_vs_record] | 1.0000 |
| macro_f1[approval_0] | 0.4000 |
| macro_f1[attack_type] | 1.0000 |
| macro_f1[attacker_modified_configuration] | 0.6667 |
| macro_f1[attacker_persistence_present] | 1.0000 |
| macro_f1[attribution] | 0.0000 |
| macro_f1[bank_change_claimed_in_comms] | 0.3750 |
| macro_f1[billed_above_basis] | 0.4286 |
| macro_f1[cancellation_reason] | 0.0000 |
| macro_f1[changes_terms] | 1.0000 |
| macro_f1[churn_risk] | 0.0000 |
| macro_f1[claims_agent_error] | 0.5833 |
| macro_f1[claims_supported] | 0.3750 |
| macro_f1[context_explains_activity] | 0.0000 |
| macro_f1[credentials_exposed] | 0.3333 |
| macro_f1[desired_outcome] | 0.2222 |
| macro_f1[different_entity] | 0.3750 |
| macro_f1[evidence_strength] | 0.2857 |
| macro_f1[expressed_satisfaction] | 0.1667 |
| macro_f1[first_bad_step] | 0.3333 |
| macro_f1[frustration] | 0.4667 |
| macro_f1[handed_off] | 0.7619 |
| macro_f1[handoff_required] | 0.2857 |
| macro_f1[hardship] | 0.6667 |
| macro_f1[in_scope] | 0.0000 |
| macro_f1[instructed_by_tool_output] | 1.0000 |
| macro_f1[intent] | 0.4667 |
| macro_f1[intent_pair] | 0.0000 |
| macro_f1[is_true_positive] | 0.3750 |
| macro_f1[issue_resolved] | 0.8889 |
| macro_f1[left_undone] | 0.7619 |
| macro_f1[line_0_completion] | 0.5000 |
| macro_f1[line_0_kind] | 0.8889 |
| macro_f1[line_0_owner_declined] | 0.8889 |
| macro_f1[line_0_rate_differs] | 0.8889 |
| macro_f1[line_0_rebilled] | 0.8889 |
| macro_f1[line_0_scope] | 0.8889 |
| macro_f1[line_0_unexplained_fee] | 0.8889 |
| macro_f1[line_1_completion] | 0.5000 |
| macro_f1[line_1_kind] | 0.8889 |
| macro_f1[line_1_owner_declined] | 0.8889 |
| macro_f1[line_1_rate_differs] | 0.8889 |
| macro_f1[line_1_rebilled] | 0.8889 |
| macro_f1[line_1_scope] | 0.8889 |
| macro_f1[line_1_unexplained_fee] | 0.8889 |
| macro_f1[line_2_completion] | 1.0000 |
| macro_f1[line_2_kind] | 1.0000 |
| macro_f1[line_2_owner_declined] | 1.0000 |
| macro_f1[line_2_rate_differs] | 1.0000 |
| macro_f1[line_2_rebilled] | 1.0000 |
| macro_f1[line_2_scope] | 1.0000 |
| macro_f1[line_2_unexplained_fee] | 1.0000 |
| macro_f1[line_3_completion] | 0.4000 |
| macro_f1[line_3_kind] | 0.4000 |
| macro_f1[line_3_owner_declined] | 1.0000 |
| macro_f1[line_3_rate_differs] | 1.0000 |
| macro_f1[line_3_rebilled] | 1.0000 |
| macro_f1[line_3_scope] | 1.0000 |
| macro_f1[line_3_unexplained_fee] | 0.4000 |
| macro_f1[line_4_completion] | 1.0000 |
| macro_f1[line_4_kind] | 1.0000 |
| macro_f1[line_4_owner_declined] | 1.0000 |
| macro_f1[line_4_rate_differs] | 1.0000 |
| macro_f1[line_4_rebilled] | 1.0000 |
| macro_f1[line_4_scope] | 1.0000 |
| macro_f1[line_4_unexplained_fee] | 1.0000 |
| macro_f1[malicious_content_in_mailboxes] | 1.0000 |
| macro_f1[malicious_process_running] | 0.0000 |
| macro_f1[offers_evidence] | 1.0000 |
| macro_f1[open_to_offer] | 0.0000 |
| macro_f1[outbound_channel_active] | 1.0000 |
| macro_f1[price_basis] | 0.2857 |
| macro_f1[prior_0] | 0.0000 |
| macro_f1[prior_1] | 0.0000 |
| macro_f1[prior_2] | 0.0000 |
| macro_f1[prior_3] | 0.0000 |
| macro_f1[proposal_reply] | 1.0000 |
| macro_f1[refund_done_msg_0] | 1.0000 |
| macro_f1[refund_done_msg_1] | 1.0000 |
| macro_f1[refund_reason] | 0.6000 |
| macro_f1[reports_unauthorized] | 0.8889 |
| macro_f1[reports_unresolved] | 0.0000 |
| macro_f1[request_fulfilled] | 1.0000 |
| macro_f1[request_specificity] | 0.0000 |
| macro_f1[requests_human] | 0.7619 |
| macro_f1[safety_1__instructed_by_tool_output__1] | 1.0000 |
| macro_f1[safety_1__user_asked_or_agreed__1] | 1.0000 |
| macro_f1[safety_1__within_grant__1] | 1.0000 |
| macro_f1[safety_2__instructed_by_tool_output__2] | 1.0000 |
| macro_f1[safety_2__user_asked_or_agreed__2] | 1.0000 |
| macro_f1[safety_2__within_grant__2] | 1.0000 |
| macro_f1[secured_msg_0] | 1.0000 |
| macro_f1[secured_msg_1] | 1.0000 |
| macro_f1[sender_0] | 0.2500 |
| macro_f1[session_in_attacker_hands] | 0.3333 |
| macro_f1[shares_credentials] | 1.0000 |
| macro_f1[spread_beyond_initial_entity] | 0.3333 |
| macro_f1[statement_not_invoice] | 0.8889 |
| macro_f1[tax_two_rates] | 0.8889 |
| macro_f1[threat_chargeback_or_public] | 1.0000 |
| macro_f1[threat_legal_regulatory] | 1.0000 |
| macro_f1[too_ambiguous] | 0.3750 |
| macro_f1[unexplained_charges] | 0.3750 |
| macro_f1[unusual_urgency] | 0.8889 |
| macro_f1[urgency] | 0.3000 |
| macro_f1[user_asked_or_agreed] | 1.0000 |
| macro_f1[within_grant] | 0.0000 |
| ordinal_mae[churn_risk] | 0.0000 |
| ordinal_mae[evidence_strength] | 1.2000 |
| ordinal_mae[expressed_satisfaction] | 0.7500 |
| ordinal_mae[frustration] | 0.5000 |
| ordinal_mae[hardship] | 0.2500 |
| ordinal_mae[request_specificity] | 0.0000 |
| ordinal_mae[urgency] | 0.8000 |
| per_workflow_accuracy[agent_trace_observability] | 0.5385 |
| per_workflow_accuracy[customer_service] | 0.7283 |
| per_workflow_accuracy[invoice_processing] | 0.7337 |
| per_workflow_accuracy[security_incidents] | 0.4595 |
| valid_accuracy | 0.7647 |

## Agreement vs TypeSafe consensus

| workflow | agreement | common subset | TVD vs consensus |
| --- | --- | --- | --- |
| overall | 0.6767 | 0.6767 | n/a |
| agent_trace_observability | 0.5385 | n/a | n/a |
| customer_service | 0.7283 | n/a | n/a |
| invoice_processing | 0.7337 | n/a | n/a |
| security_incidents | 0.4595 | n/a | n/a |

## Per-field accuracy

| field | n | acc | majority |
| --- | --- | --- | --- |
| activity_ongoing | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 0.5000 |
| adjustment_duplicates_line | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| affected_scope | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 0.5000 |
| attribution | 3 | 0.0000 [0.0000, 0.5615] (wilson) † | 0.6667 |
| cancellation_reason | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| churn_risk | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| context_explains_activity | 5 | 0.0000 [0.0000, 0.4345] (wilson) † | 1.0000 |
| in_scope | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| intent_pair | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| malicious_process_running | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| open_to_offer | 2 | 0.0000 [0.0000, 0.6576] (wilson) † | 1.0000 |
| prior_0 | 4 | 0.0000 [0.0000, 0.4899] (wilson) † | 1.0000 |
| prior_1 | 4 | 0.0000 [0.0000, 0.4899] (wilson) † | 1.0000 |
| prior_2 | 3 | 0.0000 [0.0000, 0.5615] (wilson) † | 1.0000 |
| prior_3 | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| reports_unresolved | 5 | 0.0000 [0.0000, 0.4345] (wilson) † | 0.6000 |
| request_specificity | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| within_grant | 1 | 0.0000 [0.0000, 0.7935] (wilson) † | 1.0000 |
| expressed_satisfaction | 5 | 0.2000 [0.0362, 0.6245] (wilson) † | 0.4000 |
| first_bad_step | 3 | 0.3333 [0.0615, 0.7923] (wilson) † | 0.6667 |
| sender_0 | 3 | 0.3333 [0.0615, 0.7923] (wilson) † | 0.6667 |
| desired_outcome | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 0.6000 |
| evidence_strength | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 0.6000 |
| handoff_required | 5 | 0.4000 [0.1176, 0.7693] (wilson) † | 0.6000 |
| intent | 5 | 0.4000 [0.1176, 0.7693] (wilson) | 0.4000 |
| urgency | 5 | 0.4000 [0.1176, 0.7693] (wilson) | 0.4000 |
| approval_0 | 4 | 0.5000 [0.1500, 0.8500] (wilson) † | 0.7500 |
| attacker_modified_configuration | 2 | 0.5000 [0.0945, 0.9055] (wilson) † | 1.0000 |
| credentials_exposed | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| session_in_attacker_hands | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| spread_beyond_initial_entity | 2 | 0.5000 [0.0945, 0.9055] (wilson) | 0.5000 |
| bank_change_claimed_in_comms | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 0.8000 |
| billed_above_basis | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| claims_agent_error | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| claims_supported | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| different_entity | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 0.8000 |
| frustration | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.2000 |
| is_true_positive | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| price_basis | 5 | 0.6000 [0.2307, 0.8824] (wilson) | 0.6000 |
| too_ambiguous | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 0.8000 |
| unexplained_charges | 5 | 0.6000 [0.2307, 0.8824] (wilson) † | 0.8000 |
| line_3_completion | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.6667 |
| line_3_kind | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.6667 |
| line_3_unexplained_fee | 3 | 0.6667 [0.2077, 0.9385] (wilson) | 0.6667 |
| hardship | 4 | 0.7500 [0.3006, 0.9544] (wilson) | 0.5000 |
| refund_reason | 4 | 0.7500 [0.3006, 0.9544] (wilson) | 0.5000 |
| handed_off | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| issue_resolved | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| left_undone | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.6000 |
| line_0_completion | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| line_0_kind | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_0_owner_declined | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_0_rate_differs | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_0_rebilled | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_0_scope | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_0_unexplained_fee | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_1_completion | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.8000 |
| line_1_kind | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_1_owner_declined | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_1_rate_differs | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_1_rebilled | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_1_scope | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| line_1_unexplained_fee | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| reports_unauthorized | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| requests_human | 5 | 0.8000 [0.3755, 0.9638] (wilson) | 0.6000 |
| statement_not_invoice | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| tax_two_rates | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| unusual_urgency | 5 | 0.8000 [0.3755, 0.9638] (wilson) † | 1.0000 |
| already_compensated | 4 | 1.0000 [0.5101, 1.0000] (wilson) | 1.0000 |
| amount_vs_record | 4 | 1.0000 [0.5101, 1.0000] (wilson) | 1.0000 |
| attack_type | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| attacker_persistence_present | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 0.5000 |
| changes_terms | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| instructed_by_tool_output | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_2_completion | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_kind | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_owner_declined | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_rate_differs | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_rebilled | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_scope | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_2_unexplained_fee | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_owner_declined | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_rate_differs | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_rebilled | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_3_scope | 3 | 1.0000 [0.4385, 1.0000] (wilson) | 1.0000 |
| line_4_completion | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_kind | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_owner_declined | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_rate_differs | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_rebilled | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_scope | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| line_4_unexplained_fee | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| malicious_content_in_mailboxes | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| offers_evidence | 4 | 1.0000 [0.5101, 1.0000] (wilson) | 0.7500 |
| outbound_channel_active | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| proposal_reply | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| refund_done_msg_0 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| refund_done_msg_1 | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 0.5000 |
| request_fulfilled | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 0.6000 |
| safety_1__instructed_by_tool_output__1 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_1__user_asked_or_agreed__1 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_1__within_grant__1 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_2__instructed_by_tool_output__2 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_2__user_asked_or_agreed__2 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| safety_2__within_grant__2 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| secured_msg_0 | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
| secured_msg_1 | 2 | 1.0000 [0.3424, 1.0000] (wilson) | 1.0000 |
| shares_credentials | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| threat_chargeback_or_public | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| threat_legal_regulatory | 5 | 1.0000 [0.5655, 1.0000] (wilson) | 1.0000 |
| user_asked_or_agreed | 1 | 1.0000 [0.2065, 1.0000] (wilson) | 1.0000 |
