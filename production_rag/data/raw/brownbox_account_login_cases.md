---
doc_id: brownbox_account_login_cases
title: BrownBox Account Login and Verification Casebook
business_domain: customer_support
doc_type: open_dataset_casebook
version: dataset_seed_v1
effective_from: 2026-06-01
effective_to:
permission_scope: internal
owner: support_ops_team
dataset_source: rjac/e-commerce-customer-support-qa
dataset_license: MIT
---

# BrownBox Account Login and Verification Casebook

This casebook expands the practice knowledge base with production-like account access support cases. It follows the issue taxonomy used by the open e-commerce customer support QA dataset and is written as long-form support knowledge so retrieval, BM25, RRF, rerank, truncation, deduplication, and citation validation have enough competing evidence.

## Case 1: Mobile number or email verification during login

A customer may be blocked during checkout because the account prompts for mobile number or email verification. Support should first confirm whether the customer is trying to log in with a registered email, mobile number, or social sign-in. If the registered contact channel is still accessible, send a fresh verification code and ask the customer to complete the login within the code validity window. If the customer cannot access the registered channel, the agent must verify order history, last delivery address, and masked payment method before changing the contact channel.

Do not bypass verification because the customer says the purchase is urgent. A successful answer should explain the verification requirement, the supported recovery path, and the security reason behind it. If the customer is attempting to buy a high-value appliance or electronics item, the support note should include that account verification protects order, refund, and warranty records.

## Case 2: Too many verification attempts

When a customer receives an error that they exceeded the number of attempts to enter the verification code, the account is temporarily rate limited. The standard resolution is to wait 30 minutes before requesting a new code. The agent should check whether the customer is repeatedly requesting codes from multiple devices, because multiple active requests can invalidate the previous code.

If the customer is locked out during an active return or replacement request, the agent can create a manual follow-up ticket, but the login lock should not be removed without identity verification. The support answer should mention the waiting period, explain why old codes may fail, and provide an escalation path only after verifying identity.

## Case 3: Reactivating an inactive account

Accounts inactive for six months may require reactivation before purchase, return, or invoice retrieval. The agent should verify the customer's registered email or mobile number, recent order ID, and one non-sensitive account detail. After successful verification, the account can be reactivated and the customer can reset the password.

If the customer cannot provide enough identity evidence, the agent must refuse account reactivation and route the case to account security review. A good RAG answer should not simply say "reactivate the account"; it should include the required identity checks and the reason support cannot disclose account details before verification.

## Case 4: Login fails after contact change

Customers sometimes update their email or mobile number and then cannot log in because they are using the old credential. Support should confirm the timestamp of the contact change and ask the customer to try the new credential. If the contact change was made by support, the agent should confirm that the change ticket is closed and that verification was completed.

If the customer suspects unauthorized change, support must freeze sensitive account operations such as refunds, wallet withdrawal, and address edits until account security review is complete.
