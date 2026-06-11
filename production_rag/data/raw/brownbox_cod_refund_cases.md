---
doc_id: brownbox_cod_refund_cases
title: BrownBox Cash on Delivery Refund Casebook
business_domain: customer_support
doc_type: open_dataset_casebook
version: dataset_seed_v1
effective_from: 2026-06-01
effective_to:
permission_scope: internal
owner: payment_ops_team
dataset_source: rjac/e-commerce-customer-support-qa
dataset_license: MIT
---

# BrownBox Cash on Delivery Refund Casebook

Cash on Delivery refunds are a frequent source of bad RAG answers because the refund path is different from prepaid orders. This file gives the retrieval pipeline several long, overlapping cases around refund method, timeline, bank validation, and failed transfer.

## Case 1: COD refund timeline after pickup

For Cash on Delivery orders, the refund is not returned to a card automatically because the customer paid in cash. After return pickup, BrownBox completes quality check first. If the item passes inspection, the refund is initiated to the bank account, UPI ID, wallet, or payout method collected from the customer. The usual processing time is 3 to 7 business days after successful quality check and payout detail validation.

If the customer asks "when will my COD refund arrive", the answer should include the dependency on pickup completion, inspection, and payout validation. Do not reuse prepaid refund timelines for COD orders.

## Case 2: Invalid bank account details

If a COD refund fails because the bank account number, IFSC code, UPI ID, or account holder name is invalid, support must request corrected payout details through the secure refund detail form. Agents should not collect full bank details in free-text chat if the secure form is available. After correction, the payout timeline restarts from the validation date.

The answer should mention that the failed transfer is not the same as a rejected refund. The refund remains approved, but payout cannot complete until the customer submits valid details.

## Case 3: Customer wants cash refund at pickup

Courier partners do not hand out cash refunds during return pickup. The pickup agent can collect the product and update pickup status, but refund processing is handled by BrownBox payment operations after inspection. If the customer refuses pickup because they expect instant cash, support should explain the process and reschedule pickup if needed.

This case is useful for citation validation because the answer must cite the pickup rule and refund processing rule together.

## Case 4: COD refund stuck after quality check

If quality check passed but refund is still pending beyond 7 business days, support should verify payout status in the payment operations dashboard. If the payout is marked failed, request corrected details. If payout is marked processing, create a payment investigation ticket with order ID, return ID, payout reference, and customer contact channel.
