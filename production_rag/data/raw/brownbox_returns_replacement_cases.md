---
doc_id: brownbox_returns_replacement_cases
title: BrownBox Returns and Replacement Casebook
business_domain: customer_support
doc_type: open_dataset_casebook
version: dataset_seed_v1
effective_from: 2026-06-01
effective_to:
permission_scope: internal
owner: returns_team
dataset_source: rjac/e-commerce-customer-support-qa
dataset_license: MIT
---

# BrownBox Returns and Replacement Casebook

This document collects longer return and replacement scenarios. The cases are intentionally similar but not identical, which is useful for testing semantic deduplication and dynamic truncation after rerank.

## Case 1: Unable to click the cancel button

A customer may report that the cancel button is disabled for a recently purchased juicer, mixer, grinder, or similar appliance. Support should first check the order state. If the item has already moved to packed, shipped, replacement, or pickup scheduled status, the self-service cancel action may be unavailable. The customer can still request a return after delivery if the item is eligible.

If the order is already part of a replacement workflow, cancellation rules differ from ordinary purchase cancellation. The agent should explain that the disabled button is not necessarily a technical defect; it may reflect the current fulfillment or replacement state. A helpful answer should tell the customer what action is still possible: wait for delivery and initiate return, contact support to stop shipment if not dispatched, or continue with the replacement if the original item was defective.

## Case 2: Replacement denied after policy window

If a customer asks why they cannot get a replacement for a faulty vacuum cleaner after several months, support must check delivery date and replacement window. BrownBox replacement policy allows replacement only within the eligible return or replacement period stated on the product page and invoice. If the customer contacts support after the policy window, the request may be denied even if the item later develops a fault.

The agent should avoid blaming the customer. The answer should explain the timeline, distinguish replacement from warranty repair, and route the customer to manufacturer warranty if available. If the issue is safety-related, support should still capture product model, batch number, and failure symptoms for quality review.

## Case 3: Return pickup failed twice

When return pickup fails twice, support should verify pickup address, customer availability window, and packaging readiness. If the courier marked customer unavailable but the customer disputes it, create a logistics dispute ticket and schedule one more pickup attempt. If pickup fails because the product is not packed with accessories, the agent should explain the packaging requirement before rescheduling.

The answer should cite pickup rules, not just tell the customer to wait. For high-value electronics, support may require serial number photos before the final pickup attempt.

## Case 4: Exchange versus refund confusion

Some customers ask for replacement but expect money to return to the original payment method. Support should clarify that replacement means another item is sent, while refund means payment is returned after pickup and inspection. If the customer wants refund instead of exchange, the agent should cancel the replacement workflow before the replacement shipment is released.
