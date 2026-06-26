"""
wegofwd_llm/trust.py

The Content Trust Manifest — a customer-facing record of HOW a piece of curated
content was produced and which quality/compliance gates it passed (ADR-015).

WHY this lives in the seam: the two hardest-to-fake trust signals — *which
verified model produced the content* and *whether it passed schema conformance*
— are already known only here (registry.provenance + conformance.generate_validated).
Every consumer (Mentible, StudyBuddy OnDemand, Kathai Chithiram, Pramana) needs
to surface them, so the SHAPE of the whole manifest is owned here even though the
seam fills only the blocks it can vouch for. The product packager supplies the
rest (format-compliance score, content hash/signature, citations, human-approval,
data policy) and assembles the full manifest from the same dataclasses, so a
manifest serialises identically across every product and stack.

SUBTRACTIVE, like the rest of the seam: this module emits DATA only. It renders
nothing, stamps no clock (the caller passes `generated_at`), and reads no I/O.

CUSTOMER-FACING SAFETY: a manifest is meant to be shipped to a front-end and shown
to an end user. No field here ever holds key material, a prompt, or a raw vendor
payload — only the provenance/validation/compliance facts. `to_public_dict()` is
the serialisation boundary; keep it that way.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from wegofwd_llm.registry import provenance

# Version of the manifest SHAPE defined in this module. Bump when a stored manifest
# or a front-end renderer must notice a change (new required block, changed
# semantics). Distinct from LLM_CONTRACT_VERSION (the request/response seam) and a
# provider's integration_version. Stamped into every manifest so a stale renderer
# can detect it is reading a newer shape than it understands.
TRUST_MANIFEST_VERSION = 1


# --- Blocks the SEAM can vouch for (filled by engine_trust) --------------------


@dataclass(frozen=True)
class ProvenanceBlock:
    """Which verified model/provider produced the content, and under which
    integration/contract versions. Lifted from registry.provenance() so there is
    one source of truth. `generated_at` is caller-stamped (ISO-8601 UTC) — the
    seam owns no clock."""

    provider: str
    model: str
    model_verified: bool
    integration_version: int
    contract_version: int
    generated_at: str | None = None


@dataclass(frozen=True)
class ValidationBlock:
    """Whether the generation passed the validate→repair conformance loop
    (conformance.generate_validated). `repair_attempts` = 0 means it validated
    first try; a non-zero count is still a pass, just a costlier one."""

    schema_validated: bool
    repair_attempts: int = 0
    schema_id: str | None = None  # e.g. "lesson@1", "course-module@2"


# --- Blocks the PRODUCT supplies (the seam defines the shape, not the values) ---


@dataclass(frozen=True)
class ComplianceBlock:
    """Result of the product's format/brand compliance check (e.g. Mentible's
    13-parameter `doc format.xlsx` ruleset). `status` is the headline a badge
    shows; the counts let the UI render "11/13"."""

    ruleset: str  # "mentible-professional@1.0"
    checks_passed: int
    checks_total: int
    status: str  # "pass" | "pass_with_notes" | "fail"


@dataclass(frozen=True)
class IntegrityBlock:
    """Tamper-evidence for the content body (ADR-011 Consumable Package)."""

    content_hash: str  # "sha256:…"
    signed: bool = False


@dataclass(frozen=True)
class SourcingBlock:
    """Traceability of claims to a cited source (curriculum, regulation clause)."""

    every_claim_cited: bool
    source_refs: int = 0


@dataclass(frozen=True)
class ReviewBlock:
    """Human approval gate. `approver_distinct_from_generator` is the SoD signal
    an auditor cares about (ADR-011 §7) — the approver was not the generator."""

    human_approved: bool
    approver_distinct_from_generator: bool = False


@dataclass(frozen=True)
class PolicyBlock:
    """The standing data posture under which the content was made (PARAMETERS §5,
    ADR-001). Product-level, not per-artefact, but carried on the manifest so a
    badge can state it inline."""

    byok: bool
    prompts_stored: bool
    key_stored: bool


# --- The assembled manifest ----------------------------------------------------


@dataclass(frozen=True)
class ContentTrustManifest:
    """The full customer-facing trust record for one unit of curated content.

    The seam fills `provenance` + `validation` (engine_trust); the product
    packager attaches the optional blocks it can vouch for. A block left None is
    rendered as "not assessed" by the front-end — never as a pass."""

    trust_manifest_version: int
    provenance: ProvenanceBlock
    validation: ValidationBlock
    compliance: ComplianceBlock | None = None
    integrity: IntegrityBlock | None = None
    sourcing: SourcingBlock | None = None
    review: ReviewBlock | None = None
    policy: PolicyBlock | None = None

    def to_public_dict(self) -> dict:
        """Serialise for shipping to a front-end. Drops unset (None) blocks so the
        UI distinguishes "not assessed" from a false value. This is the ONLY
        sanctioned serialisation boundary — every field here is non-secret by
        construction (provenance/validation/compliance facts), so the manifest is
        safe to embed in a client payload."""
        out = asdict(self)
        return {k: v for k, v in out.items() if v is not None}


def engine_trust(
    provider_id: str,
    model: str | None = None,
    *,
    schema_validated: bool,
    repair_attempts: int = 0,
    schema_id: str | None = None,
    generated_at: str | None = None,
) -> ContentTrustManifest:
    """Emit the part of the manifest the SEAM can vouch for: provenance (reused
    from registry.provenance, so model-verification status stays single-sourced)
    plus the conformance-loop outcome. The product packager then attaches the
    compliance/integrity/sourcing/review/policy blocks via dataclasses.replace.

    `generated_at` is caller-supplied ISO-8601 UTC — the seam stamps no clock."""
    prov = provenance(provider_id, model)
    return ContentTrustManifest(
        trust_manifest_version=TRUST_MANIFEST_VERSION,
        provenance=ProvenanceBlock(**prov, generated_at=generated_at),
        validation=ValidationBlock(
            schema_validated=schema_validated,
            repair_attempts=repair_attempts,
            schema_id=schema_id,
        ),
    )
