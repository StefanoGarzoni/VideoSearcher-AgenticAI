# Public repository — publication checklist

This checklist distinguishes completed *repository hygiene* from **rights/consent verification that cannot be completed from code alone**. Publishing a repository does not grant licenses for included material.

## Changes included in the public-release patch

- [x] Remove the direct link and scripted download for the historical media archive from the current tree.
- [x] Remove historical TED transcript-derived JSON files from the current tree.
- [x] Remove unverified video-derived JPEG reference frames and segmentation demonstration image from the current tree.
- [x] Document the original fixture names without redistributing source media.
- [x] Include the `llama.cpp` MIT notice for vendored/modified Jinja chat templates.
- [x] Update public-visibility language and remove scripts to create a new private repository.

## Required outside Git (not claimed as complete)

- [ ] **Confirm contributor permission:** check publication and per-file ownership with the other project authors. Listing authors is not equivalent to consent.
- [ ] **Verify source/code rights:** ensure any third-party code, images, graphics, datasets, and external portions of the academic deliverables may be publicly distributed.
- [ ] **Disable unintended hosted access:** if you own the historical Google Drive archive, restrict or remove public link-sharing. Removing its URL from this repository does **not** disable the file's sharing settings.
- [ ] **Review presentations and derived experimental records:** PDF/PPTX slide media and structured extracts may contain copyrighted imagery/text and need an independent rights check.
- [ ] **Consider Git history:** the **first public commit remains reachable** after a normal follow-up commit and contains the formerly published external URL, frames and historical TED text. If those materials must be inaccessible from GitHub history, coordinated Git-history rewriting and cache/support handling may be necessary. This patch deliberately **does not force-push or rewrite history**.

## Licensing note

There is intentionally **no project-wide `LICENSE`**. Public read access alone is not an explicit open-source permission to modify or redistribute original project source. The upstream MIT notice applies to the attributed `llama.cpp`-derived template portions only.

## Reproducibility note

Structured ground-truth summaries and numerical benchmark results are retained as historical measurements. Removing uncertain-licensing media artifacts and omitting the private VLM framework means full end-to-end reproduction is **not** supported. The CPU-only metric tests remain the supported executable verification path.
