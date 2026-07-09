# Third-Party Notice

This directory vendors the modified EpiABM simulator code used for the CRIPO
EpiCF experiments so that reviewers can run the data-generation pipeline without
applying patches to a separate checkout.

The upstream EpiABM code is based on:

- Repository: `https://github.com/complex-ai-lab/epi-diff-abm`
- Base commit used before CRIPO modifications: `824ca2a9785038eaec4e277903856d796ac4adb3`

The upstream README states that its `agent_torch/` and `covid_abm/` directories
are based on AgentTorch:

- Repository: `https://github.com/AgentTorch/AgentTorch`
- License: GNU Affero General Public License v3.0
- License text included in `LICENSE.AGENTTORCH.md`

CRIPO modifications include dynamic population loading, online intervention
injection, continuous freezing-interval actions, multi-county preparation
helpers, and API retry/direct-fetch support. Generated data and calibrated
assets are intentionally not included.
