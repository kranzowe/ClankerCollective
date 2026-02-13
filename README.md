# ClankerCollective
Advanced robits

## Repository Structure

This is the integration repo that ties together all subsystems. Each subsystem is a separate Git repository included as a submodule.

```
ClankerCollective/
├── subsystems/
│   ├── Autonomy/    — Autonomy subsystem
│   ├── Estimation/  — Estimation subsystem
│   └── Controls/    — Controls subsystem
└── ...
```

## Getting Started

Clone with submodules:
```bash
git clone --recurse-submodules https://github.com/kranzowe/ClankerCollective.git
```

If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

## Updating Submodules

Pull latest changes from all subsystems:
```bash
git submodule update --remote --merge
```
