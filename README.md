# Syncytia Model

Python-based ODE model for fitting digitized experimental curves of syncytia formation and cell index dynamics.

This repository contains:
- Digitized CSV data extracted from published figures
- A mechanistic ODE model with fusion intermediates and cell death
- Single-curve and multi-curve fitting with shared biological parameters

---

## Model Summary

The model describes syncytia formation using an Erlang two-step fusion process with an explicit death term.

### State Variables
- **D** — Donor cells  
- **A** — Acceptor cells  
- **F1** — Fusion intermediate (stage 1)  
- **F2** — Fusion intermediate (stage 2)  
- **S** — Syncytia  
- **X** — Dead / lost cells  

### Key Parameters
- `gamma` — Fusion initiation rate  
- `k` — Erlang transition rate  
- `delta` — Death / loss rate  
- `rA` — Optional acceptor growth rate  
- `K` — Carrying capacity  
- `gamma2` — Secondary fusion rate  

Observed signal is modeled as:
