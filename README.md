# Syncytia Model

Python ODE model for fitting digitized syncytia-formation and cell-index curves.

The repository contains the fitting script, the original figure CSVs in `Fig.csvs/`,
and the digitized/combined CSVs in `figure_digitized_csvs/`.

## Model

`syncytia-model_christmas.py` uses an Erlang two-step fusion model with five
living compartments:

- `D`: donor cells
- `A`: acceptor cells
- `F1`: first fusion intermediate
- `F2`: second fusion intermediate
- `S`: syncytia

Each curve fits four parameters:

- `donor_fraction`: initial donor proportion
- `gamma`: fusion-initiation rate
- `k_val`: Erlang transition rate
- `delta`: death/loss rate applied to every living compartment

The plotted model signal is `S / (D + A + F1 + F2)`. A fitted line is omitted
when a dataset contains fewer than two distinct time points.

## Usage

```bash
python syncytia-model_christmas.py --data "Fig.csvs/*.csv"
```

Optional flags are `--max_time`, `--no_baseline`, `--no_normalize`, and
`--invert`. The script groups input curves by figure panel and writes one PNG
per panel.
