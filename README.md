# More Links, Same Airtime

[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23574119.svg)](https://doi.org/10.5281/zenodo.23574119)
[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23574129.svg)](https://doi.org/10.5281/zenodo.23574129)
[![arXiv](https://img.shields.io/badge/arXiv-2602.10823-b31b1b.svg)](https://arxiv.org/abs/2602.10823)

Code for *More Links, Same Airtime: Spending a Shared Channel's Sounding Budget on WiFi Sensing*
(Karim Khamaisi and Bruno Rodrigues, Sensing Group, University of St. Gallen).

Under IEEE 802.11bf, every pair of devices that senses people draws its soundings from the airtime
of one shared channel. The paper measures how a fixed sounding budget should be split: a budget
detects people best when spread over as many pairs as leave each pair about two soundings per
decision, and receivers that overhear a sounding add evidence at no airtime cost. Spreading costs
calibration (label-free detectors need a longer empty-room reference) and privacy (an adaptive
schedule reveals occupancy).

<p align="center">
  <img src="https://github.com/user-attachments/assets/a0301509-113c-4c70-9528-9eec0db3c313" width="400">
</p>

**Dataset:** CSI from a 14-node office mesh (27 sessions) and six scripted trials in a living room
and a laboratory, on Zenodo: https://doi.org/10.5281/zenodo.23574119. This code is archived at
https://doi.org/10.5281/zenodo.23574129.

## Quick start

```
git clone https://github.com/sensing-group/more-links-same-airtime
cd more-links-same-airtime
pip install -r requirements.txt
cd paper && for s in fig_*.py make_tab_*.py; do python3 $s; done   # figures and tables into ../figures/
```

This redraws every plotted figure and Tables 3 and 4 from the shipped analysis outputs in
`results/`, without the dataset. To rerun the analyses themselves, fetch the dataset (21 GB) into
`data/` and run `run_all.sh`:

```
mkdir -p data && cd data
for f in office-sessions-1.tar office-sessions-2.tar office-sessions-3.tar office-features-labels.tar trials.tar SHA256SUMS; do
  curl -L -o $f "https://zenodo.org/records/23574119/files/$f?download=1"
done
sha256sum -c SHA256SUMS && for t in *.tar; do tar xf $t; done
cd .. && ./run_all.sh
```

## Earlier version

This repository first accompanied the preprint
[*Less is More: The Dilution Effect in Multi-Link Wireless Sensing*](https://arxiv.org/abs/2602.10823),
which reported that a single well-placed link outperformed the full mesh. The paper above revisits
that question at equal airtime and with sound evaluation, and finds the opposite. Its §8 measures
the evaluation practices that produce an apparent dilution effect, and its §10 explains why the
preprint's apartment recording is not used. The preprint's recorder and analysis are kept in
`tdma-mesh/` and at tag `v1-dilution-preprint`.

License: MIT.

## Layout

| Folder | Contents |
|---|---|
| `scripts/` | Feature extraction and every analysis behind the paper's numbers |
| `paper/` | Figure and table scripts; they read only `results/` |
| `results/` | The analysis outputs the paper reports (JSON) |
| `capture/` | The recorder and ESP32 firmware that captured the scripted trials, at commit `618c876`, with WiFi credentials replaced by placeholders |
| `ghost_eval/` | Memory limit helper used by the scripts |
| `data/` | Place the dataset here: `data/office/sessions/`, `data/office/features/` and `data/trials/` |
| `tdma-mesh/` | The earlier TDMA recorder and the preprint's analysis (see its README) |

## Citation

See `CITATION.cff`. The preprint can be cited as arXiv:2602.10823.
