# Labeling Guide for Partners

Thanks for helping. You'll draw boxes around every vehicle in about 200 CCTV frames. The tool starts each frame with machine guesses, and your job is to fix them: delete wrong boxes, fix classes and add the vehicles the machine missed. Missed small cars and pickups called "truck" are the two big problems we're fixing, so those deserve the most attention.

## Setup (once)

1. Install **Python 3.9 or newer** from python.org. Nothing else is needed: no GPU, no pip packages.
2. Get the code. Clone the repo and check out the branch we're using (or `git pull` if you already have it).
3. Download `manual_v1_pack.zip` from the Google Drive link and unpack it from the `smart-traffic-vision` folder (use the path the file was actually downloaded to):

   ```bash
   python tools/manual_label/sync.py receive ~/Downloads/manual_v1_pack.zip
   ```

   This creates `data/manual_v1/` with the frames list, images and machine pre-labels.

## Label

From the `smart-traffic-vision` folder:

```bash
python tools/manual_label/label_server.py --pack data/manual_v1 --user friend
```

Your browser opens the labeler. You only ever see **your own** frames, so we can't both label the same image. Everything saves automatically to `data/manual_v1/work/friend/`. Close the terminal window (or press Ctrl+C) when you're done for the day.

Your queue starts with **6 calibration frames** that both of us label. Do those first, on your own, and send them over (see below) so we can compare and agree on the rules before doing the rest.

## The rules

1. **Box every vehicle you can recognise as a vehicle**: moving, waiting, parked at the roadside or in a lot, tiny in the background. Zoom in (mouse wheel) to check the horizon and the gaps between big vehicles. If you can tell it's a vehicle but not which kind, pick the most likely class (usually car).
2. **car** (`1`): sedans, hatchbacks, taxis, SUVs, **every 1-ton pickup** (Hilux, D-Max, Navara, Ranger, Triton), high-cage pickups (รถคอก), **passenger vans** (Commuter/HiAce), and **songthaews built on a pickup**.
3. **truck** (`4`): 6-, 10- and 18-wheelers (one box for cab + trailer), box/delivery trucks with a separate cab, dump trucks, mixers, and songthaews on a truck chassis. *Dual rear wheels or a box body = truck. Pickup bed = car.*
4. **bus** (`3`): city buses, coaches, buses with a bus body. Vans are car.
5. **motorcycle** (`2`): one box around bike + rider(s). Parked bikes count. Bicycles get no box.
6. **three_wheeler** (`5`): tuk-tuks and salengs (motorbike with sidecar/cargo bucket).
7. **Tight boxes** around what's visible. A vehicle mostly hidden behind another: box the visible part if at least ~25% shows.
8. **Dashed boxes with a yellow ?** are guesses the machines disagreed on. Press `N` to jump to the next one, fix it or press `C` if it's right. Solid boxes are guesses too, so look at every one of them.
9. No boxes for people, animals, bicycles, pushcarts, or vehicles on billboards.
10. Broken frame (frozen, glitched, camera moved)? Press **Skip frame** and write why.

## Keys

| | |
| --- | --- |
| Draw a box | drag on the image |
| Pick class (also changes the selected box) | `1` car · `2` motorcycle · `3` bus · `4` truck · `5` three_wheeler |
| Select / cycle overlapping boxes | click / click again |
| Move / resize | drag inside the selected box / drag its handles |
| Delete · Undo · Redo | `Del` · `Ctrl+Z` · `Ctrl+Y` |
| Next `?` box · it's correct | `N` · `C` |
| Zoom · Pan · Fit | wheel · right-drag or `Space`+drag · `F` |
| Peek under the boxes · brighten night frames | hold `H` · `B` |
| **Frame done, go to next** | `Ctrl+Enter` |

The **? Help** button in the top right repeats all of this.

## Sending your work back

Whenever you finish a session, run this from the `smart-traffic-vision` folder and upload the file it prints (`data/manual_v1_work_friend.zip`) to the shared Drive folder:

```bash
python tools/manual_label/sync.py send --pack data/manual_v1 --user friend
```

It only contains your own labels, so it never overwrites anyone else's work.

To review my frames, download my `manual_v1_work_thiramet.zip`, unpack it the same way, then click **Review partner** in the labeler and mark each frame 👍 or ⚠ with a comment:

```bash
python tools/manual_label/sync.py receive ~/Downloads/manual_v1_work_thiramet.zip
```

Your comments are saved in *your* folder, so they reach me with your next zip. Never edit files in someone else's folder.
