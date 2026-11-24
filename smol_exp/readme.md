# UAP on tiny VLMs — the plan in short

*Updated Nov 17, 2026*

## TL;DR

Here's the thing. We built one tiny, almost invisible noise pattern that you can add to basically any image, and a small AI that describes pictures (SmolVLM-256M) suddenly starts saying nonsense about it. Like, you show it a dog and it goes "a person in a white shirt". Same pattern, every image. That's a **universal adversarial perturbation** (UAP), and ours breaks about 7 out of 10 captions.

The interesting part: the model cuts every image into 17 pieces (16 zoomed-in tiles + 1 small "overview" tile), and our attack basically lives in that one overview tile. Block that tile and the attack mostly dies. That's our main finding.

## What the reviewer said

We sent the draft to a reviewer, and the summary was: **good idea and solid work, but we don't have enough evidence yet.** No one said it's wrong. They said we need to *back it up better*. Their main complaints, in plain words:

- **"How do you know it's not luck?"** We ran things once, on 200 images, with no error bars. We need stats.
- **"Captions changed, ok, but are they actually wrong?"** We only checked that the caption is different from before, not that it's false. We need real datasets with human answers.
- **"Does this only work on your one tiny model?"** Everything is SmolVLM. We need to try other models.
- **"There's a stronger attack out there you didn't try."** Hitting the middle of the model instead of the end (VEV-UAP) might hit harder, so we should test it.
- **"Some of your big claims skip a step."** The 17-tile story and the defense ranking need cleaner experiments to be convincing.

The good news: none of it means starting from zero. Most fixes reuse stuff we already ran, and the rest is a few hours of free Colab GPU each.

## The 5 parts

Five topics, one person each, and nobody waits on anybody. Everyone starts from the same saved results and works in their own folder, so you can go full speed from day one. I'm taking part 1 (critical fixes), and we pick the other four together.

### 1. Critical fixes

**Why it matters:** these are the review's most important points, the ones that could actually sink the paper. If the 17-tile story or the defense ranking has a hole, everything built on top of it falls apart. This part also includes writing the final story.

- Run a clean test that proves the attack really lives in the overview tile (swap tiles around and see what breaks)
- Figure out why the attack drops from 91% to 72% when we switch to the real image pipeline (is it the 8-bit rounding or the resizing?)
- Re-rank the defenses so every one faces the same strong attacker, so the comparison is fair
- Fix the methods section so anyone could rerun what we did
- Merge everyone's results into the report

### 2. Stats

**Why it matters:** right now our numbers have no error bars, so the reviewer can't tell if they're real or just luck. Good news: zero GPU needed, it's all math on the files we already saved.

- Add confidence intervals to every main number
- Run proper paired tests when we compare two things
- Clean up the human-check numbers (we spotted one of the reviewer's own numbers is wrong)
- Make one clean results table with counts everyone can trust

### 3. Re-runs

**Why it matters:** one run on one random setup is not convincing. Re-running with more seeds and testing the stronger "middle of the model" attack shows our results hold up and weren't just luck.

- Rerun the main attack with a couple of extra random seeds
- Try the middle-layer attack (VEV-UAP) and compare it to ours
- Redo the transfer checks to the bigger SmolVLM models with fair controls
- Double-check the JPEG defense numbers

### 4. New datasets

**Why it matters:** "the caption changed" isn't the same as "the caption is wrong". Testing on datasets with real human answers proves the model is actually being fooled, not just rephrasing.

- Run the attack on COCO and compare captions to human-written ones
- Try a hallucination benchmark (does the model start seeing stuff that isn't there?)
- Report how much worse the answers get, with real scores

### 5. New architectures

**Why it matters:** if it only works on one tiny model, it's a fun trick, not a finding. Testing other kinds of vision encoders (including JEPA, which is getting a lot of attention right now) tells us if this is a general thing.

- Attack three big image encoders with different training styles: I-JEPA, MAE and CLIP
- Measure the damage fairly across them (each encoder speaks a different "language", so we calibrate)
- Check if the middle layers are weaker than the final layer
- Later, for the paper: try a second family of VLMs (like Qwen2-VL)

## Timeline

Short version: **all experiments done by Friday Nov 21**, then two weeks to write it up and prepare the presentation.

1. **Tue Nov 17: kickoff.** Pick your part, grab the saved results and the detailed brief, start running stuff.
2. **Fri Nov 21: everything's in.** 30-min sync, everyone shows their numbers and what surprised them, we decide together what it all means.
3. **Nov 22 to early December: polishing.** Final report, slides, and practicing the presentation.

With an AI helping you code, most of each list is about one full day of work, so try not to leave it for the last night.

## What we ship at the end

- **The final report:** same story as now, but every claim backed by stats, real datasets and more than one model. Basically, the version that answers the review.
- **The slides + presentation:** the attack, the 17-tile twist, how defenses hold up, and what changes on other models.
- **A clean code folder:** clean notebooks (already done) plus everyone's results, so anyone can rerun it.

And if it all goes well, after the course we turn this into a real paper and submit it to a workshop. Your name on a paper.

I'll share the detailed to-do list for each part right after this, so don't worry about the specifics yet.