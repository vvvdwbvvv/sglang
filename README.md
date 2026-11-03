# fork-sync

This branch is the fork's default branch. It holds no code and no workflows.

- The code lives on [`main`](../../tree/main), which is force-updated every day
  to match [sgl-project/sglang](https://github.com/sgl-project/sglang) `main`.
- The sync runs locally on the owner's Mac through a launchd agent
  (`com.vvvdwbvvv.sglang-sync`, daily 08:07 local time). It runs
  `gh repo sync vvvdwbvvv/sglang --source sgl-project/sglang --branch main --force`.
- Do not commit to `main` on this fork; anything pushed there is overwritten by the next sync.
- Trigger a sync by hand: run `gh repo sync vvvdwbvvv/sglang --source sgl-project/sglang --branch main --force`.
