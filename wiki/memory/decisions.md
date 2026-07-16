# Decision records

Append-only. One entry per decision: context, options, the call, and why.

## 2026-07-15 18:24 UTC — tags: labels, track-a

Track A is 5-class (not the committed reference's 4-class RC->ET merge via Y[Y==4]=3): reproducing the merge would break the held-constant-architecture rule (out_channels would differ from Track B) and make RC Dice unmeasurable. Track A reproduces the DATA-pipeline defects at 5 classes.

## 2026-07-15 19:58 UTC — tags: fetch, downloads, robustness

Kaggle downloads must set a socket read timeout (socket.setdefaulttimeout(90) in 01_fetch_subsample.py). The client sets none, so a stalled HTTPS recv() blocks a worker thread forever and the ThreadPoolExecutor never completes — observed as a 20-min 0%-CPU hang at 698/700 masks that would have recurred on the 1500-file modality stage. The existing 4-attempt backoff loop only helps once the call raises; the timeout is what makes it raise.
