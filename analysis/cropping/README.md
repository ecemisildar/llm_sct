# Arena-only video copies

Original videos remain in `videos/exploration/` and `videos/patrolling/`.
Processed copies keep the same filenames in each group's `cropped/` subfolder.

The fixed camera has a slightly skewed arena boundary. The manually selected
inner floor corners (1280 × 720 source pixels, clockwise from top left) are:

- Top left: (258, 90)
- Top right: (1000, 78)
- Bottom right: (1020, 617)
- Bottom left: (275, 633)

Corners are slightly inset to exclude the white walls. Perspective correction
maps this quadrilateral to an 800 × 600 rectangle. These are approximate manual
boundaries, rather than a surveyed calibration. The user specified an arena area of approximately 12 m². The 4:3 output
aspect ratio follows the existing `corners.json` room dimensions of 4 × 3 m.
Thus the cropped output is treated as 200 pixels/m, with each pixel representing
0.000025 m². This is an approximate physical calibration. Values and the source
to output homography are saved in `arena_calibration.json`.
For percentage coverage, use this same arena region consistently across trials.

Exports use H.264 CRF 18 and preserve source frames and timing, with audio copied
if present. The script checks packet/frame count, nominal frame rate, duration,
output dimensions, and that original file sizes and modification times remain
unchanged. Completed export metadata is recorded in `crop_manifest.json`.

Run `python3 analysis/cropping/crop_videos.py` from the experiment directory to
process missing copies. Existing copies with matching frame counts are verified
and retained. The preview images show the selected arena region.
