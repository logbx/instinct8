# Instinct8 Brag Video

This directory contains the source files for the Instinct8 launch video.

## Media Files (Hosted Externally)

The actual video and poster files are not committed to the repository to keep the repo size small.

**Video and poster will be uploaded and linked in PR #24.**

## Source Files

- `brag-plan.md` - Creative plan and storyboard
- `composition-brief.md` - Hyperframes composition specification  
- `composition/index.html` - Hyperframes composition source
- `share-copy.txt` - Social media caption

## Rendering

To recreate the video:

1. Copy a music track to `composition/assets/music/track.mp3` (2MB)
2. Run: `cd composition && npx hyperframes render --quality high --output ../brag.mp4`
3. Extract poster: `ffmpeg -ss 15.5 -i brag.mp4 -frames:v 1 -q:v 2 brag.jpg`
4. Bake poster as frame 0: 
   ```bash
   ffmpeg -y -i brag.mp4 -i brag.jpg \
     -filter_complex "[0:v][1:v]overlay=0:0:enable='eq(n,0)'[v]" \
     -map "[v]" -map 0:a? -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \
     -c:a copy -movflags +faststart brag.poster.mp4 && mv brag.poster.mp4 brag.mp4
   ```

Output: `brag.mp4` (983KB, 20s, 1920x1080) + `brag.jpg` (51KB poster)
