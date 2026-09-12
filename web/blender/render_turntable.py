"""
Render the padlock turntable to public/turntable/frame_0001.webp … frame_0120.webp.

    npm run render
    # or, by hand:
    blender -b blender/padlock.blend --python blender/render_turntable.py
    # options after `--`: --frames 1,60  --scale 50  --out DIR  --format PNG

The scene (Padlock_Turntable) already carries the camera, the studio lights, the
packed HDR world, a transparent film and the 360° key on Lock_Body over frames
1–120; this script only pins the output size, format and path.
"""
import os, sys
import bpy  # type: ignore  (only resolvable inside Blender)

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
opt = dict(zip(argv[::2], argv[1::2]))
scale = int(opt.get('--scale', 100))
frames = [int(f) for f in opt['--frames'].split(',')] if '--frames' in opt else None
fmt = opt.get('--format', 'WEBP')
root = os.path.dirname(os.path.dirname(os.path.abspath(bpy.data.filepath)))
out = opt.get('--out', os.path.join(root, 'public', 'turntable'))
os.makedirs(out, exist_ok=True)

s = bpy.context.scene
r = s.render
r.resolution_x, r.resolution_y = 900, 1125   # the scrubber's canvas size
r.resolution_percentage = scale
r.film_transparent = True
r.image_settings.file_format = fmt
r.image_settings.color_mode = 'RGBA'
r.image_settings.quality = 90                # WebP quality; the lock is ~30 KB a frame
r.image_settings.color_depth = '8'
r.use_file_extension = True

# The key on Lock_Body reaches 360° at frame 121 (loop convention), so frame 120
# would sit 3° short of front — and that is the pose the lock holds while the tap
# hint shows. Sample the curve so frame 1 and frame 120 are both exactly front.
N = 120
for f in frames or range(1, N + 1):
    t = 1 + (f - 1) * N / (N - 1)
    s.frame_set(int(t), subframe=t - int(t))
    r.filepath = os.path.join(out, 'frame_%04d' % f)
    bpy.ops.render.render(write_still=True)
print('rendered to', out)
