"""Create arena-only copies; never modify source videos."""
import concurrent.futures
import json
import uuid
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2] / 'videos'
FILTER = ('crop=780:568:250:72,'
          'perspective=x0=8:y0=18:x1=750:y1=6:x2=25:y2=561:x3=770:y3=545:sense=source,'
          'scale=800:600,setsar=1')

def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-count_packets', '-show_entries',
        'stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_read_packets:format=duration',
        '-of', 'json', str(path)]))

def process(source):
    destination = source.parent / 'cropped' / source.name
    destination.parent.mkdir(exist_ok=True)
    before = source.stat()
    partial = destination.with_name(destination.stem + '.' + uuid.uuid4().hex[:8] + '.partial.mp4')
    original = probe(source)
    a = next(s for s in original['streams'] if s['codec_type'] == 'video')
    needs_encoding = not destination.exists()
    if not needs_encoding:
        existing = next(s for s in probe(destination)['streams'] if s['codec_type'] == 'video')
        needs_encoding = existing['nb_read_packets'] != a['nb_read_packets']
    if needs_encoding:
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-threads', '2',
                        '-i', str(source), '-map', '0:v:0', '-map', '0:a?',
                        '-vf', FILTER, '-filter_threads', '2', '-c:v', 'libx264',
                        '-preset', 'veryfast', '-crf', '18', '-threads', '2',
                        '-vsync', '0', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart',
                        str(partial)], check=True)
        partial.rename(destination)
    original, cropped = probe(source), probe(destination)
    a = next(s for s in original['streams'] if s['codec_type'] == 'video')
    b = next(s for s in cropped['streams'] if s['codec_type'] == 'video')
    assert (b['width'], b['height']) == (800, 600)
    assert a['nb_read_packets'] == b['nb_read_packets'], (source, a, b)
    assert a['r_frame_rate'] == b['r_frame_rate']
    assert abs(float(original['format']['duration']) - float(cropped['format']['duration'])) < .1
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    print('Verified:', source.parent.name, source.name, flush=True)
    return {'original': str(source), 'cropped': str(destination), 'metadata': cropped,
            'original_size': before.st_size, 'original_mtime_ns': before.st_mtime_ns}

if __name__ == '__main__':
    sources = sorted(ROOT.glob('*/*.mp4'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(process, sources))
    manifest = {'inner_floor_corners_tl_tr_br_bl': [[258,90],[1000,78],[1020,617],[275,633]],
                'arena_width_m': 4.0, 'arena_height_m': 3.0, 'arena_area_m2': 12.0,
                'source_resolution': [1280,720], 'output_resolution': [800,600],
                'filter': FILTER, 'note': 'Manually selected floor corners, slightly inset to exclude walls. '
                'Arena represents approximately 12 square metres as specified by the user, using 4 x 3 m dimensions from existing corners.json.',
                'videos': results}
    Path(__file__).with_name('crop_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('All', len(results), 'cropped copies verified.', flush=True)
