"""Download/verify pinned public ONNX files. Never overwrites an existing file."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request


def manifest():
    return json.loads(Path(__file__).with_name('manifest.json').read_text())


def verify(path, row):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes']:
        raise ValueError('Model missing, symlink, or wrong size: ' + path.name)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != row['sha256']:
        raise ValueError('Model checksum mismatch: ' + path.name)
    return path


def model_record(path, kind):
    rows = [r for r in manifest()['files'] if r['file'] == Path(path).name and r['kind'] == kind]
    if len(rows) != 1:
        raise ValueError('Use the pinned model filename for ' + kind)
    verify(path, rows[0])
    return rows[0]


def download(folder, row):
    folder = Path(folder)
    name = row['file']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Unsafe model filename')
    if not row['url'].startswith('https://github.com/OSS-TEST-GROUP/katech-autoware-sils/releases/download/'):
        raise ValueError('Unexpected model release URL')
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if target.exists() or target.is_symlink():
        return verify(target, row)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.model-', suffix='.part', delete=False) as out:
            temporary = Path(out.name)
            request = urllib.request.Request(row['url'], headers={'User-Agent': 'katech-korean-tl/0.1'})
            total = 0
            with urllib.request.urlopen(request, timeout=60) as response:
                if not response.geturl().startswith('https://'):
                    raise ValueError('Non-HTTPS download redirect')
                while True:
                    data = response.read(1024 * 1024)
                    if not data:
                        break
                    total += len(data)
                    if total > row['bytes']:
                        raise ValueError('Oversized model response')
                    out.write(data)
        verify(temporary, row)
        # No replacement, including if another process created target during download.
        os.link(temporary, target)
        return verify(target, row)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('download', 'verify'))
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    for row in manifest()['files']:
        path = download(args.directory, row) if args.action == 'download' else verify(args.directory / row['file'], row)
        print(json.dumps({'file': path.name, 'sha256': row['sha256'], 'verified': True}))


if __name__ == '__main__':
    main()
