"""
download_model.py

Simple helper to download a model file from a URL and save it to models/emotion_model.h5.
Usage (run locally where internet is available):

python download_model.py --source <direct_download_url>

If you want to fetch from Hugging Face, you can use the 'resolve' URL for the file.
Example:
python download_model.py --source "https://huggingface.co/geeknix/emotion-reg/resolve/main/emotion_model.h5"

Notes:
- This environment where the zip was created may not be able to download large files; run this on your machine.
"""
import argparse, requests, os, sys, shutil

def download_file(url, dest_path, chunk_size=8192):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    done = int(50 * downloaded / total) if total else 0
                    sys.stdout.write(f"\r[{'='*done}{' '*(50-done)}] {downloaded}/{total} bytes")
                    sys.stdout.flush()
    print("\nDownload complete")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='Direct URL to .h5 model file (resolve/raw link)')
    parser.add_argument('--out', default='models/emotion_model.h5', help='Output path')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print("Downloading from", args.source)
    try:
        download_file(args.source, args.out)
        print("Saved to", args.out)
    except Exception as e:
        print("Failed to download:", e)

if __name__ == '__main__':
    main()
